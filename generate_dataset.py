import os
import re
import json
import random
import argparse
from io import StringIO
from hashlib import sha256
from dotenv import load_dotenv

from src.utils.extract import extract_from_pdf
from src.utils.regex import remove_images, remove_links
from langchain_text_splitters import MarkdownHeaderTextSplitter

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig
from deepeval.models.base_model import DeepEvalBaseLLM

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
from src.model.model import OllamaLanguageModel, LanguageClassifier

from src.model.inference import translate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter





class SynthetizerModel(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model_ref = model
        self.model = (
            RunnablePassthrough()
            | self.model_ref
            | StrOutputParser()
        )

    def load_model(self):
        return self.model

    def get_model_name(self):
        return "gemini-2.0-flash"

    def generate(self, prompt):
        model = self.load_model()
        return model.invoke(prompt)

    async def a_generate(self, prompt):
        model = self.load_model()
        return await model.invoke(prompt)


def calculate_hash(text):
    BUFFERSIZE = 65536
    hasher = sha256()

    with StringIO(text) as f:
        buffer = f.read(BUFFERSIZE)
        while len(buffer) > 0:
            hasher.update(buffer.encode())
            buffer = f.read(BUFFERSIZE)
    return hasher.hexdigest()




def extract_text(args):
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    
    for folder in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for i, file in enumerate(os.listdir(folder_path)):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(folder_path, file)
                md, _ = extract_from_pdf(pdf_path)
                
                md = remove_images(md)
                md = remove_links(md)
                with open(os.path.join(output_path, f"{folder}_{i}.md"), "w", encoding="utf-8") as f:
                    f.write(md)
            else:
                print(f"Skipping non-PDF file: {file}")



def generate_chunks(args):
    input_path = args.output_path

    if input_path.endswith("/"):
        input_path = input_path[:-1]
    output_path = os.path.dirname(input_path)
    os.makedirs(output_path, exist_ok=True)

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on = [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3")
                ],
        strip_headers=False
        )

    dataset_dict = dict()
    for file in os.listdir(input_path):
        if file.endswith(".md"):
            with open(os.path.join(input_path, file), "r") as f:
                md = f.read()
            chunks = splitter.split_text(md)

            buffer = ""
            for chunk in chunks:
                if re.sub(r"^# [A-Za-z0-9 \\n?()&\-#,;:_\".!=@\%/'èìòàùé]+|#", "", chunk.page_content).strip() == "":
                    buffer += chunk.page_content
                    continue

                if buffer and len(buffer) > 100:
                    chunk.page_content = buffer + "\n" + chunk.page_content
                    buffer = ""

                if len(chunk.page_content) < 100:
                    continue

                hash = calculate_hash(chunk.page_content)
                if hash in dataset_dict:
                    print(f"Duplicate chunk found: {hash}. Skipping.")
                    continue
                
                data = {
                    "content": chunk.page_content,
                    "query": ""
                }
                dataset_dict[hash] = data

    with open(os.path.join(output_path, "dataset.json"), "w") as f:
        json.dump(dataset_dict, f, indent=4)



def generate_queries_goldens(args):
    NUM_TOTAL_CHUNKS = args.num_chunks

    load_dotenv(".env")
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    
    output_path = args.output_path
    if output_path.endswith("/"):
        output_path = output_path[:-1]
    output_path = os.path.dirname(output_path)
    data_path = os.path.join(args.output_path, "dataset.json")
    with open(data_path, "r") as f:
        data = json.load(f)

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.25,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    )

    llm = ChatGoogleGenerativeAI(
        model=args.synthesizer_model,
        temperature=0,
        rate_limiter=rate_limiter,
    )
    
    language_classifier = LanguageClassifier(model_name=args.language_classifier, device="cuda")
    local_model = OllamaLanguageModel(args.translater, 0.0).get().model
    model = SynthetizerModel(llm)

    styling_config = StylingConfig(
        input_format="Extracted text from university course slides written in Italian or English.",
        task="Generate a question that a student might ask about the text, in English. Do not cite the text verbatim, paraphrase when possible. Generate the question in English.",
        scenario="A student is preparing for an exam and wants to test their understanding of the material or want an explanation of the content.",
    )
    synthesizer = Synthesizer(model=model, 
                            async_mode=False,
                            max_concurrent=1,
                            styling_config=styling_config,)

    chunks_done = [key for key, elem in data.items() if elem["query"] != "" and elem.get("golden", "") != ""]
    num_done = len(chunks_done)
    num_remaining = NUM_TOTAL_CHUNKS - num_done
    print(f"Chunks done: {num_done}/{NUM_TOTAL_CHUNKS}")
    
    todo_keys = [key for key, elem in data.items() if elem["query"] == "" and elem.get("golden", "") == ""]
    
    random.seed(42)
    random.shuffle(todo_keys)

    contexts = [data[key]["content"] for key in todo_keys[:num_remaining]]


    for context in contexts:
        res = synthesizer.generate_goldens_from_contexts(contexts=[[context]], 
                                                        include_expected_output=True,
                                                        max_goldens_per_context=1)

        res = res[0]
        key = calculate_hash(res.context[0])
        input = res.input
        expected_output = res.expected_output

        if language_classifier.classify(input).lower() != "en":
            input = translate(input, local_model, "it")

        if language_classifier.classify(expected_output).lower() != "en":
            expected_output = translate(expected_output, local_model, "it")

        data[key] = {
            "query": input,
            "content": res.context[0],
            "golden": expected_output,
        }
        
        print(len([key for key, elem in data.items() if elem["query"] != "" and elem.get("golden", "") != ""]))
        
        with open(data_path, "w") as f:
            json.dump(data, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset from PDF files.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input directory containing PDF files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory for text files.")
    parser.add_argument("--num_chunks", type=int, default=150, help="Number of chunks to generate.")
    parser.add_argument("--language_classifier", type=str, default="qanastek/51-languages-classifier", help="Path to the language classifier model.")
    parser.add_argument("--translater", type=str, default="gemma3:27b-it-qat", help="Path to the translation model.")
    parser.add_argument("--synthesizer_model", type=str, default="gemini-2.0-flash", help="Name to the synthesizer model.")
    args = parser.parse_args()
    
    extract_text(args)
    generate_chunks(args)
    generate_queries_goldens(args)
