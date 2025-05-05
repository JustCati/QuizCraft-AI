import os
import re
import json
import argparse
from io import StringIO
from hashlib import sha256

from src.utils.extract import extract_from_pdf
from src.utils.regex import remove_images, remove_links
from langchain_text_splitters import MarkdownHeaderTextSplitter

from pydantic import BaseModel, Field
from src.model.model import OllamaLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser




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

    i = 0
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
                
                data = {
                    "content": chunk.page_content,
                    "hash": calculate_hash(chunk.page_content),
                    "query": ""
                }
                dataset_dict[i] = data
                i += 1

    with open(os.path.join(output_path, "dataset.json"), "w") as f:
        json.dump(dataset_dict, f, indent=4)
        
        
        
def generate_query(llm, query):
    system_prompt = '''
        # Role
        
        You are an expert AI capable of generating a query from a a given text chunk. The final goal is to generate the right query so that, if this query is used to search for the text chunk, the text chunk will be retrieved.
        
        # Input

            - Chunk: A chunk of text that is a part of a larger document. The chunk may contain multiple paragraphs and it is formatted with markdown headers.

        # Instruction
        
            1. Read the chunk carefully.
            2. Generate a query that is relevant to the content of the chunk.
            3. The query should be a single sentence that captures the main idea of the chunk.
            4. Generate the query in italian.
            5. The query should be in the form of a question.
            6. If the chunk does not contain any relevant explanation of a concept, generate a query that is a single word that captures the main idea of the chunk.
    '''

    user_prompt = '''
        # Input:
        
        Chunk:
        {chunk}

        Output:
        {generated_query}
    '''

    class Query(BaseModel):
        generated_query: str = Field(description="The generated query")

    parser = JsonOutputParser(pydantic_object=Query)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt)
        ]
    ).partial(generated_query=parser.get_format_instructions())

    chain = (
        prompt
        | llm
        | parser
    )
    result = chain.invoke({"chunk": query})["generated_query"]
    return result



def generate_queries(args):
    input_path = args.output_path

    if input_path.endswith("/"):
        input_path = input_path[:-1]
    output_path = os.path.dirname(input_path)
    llm = OllamaLanguageModel("gemma3:27b-it-qat", 0.4).get().model
    
    dataset_path = os.path.join(output_path, "dataset_with_query.json")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(output_path, "dataset.json")
        if not os.path.exists(dataset_path):
            print(f"Dataset file not found at {dataset_path}. Please run the chunk generation first.")
            return


    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    for k, v in dataset.items():
        chunk = v["content"]
        query = v["query"]

        if query:
            print(f"Skipping query generation for chunk {k} as it already has a query.")
            continue
        generated_query = generate_query(llm=llm, query=chunk)
        dataset[k]["query"] = generated_query

        with open(os.path.join(output_path, "dataset_with_query.json"), "w") as f:
            json.dump(dataset, f, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset from PDF files.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input directory containing PDF files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory for text files.")
    args = parser.parse_args()
    
    extract_text(args)
    generate_chunks(args)
    generate_queries(args)
