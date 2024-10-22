import os
import tempfile
import subprocess

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.model.model import OllamaLanguageModel

from src.utils.pdf2img2pdf import convert_img2pdf
from src.text.indexing import (get_semantic_doc, 
                               get_empty_vector_store,
                               get_retriever)


# qwen2.5:32b-instruct-q3_K_M
# mxbai-embed-large


def extract_file(file, batch_multiplier=2):
    text = []
    with tempfile.TemporaryDirectory() as dir:
        if file.endswith(".png") or file.endswith(".jpg"):
            images = [file]
            pdf_path = convert_img2pdf(images, os.path.join(dir, "temp.pdf"))
            file = pdf_path

        process = subprocess.run(["marker_single", "--batch_multiplier", str(batch_multiplier), file, dir], stdout=subprocess.PIPE)
        outFolder = process.stdout.decode("utf-8").strip().split(" ")[-2]

        path = os.path.join(dir, outFolder)
        for file in os.listdir(path):
            if file.endswith(".md"):
                with open(os.path.join(path, file), "r") as f:
                    text.append(f.read())
    return text


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


async def index_files(files_text, embed_model):
    vector_store = get_empty_vector_store(embed_model)
    for text in files_text:
        docs = get_semantic_doc(text, embed_model)
        await vector_store.aadd_documents(docs)


def summarize(msg, vector_store):
    PROMPT = ""
    with open(os.path.join("src", "model", "prompt.txt"), "r") as f:
        PROMPT = f.read()
    with OllamaLanguageModel("qwen2.5:32b-instruct-q3_K_M") as llm:
        retriever = get_retriever(vector_store)
        prompt = PromptTemplate.from_template(PROMPT)

        lang_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return lang_chain.invoke(msg)