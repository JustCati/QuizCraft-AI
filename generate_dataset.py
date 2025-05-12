import os
import re
import json
import argparse
from io import StringIO
from hashlib import sha256

from src.utils.extract import extract_from_pdf
from src.utils.regex import remove_images, remove_links
from langchain_text_splitters import MarkdownHeaderTextSplitter




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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset from PDF files.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input directory containing PDF files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory for text files.")
    args = parser.parse_args()
    
    extract_text(args)
    generate_chunks(args)
