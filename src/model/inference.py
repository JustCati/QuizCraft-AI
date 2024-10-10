import os
import tempfile
import subprocess

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.utils.pdf2img2pdf import convert_img2pdf




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


    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]


def extract_text_from_image(model, img, prompt):
    img = convert_to_base64(img)

    chain = prompt_func | model | StrOutputParser()
    query_chain = chain.invoke({"text": prompt, "image": img})
    return query_chain
