import os
from PIL import Image
from tqdm import tqdm
from tempfile import NamedTemporaryFile

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.utils.pdf2img import convert_to_img, convert_to_base64




def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]




def model_extract(img, prompt):
    img = convert_to_base64(img)

    model = ChatOllama(model="llava:34b-v1.6-q3_K_M", 
                       temperature=0.0,
                       num_predict=512)

    chain = prompt_func | model | StrOutputParser()
    query_chain = chain.invoke({"text": prompt, "image": img})
    return query_chain




def extract(file_path, prompt):
    out_path = file_path.replace("RAW", "EXTRACTED")
    out_path = out_path.replace(".pdf", ".txt").replace(".jpg", ".txt")

    if os.path.exists(out_path):
        print(f"File {out_path} already exists. Skipping...")
        return

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    if file_path.endswith(".pdf"):
        dir = convert_to_img(file_path)
    else:
        dir = [file_path]

    text = []
    for img in tqdm(dir):
        if type(img) == str:
            img = Image.open(file_path)
        output_text = model_extract(img, prompt)
        text.append(output_text)

    #! FORMAT TEXT HERE

    print(f"Extracted text: {text}")
    # with open(out_path, "a") as f:
    #     f.write("\n".join(text))
    #     f.write("\n")
    return
