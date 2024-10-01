from PIL import Image

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.utils.pdf2img import convert_to_img, convert_to_base64




def extract_file(file, model, PROMPT):
    if file.path.endswith(".pdf"):
        dir = convert_to_img(file.path)
    else:
        dir = [file.path]

    text = []
    for img in dir:
        if type(img) == str:
            img = Image.open(file.path)
        output_text = extract_text_from_image(model, img, PROMPT)
        text.append(output_text)
    return text



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


def extract_text_from_image(model, img, prompt):
    img = convert_to_base64(img)

    chain = prompt_func | model | StrOutputParser()
    query_chain = chain.invoke({"text": prompt, "image": img})
    return query_chain
