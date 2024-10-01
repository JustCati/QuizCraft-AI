import subprocess
from src.utils.pdf2img import convert_to_base64

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser




def runModel(name):
    subprocess.Popen(['ollama', 'run', name], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def stopModel(name):
    subprocess.Popen(['ollama', 'stop', name], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)



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


# "llava:34b-v1.6-q3_K_M"
def getModel(name, temperature = 0.0, num_predict = 512):
    return ChatOllama(model=name, 
                       temperature=temperature,
                       num_predict=num_predict)


def extract_text_from_image(model, img, prompt):
    img = convert_to_base64(img)

    chain = prompt_func | model | StrOutputParser()
    query_chain = chain.invoke({"text": prompt, "image": img})
    return query_chain
