import os
import toml
from PIL import Image

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.utils.pdf2img2pdf import convert_to_base64




def summarize(llm, msg, vector_store):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    with open(os.path.join("src", "model", "prompts", "summarize.toml"), "r") as f:
        prompts = toml.load(f)
        system_prompt = prompts["prompts"]["system"]
        user_prompt = prompts["prompts"]["user"]

    retriever = vector_store.get_retriever()
    prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt),
        ])

    lang_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return lang_chain.invoke(msg)



def extract(llm, dir):
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

    with open(os.path.join("src", "model", "prompts", "ocr.txt"), "r") as f:
        ocr_prompt = f.read()

    file_extracted_text = ""
    files = sorted(os.listdir(dir))
    for i, image_file in enumerate(files):
        image_path = os.path.join(dir, image_file)
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            image_b64 = convert_to_base64(image)

            chain = prompt_func | llm | StrOutputParser()
            extracted = chain.invoke({"text": ocr_prompt, "image": image_b64})
            file_extracted_text += extracted
            file_extracted_text += "\n"
        print(f"Extracted text from {i+1} of {len(files)} images")
    return file_extracted_text
