import os
import toml
from PIL import Image
from tqdm import tqdm
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from src.utils.pdf2img2pdf import convert_to_base64




def rewrite_query(query, history, llm, history_length = 10):
    class RewriteOutput(BaseModel):
        rewritten_query: str = Field(description="Rewritten query based on the user query and conversation history.")

    with open(os.path.join("src", "model", "prompts", "chat_history.toml"), "r") as f:
        prompts = toml.load(f)
        system_prompt = prompts["prompts"]["system"]
        user_prompt = prompts["prompts"]["user"]

    parser = JsonOutputParser(pydantic_object=RewriteOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ]).partial(rewritten_query=parser.get_format_instructions())

    history_string = ""
    for i, message in enumerate(history):
        if i <= history_length:
            if isinstance(message, AIMessage):
                history_string += f"Assistant: {message.content}\n"
            if isinstance(message, HumanMessage):
                history_string += f"User: {message.content}\n"

    rewrite_chain = (
        prompt
        | llm
        | parser
    )
    return rewrite_chain.invoke({"history": history_string, "user_query": query})["rewritten_query"]





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

    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(msg)



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

    with open(os.path.join("src", "model", "prompts", "ocr.toml"), "r") as f:
        ocr_prompt = toml.load(f)["prompts"]["system"]

    file_extracted_text = ""
    files = sorted(os.listdir(dir))
    for image_file in tqdm((files)):
        image_path = os.path.join(dir, image_file)
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            image_b64 = convert_to_base64(image)

            chain = prompt_func | llm | StrOutputParser()
            extracted = chain.invoke({"text": ocr_prompt, "image": image_b64})
            file_extracted_text += extracted
            file_extracted_text += "\n"
    return file_extracted_text
