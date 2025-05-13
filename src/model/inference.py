import os
import toml
import base64
import chainlit as cl
from pydantic import BaseModel, Field

from src.model.model import LanguageClassifier

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser




def classify_image(llm, image):
    class ImageOutput(BaseModel):
        is_valid: str = Field(description="Image classification result.")

    with open(os.path.join("src", "model", "prompts", "image_classification.toml"), "r") as f:
        prompts = toml.load(f)
        system_prompt = prompts["prompts"]["system"]
        user_prompt = prompts["prompts"]["user"]
        

    parser = JsonOutputParser(pydantic_object=ImageOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,{image_data}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,{wrong_empty}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,{wrong_text}"},
            },
            {
                "type": "text",
                "text": user_prompt,
            },
        ]),
    ]).partial(is_valid=parser.get_format_instructions())

    with open(os.path.join("src", "model", "prompts", "wrong_empty.jpg"), "rb") as f:
        wrong_empty = base64.b64encode(f.read()).decode("utf-8")
    with open(os.path.join("src", "model", "prompts", "wrong_text.jpg"), "rb") as f:
        wrong_text = base64.b64encode(f.read()).decode("utf-8")

    classification_chain = (
        prompt
        | llm
        | parser
    )
    res = classification_chain.invoke({"image_data": image,
                                        "wrong_empty": wrong_empty,
                                        "wrong_text": wrong_text})["is_valid"]
    return res.lower() == "yes"



def rewrite_query(query, llm, history, history_length=10):
    class RewriteOutput(BaseModel):
        rewritten_query: str = Field(description="Rewritten query based on the user query and conversation history.")
    
    with open(os.path.join("src", "model", "prompts", "rewrite_query.toml"), "r") as f:
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

    language_classifier = cl.user_session.get("language_classifier")

    rewrite_chain = (
        prompt
        | llm
        | parser
    )

    MAX_TRIES = 3
    for _ in range(MAX_TRIES):
        try:
            rewritten = rewrite_chain.invoke({
                "history": history_string,
                "user_query": query
            })["rewritten_query"]
            if language_classifier.classify(rewritten) == "it":
                return rewritten
            else:
                print("Rewritten query is not in Italian. Retrying...")
                continue
        except Exception as e:
            print(f"Error during query rewriting: {e}")
            continue
    return query



def translate(query, llm, source_language="it"):
    with open(os.path.join("src", "model", "prompts", "translation.toml"), "r") as f:
        prompts = toml.load(f)
        system_prompt = prompts["prompts"]["system"]
        user_prompt = prompts["prompts"]["user"]
    
    prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt),
    ])
    
    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    source_language = "italian" if source_language == "it" else "english"
    translated = rag_chain.invoke({"source_text": query,
                                "source_language": source_language})
    return translated


def summarize(query, llm, vector_store, search_image=False):
    def format_docs(docs):
        context = ""
        for doc in docs:
            if doc.metadata.get("type") == "text":
                context += f"{doc.page_content}\n\n"
            elif doc.metadata.get("type") == "image":
                context += f"{doc.metadata.get('img_caption')}\n\n"
        return context


    with open(os.path.join("src", "model", "prompts", "conversation.toml"), "r") as f:
        prompts = toml.load(f)
        system_prompt = prompts["prompts"]["system"]
        user_prompt = prompts["prompts"]["user"]
    
    prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt),
        ])

    language_classifier = cl.user_session.get("language_classifier")
    if language_classifier.classify(query) != "en":
        query = translate(query, llm, "it")

    filter = {"type": "image" if search_image else "text"}
    retriever = vector_store.get_retriever(filter=filter)
    docs = retriever.invoke(query)
    context = format_docs(docs)

    if search_image:
        image = docs[0].page_content
    
    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke({"context": context,
                             "query": query}), image if search_image else None
    text_answer = answer[0]
    image_answer = answer[1]

    if language_classifier.classify(text_answer) != "it":
        text_answer = translate(text_answer, llm, "en")
    return text_answer, image_answer
