import os
import toml
import base64
from pydantic import BaseModel, Field
from tempfile import TemporaryDirectory

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser



def ensure_language_consistency(func):
    def wrapper(query, llm, *args, **kwargs):
        wrong_query = kwargs.pop("wrongly_rewritten_query", "")

        MAX_RETRIES = 3
        origin_language = classify_language(llm, query).lower()

        for i in range(MAX_RETRIES):
            rewritten_query = func(query, llm, **kwargs, wrongly_rewritten_query=wrong_query)
            new_language = classify_language(llm, rewritten_query).lower()

            if new_language == origin_language:
                return rewritten_query

            wrong_query = rewritten_query
            print(f"Language mismatch detected. Retrying... (Attempt {i + 1}/{MAX_RETRIES})")
        return rewritten_query
    return wrapper



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


def classify_language(llm, msg):
    class LanguageOutput(BaseModel):
        language: str = Field(description="Language of the input text.")

    with open(os.path.join("src", "model", "prompts", "language_classification.toml"), "r") as f:
        prompts = toml.load(f)
        system_prompt = prompts["prompts"]["system"]
        user_prompt = prompts["prompts"]["user"]

    parser = JsonOutputParser(pydantic_object=LanguageOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ]).partial(language=parser.get_format_instructions())

    classification_chain = (
        prompt
        | llm
        | parser
    )
    return classification_chain.invoke({"text": msg})["language"]



@ensure_language_consistency
def rewrite_query(query, llm, history, history_length=10, wrongly_rewritten_query=""):
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

    rewrite_chain = (
        prompt
        | llm
        | parser
    )

    return rewrite_chain.invoke({
        "history": history_string, 
        "wrongly_rewritten_query": wrongly_rewritten_query,
        "user_query": query
    })["rewritten_query"]



@ensure_language_consistency
def summarize(query, llm, vector_store, search_image=False, wrongly_rewritten_query=""):
    def format_docs(docs):
        context = ""
        for doc in docs:
            if doc.metadata.get("type") == "text":
                context += f"Text: {doc.page_content}\n\n"
            elif doc.metadata.get("type") == "image":
                context += f"Image: {doc.metadata.get('img_caption')}\n\n"
        return context


    with open(os.path.join("src", "model", "prompts", "conversation.toml"), "r") as f:
        prompts = toml.load(f)
        system_prompt = prompts["prompts"]["system"]
        user_prompt = prompts["prompts"]["user"]
    
    prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt),
        ])

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

    return rag_chain.invoke({"context": context,
                             "query": query,
                             "wrong_output": wrongly_rewritten_query}), image if search_image else None
