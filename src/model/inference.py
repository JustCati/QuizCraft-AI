import os
import toml
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser



def ensure_language_consistency(func):
    def wrapper(query, llm, *args, **kwargs):
        print(type(query))
        wrong_query = kwargs.pop("wrongly_rewritten_query", "")
        print(kwargs)

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
def summarize(query, llm, vector_store, wrongly_rewritten_query=""):
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
        {"context": retriever | format_docs, "query": RunnablePassthrough(), "wrong_output": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(query, rewritten_query=wrongly_rewritten_query)
