import os
import toml
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser




def rewrite_query(query, history, llm, history_length = 10):
    class RewriteOutput(BaseModel):
        rewritten_query: str = Field(description="Rewritten query based on the user query and conversation history.")


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
