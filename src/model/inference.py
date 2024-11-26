import os

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from src.model.model import OllamaLanguageModel


# qwen2.5:32b-instruct-q3_K_M
# mxbai-embed-large


def summarize(msg, vector_store):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    PROMPT = ""
    with open(os.path.join("src", "model", "prompt.txt"), "r") as f:
        PROMPT = f.read()
    with OllamaLanguageModel("qwen2.5:32b-instruct-q3_K_M") as llm:
        retriever = vector_store.get_retriever()
        prompt = PromptTemplate.from_template(PROMPT)

        lang_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return lang_chain.invoke(msg)
