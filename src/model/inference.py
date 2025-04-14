import os

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser




def summarize(llm, msg, vector_store):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    PROMPT = ""
    with open(os.path.join("src", "model", "summarize.txt"), "r") as f:
        PROMPT = f.read()
    retriever = vector_store.get_retriever()
    prompt = PromptTemplate.from_template(PROMPT)

    lang_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return lang_chain.invoke(msg)
