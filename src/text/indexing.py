from uuid import uuid4
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker




def get_semantic_doc(text, embeddings_model):
    splitter = SemanticChunker(embeddings=embeddings_model, breakpoint_threshold_type="percentile")
    docs = splitter.create_documents([text])
    docs = splitter.split_documents(docs)
    return docs


def get_empty_vector_store(embeddings_model):
    vector_store = Chroma(
        collection_name="vector_store",
        embedding_model=embeddings_model
    )
    return vector_store


def add_to_vector_store(vector_store: Chroma, docs):
    uuids = [str(uuid4()) for _ in range(len(docs))] 
    vector_store.add_documents(documents=docs, ids=uuids)


def get_retriever(vector_store: Chroma):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
