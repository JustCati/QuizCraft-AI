from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker




def get_semantic_doc(text, embeddings_model):
    splitter = SemanticChunker(embeddings=embeddings_model, breakpoint_threshold_type="percentile")
    docs = splitter.create_documents(text)
    docs = splitter.split_documents(docs)
    return docs


def get_empty_vector_store(embeddings_model):
    vector_store = Chroma(
        collection_name="vector_store",
        embedding_function=embeddings_model
    )
    return vector_store


def get_retriever(vector_store: Chroma):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
