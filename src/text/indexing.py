from uuid import uuid4
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker




def get_semantic_doc(text, embeddings_model):
    splitter = SemanticChunker(embeddings=embeddings_model, breakpoint_threshold_type="percentile")
    docs = splitter.create_documents([text])
    docs = splitter.split_documents(docs)
    return docs
