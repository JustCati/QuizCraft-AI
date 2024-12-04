from langchain_chroma import Chroma
from langchain.text_splitter import MarkdownTextSplitter
from langchain_experimental.text_splitter import SemanticChunker




class VectorStore():
    def __init__(self, embed_model: str, chunker: str = "semantic") -> None:
        self.embed_model = embed_model
        self.vector_store = Chroma(
                    collection_name = "documents",
                    embedding_function = self.embed_model
                )
        if chunker == "semantic":
            self.splitter = SemanticChunker(embeddings=self.embed_model, breakpoint_threshold_type="percentile")
        elif chunker == "markdown":
            self.splitter = MarkdownTextSplitter(embeddings=self.embed_model)
        else:
            raise ValueError(f"Invalid chunker: {chunker}")

    def __get_semantic_doc(self, text: str) -> list[dict[str, str]]:
        docs = self.splitter.create_documents(text)
        docs = self.splitter.split_documents(docs)
        return docs

    def index_files(self, texts: list[str]) -> None:
        for text in texts:
            docs = self.__get_semantic_doc(text)
            self.vector_store.add_documents(docs)
        print("Files indexed.")

    def get_retriever(self):
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
