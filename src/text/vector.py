import shutil
from io import StringIO
from hashlib import sha256

from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter



class VectorStore():
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3")
            ],
        )

        self.vector_store = Chroma(
            embedding_function=self.embed_model,
            persist_directory="data",
            collection_name="documents",
        )


    def clean(self):
        shutil.rmtree("data", ignore_errors=True)
        print("Vector store cleaned.")


    def __calculate_hash(self, text: str) -> str:
        BUFFERSIZE = 65536
        hasher = sha256()

        with StringIO(text) as f:
            buffer = f.read(BUFFERSIZE)
            while len(buffer) > 0:
                hasher.update(buffer.encode())
                buffer = f.read(BUFFERSIZE)
        return hasher.hexdigest()


    def __get_chunks(self, text):
        chunks = self.splitter.split_text(text)
        return chunks


    def add(self, texts) -> None:
        texts = [texts] if isinstance(texts, str) else texts

        for text in texts:
            chunks = self.__get_chunks(text)
            for chunk in chunks:
                chunk_hash = self.__calculate_hash(chunk.page_content)
                if not self.vector_store.get_by_ids([chunk_hash]):
                    self.vector_store.add_documents(
                        [chunk],
                        ids=[chunk_hash],
                    )
                else:
                    print(f"Document with hash {chunk_hash} already exists in the vector store.")
        print("Files indexed.")


    def get_retriever(self):
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
