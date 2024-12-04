from io import StringIO
from hashlib import sha256

from langchain_postgres.vectorstores import PGVector
from langchain_experimental.text_splitter import SemanticChunker

from src.postgres.postgres import Postgres



class VectorStore():
    def __init__(self,
                 embed_model: str,
                 db: Postgres) -> None:

        self.db = db
        self.embed_model = embed_model
        self.splitter = SemanticChunker(embeddings=self.embed_model, breakpoint_threshold_type="percentile")

        self.vector_store = PGVector(
            embeddings=self.embed_model,
            collection_name="pdfs",
            connection="postgresql+psycopg://{}:{}@{}:{}/{}".format(
                self.db.env["POSTGRES_USER"],
                self.db.env["POSTGRES_PASSWORD"],
                self.db.env.get("POSTGRES_HOST", "localhost"),
                self.db.env.get("POSTGRES_PORT", "5432"),
                self.db.env["POSTGRES_DB"]
            ),
            use_jsonb=True
        )


    def __calculate_hash(self, text: str) -> str:
        BUFFERSIZE = 65536
        hasher = sha256()

        with StringIO(text) as f:
            buffer = f.read(BUFFERSIZE)
            while len(buffer) > 0:
                hasher.update(buffer.encode())
                buffer = f.read(BUFFERSIZE)
        return hasher.hexdigest()


    def __get_semantic_doc(self, text: str) -> list[dict[str, str]]:
        docs = self.splitter.create_documents(text)
        docs = self.splitter.split_documents(docs)
        return docs


    def add(self, texts: list[str] | str) -> None:
        texts = [texts] if isinstance(texts, str) else texts

        for text in texts:
            hash = self.__calculate_hash(text)
            if not self.db.does_file_exist(hash):
                self.db.save_file_to_db(hash)
                docs = self.__get_semantic_doc(text)
                self.vector_store.add_documents(docs)
        print("Files indexed.")


    def get_retriever(self):
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
