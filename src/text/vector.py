import shutil
import base64
from PIL import Image
from io import BytesIO
from io import StringIO
from hashlib import sha256
from tempfile import TemporaryDirectory

from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter



class VectorStore():
    def __init__(self, embed_model, threshold=0.9):
        self.threshold = threshold
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


    def __calculate_hash(self, text):
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
        for chunk in chunks:
            chunk.metadata["type"] = "text"
        return chunks


    def __index_text(self, text):
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


    def __index_image(self, image):
        most_similar_img_score = self.vector_store.similarity_search_by_image_with_relevance_score(
            image,
            k=1,
            filter={"type": "image"},
        )
        most_similar_img_score = most_similar_img_score[0][1] if len(most_similar_img_score) > 0 else 0.0
        
        if most_similar_img_score < self.threshold:
            self.vector_store.add_images(
                [image],
                metadatas=[{"type": "image"}],
                ids=[self.__calculate_hash(image)],
            )
        else:
            print(f"Found similar image with score {most_similar_img_score}, skipping indexing.")


    def add(self, texts, images=None):
        texts = [texts] if isinstance(texts, str) else texts
        images = [images] if isinstance(images, str) else images
        if images is None:
            images = []

        for text in texts:
            self.__index_text(text)

        if len(images) > 0:
            tempfile = TemporaryDirectory()
            for image in images:
                image = Image.open(BytesIO(base64.b64decode(image)))
                image.save(f"{tempfile.name}/image.jpg")
                self.__index_image(f"{tempfile.name}/image.jpg")
            tempfile.cleanup()
        print("Files indexed.")


    def get_retriever(self):
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
