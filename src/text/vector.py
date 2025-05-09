import os
import shutil
import base64
from PIL import Image
from io import BytesIO
from io import StringIO
from hashlib import sha256
from tempfile import TemporaryDirectory

import chainlit as cl
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.model.inference import classify_image



class VectorStore():
    def __init__(self, embed_model, threshold=0.5):
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
        
        self.img_dir = os.path.join("data", "images")
        os.makedirs(self.img_dir, exist_ok=True)
        cl.user_session.set("img_dir", self.img_dir)


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


    def __index_image(self, image, caption=""):
        most_similar_img_score = self.vector_store.similarity_search_by_image_with_relevance_score(
            image,
            k=1,
            filter={"type": "image"},
        )
        most_similar_img_score = most_similar_img_score[0][1] if len(most_similar_img_score) > 0 else 0.0
        
        image_data = base64.b64encode(open(image, "rb").read()).decode("utf-8")
        img_hash = self.__calculate_hash(image_data)
    
        if not classify_image(cl.user_session.get("llm"), image):
            print(f"Image {image} is not a valid image.")
            return

        if not self.vector_store.get_by_ids([img_hash]) or most_similar_img_score < self.threshold:
            self.vector_store.add_images(
                [image],
                metadatas=[{
                    "type": "image",
                    "img_caption": caption,
                    }],
                ids=[img_hash],
            )
            with open(image, "rb") as fp:
                img_to_save = Image.open(fp)
                img_to_save.save(os.path.join(self.img_dir, f"{img_hash}.jpg"))
        else:
            print(f"Found similar or equal image with score {most_similar_img_score} and hash {img_hash}, skipping indexing.")


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
                data = image["image"]
                caption = image["caption"]
                
                image = Image.open(BytesIO(base64.b64decode(data)))
                image.save(f"{tempfile.name}/image.jpg")
                self.__index_image(f"{tempfile.name}/image.jpg", caption)
            tempfile.cleanup()
        print("Files indexed.")


    def get_retriever(self):
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
