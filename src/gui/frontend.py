import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import os.path as osp
import chainlit as cl

from src.text.indexing import get_empty_vector_store
from src.model.model import HuggingFaceEmbeddingModel
from src.model.inference import extract_file, index_files, summarize





@cl.on_chat_start
async def main():
    embed_model = HuggingFaceEmbeddingModel("mixedbread-ai/mxbai-embed-large-v1").model
    vector_store = get_empty_vector_store(embed_model)
    
    await cl.Message(content="Send a file!").send()
    
    cl.user_session.set("embed_model", embed_model)
    cl.user_session.set("vector_store", vector_store)


@cl.on_message
async def main(message: cl.Message):
    embed_model = cl.user_session.get("embed_model")
    vector_store = cl.user_session.get("vector_store")

    if len(message.elements) > 0:
        total_text = []
        for element in message.elements:
            file_path = element.path
            text = await cl.make_async(extract_file)(file_path, 10)
            total_text.extend(text)
        total_text = total_text[0]
        await cl.make_async(index_files)(total_text, embed_model)

    answer = await cl.make_async(summarize)(message.content, vector_store)
    await cl.Message(content=answer).send()
