import os
import sys
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

from src.model.inference import summarize
from src.text.indexing import VectorStore
from src.utils.extract import extract_text
from src.model.model import HuggingFaceEmbeddingModel, OllamaLanguageModel




@cl.on_settings_update
async def setup_agent(settings):
    llm = OllamaLanguageModel(settings["Model"], settings["Temperature"]).get()
    cl.user_session.set("llm", llm)

    if settings["Role"] == "Explain/Summarize":
        cl.user_session.set(
            "message_history", 
            [{
                "role": "system",
                "content": "You are a professor that will answer and explain the requested argument to a student."
            }]
        )
    else:
        cl.user_session.set(
            "message_history", 
            [{
                "role": "system",
                "content": "You are a professor that will create a questionnaire based on the requested argument."
            }]
        )


@cl.on_chat_start
async def main():
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Gwen2.5 Model",
                values=["qwen2.5:32b", "qwen2.5:7b"],
                initial_index=0,
            ),
            Select(
                id="Role",
                label="Role",
                values=["Explain/Summarize", "Questionnaire"],
                initial_index=0,
            ),
            Switch(id="Streaming", label="Stream Tokens", initial=True),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=0,
                min=0,
                max=1,
                step=0.1,
            ),
        ]
    ).send()

    embed_model = HuggingFaceEmbeddingModel("mixedbread-ai/mxbai-embed-large-v1").model
    vector_store = VectorStore(embed_model)
    cl.user_session.set("vector_store", vector_store)

    uploaded = None
    while uploaded == None:
         uploaded = await cl.AskFileMessage(
            content="Please upload a text file to begin!", 
            accept=["application/pdf", "image/png", "image/jpg"],
            max_files=20,
            max_size_mb=20
        ).send()

    texts = []
    with ThreadPoolExecutor(max_workers = 2) as executor:
        extracted = executor.map(extract_text, [file.path for file in uploaded], [1]*len(uploaded))
        for text in extracted:
            texts.extend(text)
    await cl.make_async(vector_store.index_files)(texts)
    await setup_agent(settings)



@cl.on_chat_end
async def cleanup():
    llm: OllamaLanguageModel = cl.user_session.get("llm")
    llm.stop()
    torch.cuda.empty_cache()



@cl.on_message
async def main(message: cl.Message):
    llm: OllamaLanguageModel = cl.user_session.get("llm")
    vector_store: VectorStore = cl.user_session.get("vector_store")

    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    if len(message.elements) > 0:
        total_text = []
        for element in message.elements:
            file_path = element.path
            text = await cl.make_async(extract_text)(file_path, 10)
            total_text.extend(text)
        total_text = total_text[0]
        await cl.make_async(vector_store.index_files)(total_text)

    answer = await cl.make_async(summarize)(llm, message.content, vector_store)

    msg = cl.Message(content="")
    for token in answer:
        await msg.stream_token(token)
    await msg.send()
