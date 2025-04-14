import os
import sys
import chainlit as cl
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.text.vector import VectorStore
from src.model.inference import summarize
from src.postgres.postgres import Postgres
from src.utils.extract import extract_text
from src.model.model import OllamaLanguageModel
from src.gui.utils import create_settings, create_vector_store, set_role, load_llm

import warnings
warnings.filterwarnings("ignore")





async def send_message(text: str) -> None:
    stream = cl.user_session.get("stream_tokens")
    if stream:
        msg = cl.Message(content="")
        for token in text:
            await msg.stream_token(token)
    else:
        msg = cl.Message(content=text)
    await msg.send()


@cl.on_settings_update
async def setup_agent(settings):
    # Load Model
    model_name = cl.user_session.get("model_name")
    if model_name is None or model_name != settings["Model"]:
        if model_name != settings["Model"]:
            cleanup()
        llm = await cl.make_async(load_llm)(settings)
        cl.user_session.set("llm", llm)

    # Check if loading knowledge base or messaging
    modality = cl.user_session.get("modality")
    if modality != settings["Modality"]:
        cl.user_session.set("modality", settings["Modality"])

    if modality == "message":
        set_role(settings)

    # Streaming tokens
    cl.user_session.set("stream_tokens", settings["Streaming"])
    print("Agent setup complete.")


@cl.on_chat_end
def cleanup():
    # Clean up the database and vector store
    vector_store: VectorStore = cl.user_session.get("vector_store")
    if vector_store is not None:
        vector_store.db.stop()
        print("Database cleanup complete.")

    # Clean up the language model
    llm = cl.user_session.get("llm")
    if llm is not None:
        llm.stop()
        print("Agent cleanup complete.")


@cl.on_chat_start
async def main():
    settings = await create_settings()

    env_file = os.path.join(os.path.dirname(__file__), "src", "postgres", ".env")
    docker_compose = env_file.replace(".env", "docker-compose.yml")
    db = await cl.make_async(Postgres)(docker_compose, env_file)

    vector_store: VectorStore = await create_vector_store(db)
    cl.user_session.set("vector_store", vector_store)
    await setup_agent(settings)

    res = await cl.AskActionMessage(
        content="",
        actions=[
            cl.Action(name="Preload Knowledge", payload={"value": "preload"}, label="📝 Preload Knowledge"),
            cl.Action(name="Message", payload={"value": "message"}, label="💬 Message"),
        ],
    ).send()

    if res and res.get("payload").get("value") == "preload":
        cl.user_session.set("modality", res.get("payload").get("value"))

        uploaded = None
        while uploaded == None:
             uploaded = await cl.AskFileMessage(
                content="Please upload one or multiple file/s (PDF, PNG, JPG) [max 20 files, 20MB each]", 
                accept=["application/pdf", "image/png", "image/jpg", "image/jpeg"],
                max_files=20,
                max_size_mb=20,
                timeout=360
            ).send()
        await send_message("Processing files...")

        extracted_text = await cl.make_async(extract_text)(cl.user_session.get("llm"), uploaded)
        extract_text = await cl.make_async

        await cl.make_async(vector_store.add)(extracted_text)
        await send_message("Knowledge base loaded.")


@cl.on_message
async def main(message: cl.Message):
    llm: OllamaLanguageModel = cl.user_session.get("llm")
    vector_store: VectorStore = cl.user_session.get("vector_store")

    message_history: list = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    if len(message.elements) > 0:
        total_text = await cl.make_async(extract_text)(cl.user_session.get("llm"), message.elements)
        await cl.make_async(vector_store.add)(total_text)

    answer = await cl.make_async(summarize)(llm, message.content, vector_store)
    await send_message(answer)
    message_history.append({"role": "assistant", "content": answer})
