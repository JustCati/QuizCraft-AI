import os
import sys
import chainlit as cl
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.gui.utils import *
from src.text.vector import VectorStore
from src.utils.regex import index_or_not
from src.model.inference import summarize
from src.model.model import OllamaLanguageModel

import warnings
warnings.filterwarnings("ignore")




async def index_files(llm, uploaded):
    llm = cl.user_session.get("llm")
    vector_store: VectorStore = cl.user_session.get("vector_store")
    extracted_text = await cl.make_async(extract_text)(llm, uploaded)
    await cl.make_async(vector_store.add)(extracted_text)


async def create_db():
    env_file = os.path.join(os.path.dirname(__file__), "src", "postgres", ".env")
    docker_compose = env_file.replace(".env", "docker-compose.yml")
    db = await cl.make_async(Postgres)(docker_compose, env_file)
    cl.user_session.set("db", db)


async def init_vector_store():
    db = cl.user_session.get("db")
    vector_store = await (await cl.make_async(create_vector_store)(db))
    cl.user_session.set("vector_store", vector_store)




@cl.on_settings_update
async def setup_agent(settings):
    model_name = cl.user_session.get("model_name")
    if model_name is None or model_name != settings["Model"]:
        if model_name != settings["Model"]:
            cleanup()
        llm = await cl.make_async(load_llm)(settings)
        cl.user_session.set("llm_ref", llm)
        cl.user_session.set("llm", llm.model)
    set_role(settings)

    cl.user_session.set("stream_tokens", settings["Streaming"])
    print("Agent setup complete.")


@cl.on_chat_end
def cleanup():
    #* Clean up the database and vector store
    vector_store: VectorStore = cl.user_session.get("vector_store")
    if vector_store is not None:
        vector_store.db.stop()
        print("Database cleanup complete.")

    #* Clean up the language model
    llm = cl.user_session.get("llm_ref")
    if llm is not None:
        llm.stop()
        print("Agent cleanup complete.")


@cl.on_chat_start
async def main():
    settings = await create_settings()

    step = [
        ("Creating database", create_db),
        ("Creating vector store", init_vector_store),
    ]
    await show_sequential_progress(step)
    await setup_agent(settings)

    res = await cl.AskActionMessage(
        content="",
        actions=[
            cl.Action(name="Preload Knowledge", payload={"value": "preload"}, label="📝 Preload Knowledge"),
            cl.Action(name="Message", payload={"value": "message"}, label="💬 Message"),
        ],
    ).send()

    if res and res.get("payload").get("value") == "preload":
        uploaded = None
        while uploaded == None:
             uploaded = await cl.AskFileMessage(
                content="Please upload one or multiple file/s (PDF, PNG, JPG) [max 20 files, 20MB each]", 
                accept=["application/pdf", "image/png", "image/jpg", "image/jpeg"],
                max_files=20,
                max_size_mb=20,
                timeout=360
            ).send()

        await show_update_message(
            ["Indexing files", "✅ Files processed successfully!"], 
            index_files, 
            cl.user_session.get("llm"), 
            uploaded
        )


@cl.on_message
async def main(message: cl.Message):
    llm: OllamaLanguageModel = cl.user_session.get("llm")
    vector_store: VectorStore = cl.user_session.get("vector_store")

    message_history = cl.user_session.get("message_history")
    if message_history is None:
        message_history = []
        cl.user_session.set("message_history", message_history)
    message_history.append({"role": "user", "content": message.content})

    if len(message.elements) > 0:
        if len(message.content) == 0:
            print("Empty message, indexing files...")
            await show_update_message(
                ["Indexing files", "✅ Files processed successfully!"], 
                index_files, 
                cl.user_session.get("llm"), 
                message.elements
            )
        else:
            if index_or_not(message.content):
                print("User is asking for indexing, indexing files...")
                await show_update_message(
                    ["Indexing files", "✅ Files processed successfully!"], 
                    index_files, 
                    cl.user_session.get("llm"), 
                    message.elements
                )
            else:
                pass
                #TODO: INFERENCE WITH LLM AND IMAGE AS INPUT
