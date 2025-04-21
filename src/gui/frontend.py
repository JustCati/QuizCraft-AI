import os
import sys
import chainlit as cl
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.gui.utils import *
from src.text.vector import VectorStore
from src.utils.regex import index_or_not
from src.model.inference import summarize
from src.model.model import OllamaLanguageModel
from langchain_core.messages import HumanMessage, AIMessage

import warnings
warnings.filterwarnings("ignore")




async def index_files(llm, uploaded):
    llm = cl.user_session.get("llm")
    vector_store: VectorStore = cl.user_session.get("vector_store")
    # extracted_text = await cl.make_async(extract_text)(llm, uploaded)
    with open("extracted.txt", "r") as f:
        extracted_text = f.read()
    extracted_text = [extracted_text]
    await cl.make_async(vector_store.add)(extracted_text)


async def init_vector_store():
    vector_store = await (await cl.make_async(create_vector_store)())
    cl.user_session.set("vector_store", vector_store)



@cl.on_settings_update
async def setup_agent(settings):
    model_name = cl.user_session.get("model_name")
    if model_name is None or model_name != settings["Model"]:
        if model_name != settings["Model"]:
            cleanup(llm = True, vector = False)
        llm = await cl.make_async(load_llm)(settings)
        cl.user_session.set("llm_ref", llm)
        cl.user_session.set("llm", llm.model)
    init_history(settings)

    cl.user_session.set("stream_tokens", settings["Streaming"])
    print("Agent setup complete.")


@cl.on_chat_end
def cleanup(llm = True, vector = False):
    if vector:
        vector_store: VectorStore = cl.user_session.get("vector_store")
        if vector_store is not None:
            vector_store.clean()
            print("Database cleanup complete.")

    if llm:
        llm = cl.user_session.get("llm_ref")
        if llm is not None:
            llm.stop()
            print("Agent cleanup complete.")


@cl.on_chat_start
async def main():
    settings = await create_settings()
    cl.user_session.set("settings", settings)

    step = [("Creating vector store", init_vector_store)]
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
                content="Please upload one or multiple file/s (PDF, PNG, JPG) [max 20 files]", 
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
        init_history(cl.user_session.get("settings"))
        message_history = cl.user_session.get("message_history")
    message_history.append(HumanMessage(content=message.content))

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

    if len(message.content) > 0:
        answer = await cl.make_async(summarize)(
            llm,
            message.content,
            vector_store,
        )

        message_history.append(AIMessage(content=answer))
        cl.user_session.set("message_history", message_history)

        await send_message(answer)
