import os
import sys
import base64
import chainlit as cl
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.gui.utils import *
from src.text.vector import VectorStore
from src.utils.extract import extract_text
from src.model.inference import summarize, rewrite_query
from langchain_core.messages import HumanMessage, AIMessage

import warnings
warnings.filterwarnings("ignore")




async def index_files(uploaded):
    vector_store: VectorStore = cl.user_session.get("vector_store")
    extracted_text, images = await cl.make_async(extract_text)(uploaded)
    await cl.make_async(vector_store.add)(extracted_text, images)


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

    cl.user_session.set("stream_tokens", settings["Streaming"])
    print("Agent setup complete.")


@cl.on_chat_end
def cleanup(llm = True, vector = False):
    if vector:
        vector_store = cl.user_session.get("vector_store")
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

    vector_store = cl.user_session.get("vector_store")
    if len(vector_store.vector_store.get()["ids"]) == 0:
        print("Vector store is empty, please upload files to index.")
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
            uploaded
        )
    await send_message("CIAO! Puoi farmi qualsiasi domanda riguardo i file indicizzati. Se vuoi indicizzare nuovi file, basta che alleghi il pdf al messaggio.")


@cl.on_message
async def main(message: cl.Message):
    image = None
    llm = cl.user_session.get("llm")
    vector_store = cl.user_session.get("vector_store")

    if len(message.elements) > 0:
        files_to_index = [element for element in message.elements if "pdf" in element.mime]
        images = [element for element in message.elements if "image" in element.mime]

        if len(files_to_index) > 0:
            await index_files(files_to_index)

        if len(images) > 0:
            if len(images) > 1:
                await send_message("Please upload only one image at a time.")
                return
            for elem in images: # Always one, but for consistency
                with open(elem.path, "rb") as f:
                    image = base64.b64encode(f.read()).decode("utf-8")


    if len(message.content) > 0:
        user_query = message.content
        chat_history = get_chat_history()

        if len(chat_history) > 0:
            user_query = await cl.make_async(rewrite_query)(query=user_query,
                                                            llm=llm,
                                                            history=chat_history)
            print(f"USER QUERY REWRITTEN: {user_query}")

        answer = await cl.make_async(summarize)(
            query=user_query,
            llm=llm,
            vector_store=vector_store,
            image=image
        )

        chat_history.append(HumanMessage(content=message.content))
        chat_history.append(AIMessage(content=answer))
        cl.user_session.set("chat_history", chat_history)

        await send_message(answer)
