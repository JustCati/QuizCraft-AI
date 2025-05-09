import os
import asyncio
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

from src.text.vector import VectorStore
from src.model.model import MultiModalEmbeddingModel, OllamaLanguageModel






def get_chat_history():
    chat_history = cl.user_session.get("chat_history")
    if chat_history is None:
        chat_history = []
        cl.user_session.set("chat_history", chat_history)
    return chat_history




async def create_vector_store() -> VectorStore:
    def load_embed(text_model_name, image_model_name):
        return MultiModalEmbeddingModel(text_model_name, image_model_name)

    embed_model = cl.user_session.get("embed_model")
    if embed_model is None:
        embed_model = await cl.make_async(load_embed)("nomic-ai/nomic-embed-text-v1.5", "nomic-ai/nomic-embed-vision-v1.5")
        cl.user_session.set("embed_model", embed_model)

    vector_store = VectorStore(embed_model)
    print("Vector store created.")
    return vector_store





def load_llm(settings):
    try:
        llm = OllamaLanguageModel(settings["Model"], settings["Temperature"]).get()
        cl.user_session.set("model_name", settings["Model"])
    except:
        print("Memory error, loading next smaller model")
        while True:
            model_name = cl.user_session.get("models_list")[cl.user_session.get("model_idx") + 1]
            cl.user_session.set("model_idx", cl.user_session.get("model_idx") + 1)
            try:
                llm = OllamaLanguageModel(model_name, settings["Temperature"]).get()
                cl.user_session.set("model_name", model_name)
                break
            except:
                continue
    return llm


async def show_sequential_progress(steps):
    msg = cl.Message(content="Initializing...")
    await msg.send()

    stop_event = asyncio.Event()

    async def animate(current_step):
        dots = ""
        while not stop_event.is_set():
            dots = "." if dots == "..." else dots + "."
            msg.content = f"{current_step()}{dots}"
            await msg.update()
            await asyncio.sleep(0.5)

    current_text = {"step": "Starting"}

    def current_step():
        return current_text["step"]

    animation_task = asyncio.create_task(animate(current_step))

    try:
        for step_name, step_func in steps:
            current_text["step"] = step_name
            await asyncio.sleep(0.2)  # Allow one update cycle
            await step_func()
        stop_event.set()
        await animation_task
        msg.content = "✅ Initialization complete!"
        await msg.update()
    except Exception as e:
        stop_event.set()
        await animation_task
        msg.content = f"❌ Failed: {e}"
        await msg.update()
        raise

async def show_update_message(msgs, func_call, *args, **kwargs):
    start_msg, ending_msg = msgs
   
    loading_msg, stop_loading = await show_loading_message(start_msg)
    try:
        await func_call(*args, **kwargs)
    finally:
        stop_loading()
        loading_msg.content = ending_msg
        await loading_msg.update()


async def show_loading_message(text, interval=0.5):
    msg = cl.Message(content=text)
    await msg.send()

    stop_event = asyncio.Event()

    async def animate():
        dots = ""
        while not stop_event.is_set():
            dots = "." if dots == "..." else dots + "."
            msg.content = f"{text}{dots}"
            await msg.update()
            await asyncio.sleep(interval)

    asyncio.create_task(animate())
    return msg, stop_event.set


async def send_message(text: str, image=None) -> None:
    stream = cl.user_session.get("stream_tokens")
    if stream:
        msg = cl.Message(content="", elements=[image] if image else None)
        for token in text:
            await msg.stream_token(token)
    else:
        msg = cl.Message(content=text)
    await msg.send()



async def create_settings():
    MODELS_LIST = [
        "gemma3:27b-it-qat",
        "gemma3:12b-it-qat",
        "gemma3:4b-it-qat",
        "gemma3:1b-it-qat"
    ]
    cl.user_session.set("models_list", MODELS_LIST)
    cl.user_session.set("model_idx", 0)

    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Model",
                values=MODELS_LIST,
                initial_index=0
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
    return settings
