import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

from src.text.vector import VectorStore
from src.postgres.postgres import Postgres
from src.model.model import HuggingFaceEmbeddingModel, OllamaLanguageModel




def set_role(settings: dict[str, str]) -> None:
    if settings["Role"] == "Explain/Summarize":
        final = "answer and explain the requested argument to a student."
    else:
        final = "create a questionnaire based on the requested argument."

    message_history = cl.user_session.get("message_history")
    if message_history is None:
        message_history = []
        cl.user_session.set("message_history", message_history)

    message_history[0] = {
        "role": "assistant",
        "content": f"You are a professor that will {final}"
        }



async def create_vector_store(postgres_db: Postgres) -> VectorStore:
    embed_model = cl.user_session.get("embed_model")
    if embed_model is None:
        embed_model = await cl.make_async(load_embed)("mixedbread-ai/mxbai-embed-large-v1")
        cl.user_session.set("embed_model", embed_model)
    vector_store = VectorStore(embed_model, postgres_db)
    print("Vector store created.")
    return vector_store


def load_embed(model_name):
    return HuggingFaceEmbeddingModel(model_name).model


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



async def create_settings():
    MODELS_LIST = [
        "gemma3:27b",
        "gemma3:12b",
        "gemma3:4b",
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
    return settings
