import torch

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
    
    cl.user_session.set(
        "message_history", 
        [{
            "role": "system",
            "content": f"You are a professor that will {final}"
        }]
    )



async def create_vector_store(postgres_db: Postgres) -> VectorStore:
    embed_model = cl.user_session.get("embed_model")
    if embed_model is None:
        embed_model = await cl.make_async(load_embed)("mixedbread-ai/mxbai-embed-large-v1")
        cl.user_session.set("embed_model", embed_model)
    vector_store = VectorStore(embed_model, postgres_db)
    print("Vector store created.")
    return vector_store


def load_embed(model_name: str) -> HuggingFaceEmbeddingModel:
    return HuggingFaceEmbeddingModel(model_name).model


def load_llm(model_name: str, temperature: float) -> OllamaLanguageModel:
    return OllamaLanguageModel(model_name, temperature).get()


def get_models() -> list[dict[str, dict[str, str]]]:
    MODELS = [
            {
                "qwen2.5:7b": {
                    "model": "qwen2.5:7b",
                    "memory": 6.5e9 / 1024**3,
                    }
                },
            {
                "qwen2.5:32b": {
                    "model": "qwen2.5:32b",
                    "memory": 22.5e9 / 1024**3,
                    }
                },
            {
                "qwq:32b": {
                    "model": "qwq:32b",
                    "memory": 22.5e9 / 1024**3,
                    }
                }
        ]
    return MODELS


def get_list_models() -> list[str]:
    return [list(model.keys())[0] for model in get_models()]


def get_best_model() -> int:
    MODELS = get_models()
    available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    sorted_models = sorted(MODELS, key=lambda x: list(x.values())[0]['memory'], reverse=True)
    best = [model for model in sorted_models if list(model.values())[0]['memory'] <= available_memory][0]
    best_index = MODELS.index(best)
    return best_index


async def create_settings() -> dict[str, str]:
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="QwQ Model",
                values=get_list_models(),
                initial_index=get_best_model(),
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
