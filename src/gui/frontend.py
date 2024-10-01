import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import os.path as osp
import chainlit as cl

from src.model.model import OllamaModel
from src.model.inference import extract_file




@cl.on_chat_start
async def on_chat_start():
    PROMPT = ""
    with open(osp.join("src", "model", "prompt.txt"), "r") as f:
        PROMPT = f.read()

    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["application/pdf", "image/jpeg", "image/png"],
            max_size_mb=20,
            timeout=180,
            max_files=10
        ).send()

    # llava:34b-v1.6-q3_K_M
    # llava-llama3:8b-v1.1-fp16
    with OllamaModel("llava-llama3:8b-v1.1-fp16", temperature=0.0, num_predict=512) as model:
        for file in files:
            await cl.Message(content=f"Processing file {file.name}...").send()
            text = await cl.make_async(extract_file)(file, model, PROMPT)
            await cl.Message(content="\n".join(text)).send()
