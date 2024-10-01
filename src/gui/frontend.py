import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import os.path as osp
from PIL import Image
import chainlit as cl

from src.model.model import OllamaModel
from src.utils.pdf2img import convert_to_img
from src.model.inference import extract_text_from_image




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

    with OllamaModel("llava:34b-v1.6-q3_K_M") as model:
        for idx, file in enumerate(files):
            msg = cl.Message(content=f"Processing file n° {idx}...")
            await msg.send()

            if file.path.endswith(".pdf"):
                dir = convert_to_img(file.path)
            else:
                dir = [file.path]

            text = []
            for img in dir:
                if type(img) == str:
                    img = Image.open(file.path)
                output_text = extract_text_from_image(model, img, PROMPT)
                text.append(output_text)

            msg = cl.Message(content="\n".join(text))
            await msg.send()





# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain")
#     cb = cl.AsyncLangchainCallbackHandler()

#     res = await chain.acall(message.content, callbacks=[cb])
#     answer = res["answer"]
#     source_documents = res["source_documents"]

#     text_elements = []

#     if source_documents:
#         for source_idx, source_doc in enumerate(source_documents):
#             source_name = f"source_{source_idx}"
#             # Create the text element referenced in the message
#             text_elements.append(
#                 cl.Text(content=source_doc.page_content, name=source_name, display="side")
#             )
#         source_names = [text_el.name for text_el in text_elements]

#         if source_names:
#             answer += f"\nSources: {', '.join(source_names)}"
#         else:
#             answer += "\nNo sources found"

#     await cl.Message(content=answer, elements=text_elements).send()