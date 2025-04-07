import os
from PIL import Image
from tempfile import TemporaryDirectory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.utils.pdf2img2pdf import convert_pdf2img, convert_to_base64




def extract_text(llm, files: list[str]) -> list[str]:
    def prompt_func(data):
        text = data["text"]
        image = data["image"]

        image_part = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image}",
        }

        content_parts = []
        text_part = {"type": "text", "text": text}

        content_parts.append(image_part)
        content_parts.append(text_part)
        return [HumanMessage(content=content_parts)]


    extracted_text = []
    for file in files:
        file = file.path
        with TemporaryDirectory() as temp_dir:
            if not file.endswith(".jpg") or not file.endswith(".png") or not file.endswith(".jpeg"):
                file_extension = file.split(".")[-1]
                if file_extension == "pdf":
                    images = convert_pdf2img(file)
                    for i, image in enumerate(images):
                        image_path = os.path.join(temp_dir, f"page_{i}.jpg")
                        image.save(image_path, "JPEG")
                        file = os.path.dirname(image_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")

            with open(os.path.join("src", "model", "prompts", "ocr.txt"), "r") as f:
                ocr_prompt = f.read()

            file_extracted_text = ""
            for image_file in sorted(os.listdir(temp_dir)):
                image_path = os.path.join(temp_dir, image_file)
                if os.path.isfile(image_path):
                    image = Image.open(image_path)
                    image_b64 = convert_to_base64(image)

                    chain = prompt_func | llm | StrOutputParser()
                    extracted = chain.invoke({"text": ocr_prompt, "image": image_b64})
                    file_extracted_text += "\n\n"
                    file_extracted_text += extracted
            extracted_text.append(file_extracted_text)
        return extracted_text