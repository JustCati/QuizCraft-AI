import os
from tempfile import TemporaryDirectory

from src.model.inference import extract
from src.utils.pdf2img2pdf import convert_pdf2img
from src.utils.regex import remove_images, remove_links




def extract_text(llm, files):
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

            file_extracted_text = extract(llm, temp_dir)
            file_extracted_text = remove_images(file_extracted_text)
            file_extracted_text = remove_links(file_extracted_text)
            extracted_text.append(file_extracted_text)
        return extracted_text
