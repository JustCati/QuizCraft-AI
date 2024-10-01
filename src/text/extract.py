import os
from PIL import Image
from tqdm import tqdm

from src.model.model import getModel, extract_text_from_image
from src.utils.pdf2img import convert_to_img, convert_to_base64





def extract(file_path, prompt):
    out_path = file_path.replace("RAW", "EXTRACTED")
    out_path = out_path.replace(".pdf", ".txt").replace(".jpg", ".txt")

    if os.path.exists(out_path):
        print(f"File {out_path} already exists. Skipping...")
        return

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    if file_path.endswith(".pdf"):
        dir = convert_to_img(file_path)
    else:
        dir = [file_path]

    model = getModel("llava:34b-v1.6-q3_K_M", num_predict=512)

    text = []
    for img in tqdm(dir):
        if type(img) == str:
            img = Image.open(file_path)
        output_text = extract_text_from_image(model, img, prompt)
        text.append(output_text)

    #! FORMAT TEXT HERE

    print(f"Extracted text: {text}")
    # with open(out_path, "a") as f:
    #     f.write("\n".join(text))
    #     f.write("\n")
    return
