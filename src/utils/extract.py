import os
import torch
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor

from src.utils.pdf2img2pdf import convert_img2pdf




def batch_extract(files: list[str]) -> list[str]:
    texts = []
    with ThreadPoolExecutor(max_workers = 2) as executor:
        extracted = executor.map(extract_text, [file.path for file in files], [1]*len(files))
        for text in extracted:
            texts.extend(text)
    return texts


def extract_text(file: str, batch_multiplier=2) -> list[str]:
    text = []
    with tempfile.TemporaryDirectory() as dir:
        if file.endswith(".png") or file.endswith(".jpg"):
            images = [file]
            pdf_path = convert_img2pdf(images, os.path.join(dir, "temp.pdf"))
            file = pdf_path

        process = subprocess.run(["marker_single", "--batch_multiplier", str(batch_multiplier), file, dir], stdout=subprocess.PIPE)
        outFolder = process.stdout.decode("utf-8").strip().split(" ")[-2]
        torch.cuda.empty_cache()

        path = os.path.join(dir, outFolder)
        for file in os.listdir(path):
            if file.endswith(".md"):
                with open(os.path.join(path, file), "r") as f:
                    text.append(f.read())
    return text
