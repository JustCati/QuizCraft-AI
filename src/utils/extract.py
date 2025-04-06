import multiprocessing
from markitdown import MarkItDown
from concurrent.futures import ThreadPoolExecutor

from src.utils.pdf2img2pdf import convert_img2pdf




def batch_extract(files: list[str]) -> list[str]:
    texts = []
    worker_count = min(multiprocessing.cpu_count(), len(files))
    with ThreadPoolExecutor(max_workers = worker_count) as executor:
        extracted = executor.map(extract_text, [file.path for file in files])
        texts.extend(extracted)
    return texts


def extract_text(file: str) -> str:
    if not file.endswith(".pdf"):
        print(file)
        file_extension = file.split(".")[-1]
        print(file_extension)
        convert_img2pdf(file, file.replace(file_extension, ".pdf"))
        file = file.replace(file_extension, ".pdf")

    md = MarkItDown(enable_plugins=True)
    results = md.convert(file)
    return results.text_content
