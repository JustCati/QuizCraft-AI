import os
import base64
from tempfile import TemporaryDirectory
import chainlit as cl

from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader

from src.gui.utils import *
from src.utils.regex import *
from src.utils.pdf2img2pdf import convert_img2pdf




def extract_from_pdf(file):
    img_dir = None
    if not file.endswith(".pdf"):
        img_dir = TemporaryDirectory()
        file = convert_img2pdf(file, os.path.join(img_dir.name, f"{os.path.basename(file)}.pdf"))
    name_without_suff = file.split(".")[0]

    tmp_dir = TemporaryDirectory()
    local_image_dir = os.path.join(tmp_dir.name, "output", "images")
    local_md_dir = os.path.join(tmp_dir.name, "output")
    image_dir = str(os.path.basename(local_image_dir))
    os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(file)

    ds = PymuDocDataset(pdf_bytes)
    ocr = ds.classify() == SupportedPdfParseMethod.OCR
    infer_result = ds.apply(doc_analyze, ocr=ocr)

    pipe = infer_result.pipe_ocr_mode if ocr else infer_result.pipe_txt_mode
    pipe_result = pipe(image_writer)

    md_content = pipe_result.get_markdown(image_dir)
    pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)
    with open(os.path.join(local_md_dir, f"{name_without_suff}.md"), "r") as f:
        md_content = f.read()

    images = []
    for image in os.listdir(local_image_dir):
        if image.endswith(".jpg") or image.endswith(".png"):
            images.append(base64.b64encode(open(os.path.join(local_image_dir, image), "rb").read()).decode("utf-8"))

    tmp_dir.cleanup()
    if img_dir:
        img_dir.cleanup()
    return md_content, images




def extract_text(files):
    #* Remove LLM from GPU to clean up memory
    llm = cl.user_session.get("llm_ref")
    llm.stop()

    extracted_text = []
    extracted_images = []

    for file in files:
        file = file.path
        print("Extracting text from file:", file)

        extracted, images = extract_from_pdf(file)   
        extracted = remove_images(extracted)
        extracted = remove_links(extracted)

        extracted_text.append(extracted)
        extracted_images.extend(images)

    #* Load LLM back to GPU
    cl.make_async(load_llm)(cl.user_session.get("settings")) 
    return extracted_text, images
