import base64
import tempfile
from PIL import Image
from io import BytesIO

from pdf2image import convert_from_path



def convert_pdf2img(pdf_file_path):
    with tempfile.TemporaryDirectory() as dir:
        images_from_path = convert_from_path(pdf_file_path, output_folder=dir)
        print(f"Extracted {len(images_from_path)} images from {pdf_file_path}")
        return images_from_path


def convert_img2pdf(path, output_file_path):
    if isinstance(path, str):
        path = [path]
    for img in path:
        image = Image.open(img)
        image.save(output_file_path, "PDF", resolution=100.0, save_all=True, append_images=[image])
    print(f"Saved images as PDF to {output_file_path}")
    return output_file_path



def convert_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_64 = base64.b64encode(buffered.getvalue())
    return img_64.decode("utf-8")
