import base64
import tempfile
from io import BytesIO

from pdf2image import convert_from_path



def convert_to_img(pdf_file_path):
    with tempfile.TemporaryDirectory() as dir:
        images_from_path = convert_from_path(pdf_file_path, output_folder=dir)
        print(f"Extracted {len(images_from_path)} images from {pdf_file_path}")
        return images_from_path


def convert_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_64 = base64.b64encode(buffered.getvalue())
    return img_64.decode("utf-8")
