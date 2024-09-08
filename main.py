import os
import torch
import os.path as osp

from src.text.extract import extract
from src.model.model import getModelAndProcessor


def main(*args, **kwargs):
    data_path = osp.join(os.getcwd(), "data")
    raw_data_path = osp.join(data_path, "RAW")
    processed_data_path = osp.join(data_path, "EXTRACTED")


    PROMPT = ""
    model, processor = getModelAndProcessor(name = "Qwen/Qwen2-VL-7B-Instruct-AWQ", dtype = torch.float16, device = "cuda")
    with open(osp.join("src", "model", "prompt.txt"), "r") as f:
        PROMPT = f.read()
        print(f"Prompt: {PROMPT}")


    for rdir in sorted(os.listdir(raw_data_path)):
        if rdir.startswith("."):
            continue
        for dir in sorted(os.listdir(osp.join(raw_data_path, rdir))):
            if dir.startswith("."):
                continue
            if dir != "Slide":
                continue
            print(f"Processing {rdir, dir}...")
            for file in sorted(os.listdir(osp.join(raw_data_path, rdir, dir))):
                if file.startswith("."):
                    continue
                if file.endswith(".pdf"):
                    file_path = osp.join(osp.basename(osp.dirname(raw_data_path)), osp.basename(raw_data_path), rdir, dir, file)

                    extracted_data = extract(file_path, model, processor, PROMPT)

                    print(f"Extracted data: {extracted_data}")
                    return




if __name__ == '__main__':
    main()
