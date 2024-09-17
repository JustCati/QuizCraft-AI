import os
import torch
import argparse
import os.path as osp

from src.text.extract import extract
from src.model.model import getQwen2VL7B, getGOTOCR

import warnings
warnings.filterwarnings("ignore")


def main(args):
    data_path = osp.join(os.getcwd(), args.data_path)
    raw_data_path = osp.join(data_path, "RAW")


    PROMPT = ""

    if args.model.lower() == "qwen2":
        # model, processor = getQwen2VL7B(name = "Qwen/Qwen2-VL-7B-Instruct-AWQ", dtype = torch.float16, device = "cuda")
        model, processor = getQwen2VL7B(name = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4", dtype = "auto", device = "cuda")
        with open(osp.join("src", "model", "prompt.txt"), "r") as f:
            PROMPT = f.read()
            print()
            print(f"Prompt: {PROMPT}")
            print()
    else:
        model, processor = getGOTOCR(name = "ucaslcl/GOT-OCR2_0", trust_remote_code=True)
        model.eval().cuda()


    for rdir in sorted(os.listdir(raw_data_path)):
        if rdir.startswith("."):
            continue
        for dir in sorted(os.listdir(osp.join(raw_data_path, rdir))):
            if dir.startswith(".") or (dir != "Question" and dir != "Slide"):
                continue
            print(f"Processing {rdir, dir}...")
            for file in sorted(os.listdir(osp.join(raw_data_path, rdir, dir))):
                if file.startswith("."):
                    continue
                if file.endswith(".pdf") or file.endswith(".jpg"):
                    file_path = osp.join(osp.basename(osp.dirname(raw_data_path)), osp.basename(raw_data_path), rdir, dir, file)

                    extracted_data = extract(args, file_path, model, processor, PROMPT)
                    print(f"Extracted data: {extracted_data}")



if __name__ == '__main__':
    model_choices = ["Qwen2", "GotOcr"]
    model_choices += [model.lower() for model in model_choices]
    model_choices += [model.upper() for model in model_choices]
    model_choices = list(set(model_choices))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="Path to data directory")
    parser.add_argument("--model", type=str, default="GOTOCR", help="Model name", choices=model_choices)
    args = parser.parse_args()
    main(args)
