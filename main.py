import os
import argparse
import os.path as osp

from src.text.extract import extract
from src.model.model import manage_Model

import warnings
warnings.filterwarnings("ignore")



def ollama(func):
    def wrapper(*args, **kwargs):
        manage_Model("llava:34b-v1.6-q3_K_M", "run")
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            manage_Model("llava:34b-v1.6-q3_K_M", "stop")
    return wrapper



@ollama
def main(args):
    data_path = osp.join(os.getcwd(), args.data_path)
    raw_data_path = osp.join(data_path, "RAW")


    PROMPT = ""
    with open(osp.join("src", "model", "prompt.txt"), "r") as f:
        PROMPT = f.read()
        print()
        print(f"Prompt: {PROMPT}")
        print()


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

                    extracted_data = extract(file_path, PROMPT)
                    print(f"Extracted data: {extracted_data}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="Path to data directory")
    args = parser.parse_args()
    main(args)
