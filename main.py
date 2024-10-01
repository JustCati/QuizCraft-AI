import os
import argparse
import os.path as osp

from src.text.extract import extract
from src.model.model import manage_Model

import warnings
warnings.filterwarnings("ignore")



def ollama(func):
    def wrapper(*args, **kwargs):
        manage_Model(args[0].model, "run")

        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            manage_Model(args[0].model, "stop")

    return wrapper



@ollama
def main(args):
    PROMPT = ""
    with open(osp.join("src", "model", "prompt.txt"), "r") as f:
        PROMPT = f.read()
        print()
        print(f"Prompt: {PROMPT}")
        print()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="Path to data directory")
    parser.add_argument("--model", type=str, default="llava:34b-v1.6-q3_K_M", help="Model name")
    args = parser.parse_args()
    main(args)
