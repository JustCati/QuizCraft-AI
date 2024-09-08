import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration



def getModelAndProcessor(name = "Qwen/Qwen2-VL-7B-Instruct-AWQ", dtype = "auto", device = "cpu"):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        name, torch_dtype=dtype, device_map=device
    )
    processor = AutoProcessor.from_pretrained(name)
    return model, processor

