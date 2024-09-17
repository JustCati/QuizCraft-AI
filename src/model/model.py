from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers import AutoTokenizer, AutoModel


# Qwen/Qwen2-VL-7B-Instruct-AWQ
# Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4

def getQwen2VL7B(name = "Qwen/Qwen2-VL-7B-Instruct-AWQ", dtype = "auto", device = "cpu"):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        name, torch_dtype=dtype, device_map=device
    )
    processor = AutoProcessor.from_pretrained(name)
    return model, processor


def getGOTOCR(name = "ucaslcl/GOT-OCR2_0", trust_remote_code=True):
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(name,
                                      trust_remote_code=trust_remote_code, 
                                      low_cpu_mem_usage=True,
                                      device_map='cuda',
                                      use_safetensors=True, 
                                      pad_token_id=tokenizer.eos_token_id)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model = model.eval().cuda()
    return model, tokenizer
