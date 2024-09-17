import os
from PIL import Image
from tqdm import tqdm
from tempfile import NamedTemporaryFile

from qwen_vl_utils import process_vision_info
from src.utils.pdf2img import convert_to_img, convert_to_base64





def ocr_extract(file_path, model, tokenizer):
    res = model.chat(tokenizer, file_path, ocr_type='format')
    return res



def qwen_extract(img, model, processor, prompt):
    img_64 = convert_to_base64(img)

    messages = [
            {
                "role": "user",
                "content" : [
                    {   
                        "type": "image",
                        "image": "data:image;base64," + img_64,
                        "resized_height": 943,
                        "resized_width": 1417
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text




def extract(args, file_path, model, processor, prompt):
    out_path = file_path.replace("RAW", "EXTRACTED")
    out_path = out_path.replace(".pdf", ".txt").replace(".jpg", ".txt")

    if os.path.exists(out_path):
        print(f"File {out_path} already exists. Skipping...")
        return

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    if file_path.endswith(".pdf"):
        dir = convert_to_img(file_path)
    else:
        dir = [file_path]

    text = []
    for img in tqdm(dir):
        if args.model.lower() == "qwen2":
            if type(img) == str:
                img = Image.open(file_path)
            output_text = qwen_extract(img, model, processor, prompt)
            text.append(output_text[0])
        else:
            with NamedTemporaryFile(mode="wb") as f:
                img.save(f.name, format="JPEG")
                output_text = ocr_extract(f.name, model, processor)
                text.append(output_text)

    #! FORMAT TEXT HERE

    print(f"Extracted text: {text}")
    # with open(out_path, "a") as f:
    #     f.write("\n".join(text))
    #     f.write("\n")
    return
