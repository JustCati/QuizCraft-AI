import os
from PIL import Image
from qwen_vl_utils import process_vision_info


from src.utils.pdf2img import convert_to_img, convert_to_base64






def extract(file_path, model, processor, prompt):
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

    for idx, img in enumerate(dir):
        if type(img) == str:
            img = Image.open(file_path)
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

        with open(out_path, "a") as f:
            f.write(output_text[0])
        print(f"Extracted data from image n°{idx} with num characters: ", len(output_text[0]))
    return
