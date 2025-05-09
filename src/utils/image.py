from PIL import Image


def resize_to_169(image):
    original_width, original_height = image.size

    target_aspect_ratio = 16 / 9
    original_aspect_ratio = original_width / original_height

    if abs(original_aspect_ratio - target_aspect_ratio) < 1e-4:
        return image

    if original_aspect_ratio > target_aspect_ratio:
        final_height = original_height
        final_width = int(round(original_height * target_aspect_ratio))
    else:
        final_width = original_width
        final_height = int(round(original_width / target_aspect_ratio))

    final_width = max(1, final_width)
    final_height = max(1, final_height)
    resized_image = image.resize((final_width, final_height), Image.Resampling.LANCZOS)
    return resized_image
