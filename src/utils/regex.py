import re



def remove_images(text: str) -> str:
    return re.sub(r"![\[a-zA-Z0-9àòèìù_\.. (),:\/\/\]]+", "", text)

def remove_links(text: str) -> str:
    return re.sub(r"\[([^\[]+)\]\(([^\)]+)\)", r"\1", text)
