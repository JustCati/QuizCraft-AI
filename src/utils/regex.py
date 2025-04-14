import re



def remove_images(text: str) -> str:
    return re.sub(r"![\[a-zA-Z0-9_\.. (),:\/\/\]]+", "", text)


def remove_bloat(text: str) -> str:
    text = re.sub(r'^.*markdown.*$\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*```.*$\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    return text

def remove_links(text: str) -> str:
    return re.sub(r"\[([^\[]+)\]\(([^\)]+)\)", r"\1", text)


def index_or_not(msg):
    if re.search(r"\bindex\b", msg, re.IGNORECASE):
        return True
    return False

