import re



def remove_images(text: str) -> str:
    return re.sub(r"![\[a-zA-Z0-9_\..\]]+[\(0-9a-zA-Z\._\)]+ ", "", text)


def remove_links(text: str) -> str:
    return re.sub(r"\[([^\[]+)\]\(([^\)]+)\)", r"\1", text)


def clean(text: str) -> str:
    if isinstance(text, str):
        text = [text]

    cleaned_text = []
    for t in text:
        lines = t.splitlines()
        for i, line in enumerate(lines):
            #! Convert to regex
            if line.strip() == "":
                lines[i] = ""
                continue
            if len(line) < 5:
                lines[i] = ""
                continue
            if len(line.split(" ")) < 2:
                lines[i-1] += " " + line
                lines[i] = ""
                continue

        cleaned = "\n".join(lines)
        cleaned_text.append(cleaned)
    return cleaned_text

