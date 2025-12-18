import re

# simple text cleaning
def clean_text(txt: str) -> str:
    text = txt.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

