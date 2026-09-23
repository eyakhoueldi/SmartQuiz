import os
from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    # Iterate over every page and concatenate extracted text
    for page in reader.pages:
        extracted = page.extract_text()
        # Guard against pages that yield None (e.g. image-only pages)
        text += extracted if extracted is not None else ""
    return text


def load_txt(file_path: str) -> str:
    # Read the entire plain-text file as a UTF-8 string
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_document(file_path: str) -> str:
    # Determine the file type from its extension (lowercased for safety)
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    else:
        # Fail fast for unsupported formats rather than returning partial data
        raise ValueError(f"Unsupported file type: {ext}")