from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm


def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text


def open_and_read_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages_and_text = []
    for page_num, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_text.append(
            {
                "page_num": page_num - 41,
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count": len(text.split(". ")),
                "page_token_count": len(text) / 4,  # 1 token = ~4 characters
                "text": text,
            }
        )

    return pages_and_text
