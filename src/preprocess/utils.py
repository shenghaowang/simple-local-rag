from pathlib import Path
from typing import List

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


def split_list(input_list: List[str], slice_size: int = 10) -> List[List[str]]:
    """
    Turn groups of sentences into chunks.
    e.g. [20] -> [10, 10] or [25] -> [10, 10, 5]
    """
    return [
        input_list[i : i + slice_size] for i in range(0, len(input_list), slice_size)
    ]
