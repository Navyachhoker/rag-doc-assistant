"""
Splits per-page text into overlapping chunks suitable for embedding.
Uses RecursiveCharacterTextSplitter (industry-standard splitter that
tries paragraph -> sentence -> word boundaries in order, minimizing
mid-sentence cuts).
"""
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

settings = get_settings()


@dataclass
class Chunk:
    content: str
    page_number: int | None
    chunk_index: int


def chunk_text_by_page(text_by_page: dict[int, str]) -> list[Chunk]:
    """
    Splits each page's text independently (so chunks never cross page
    boundaries — this keeps page-number citations accurate) and returns
    a flat, globally-indexed list of chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    global_index = 0

    for page_number, page_text in text_by_page.items():
        if not page_text.strip():
            continue

        page_splits = splitter.split_text(page_text)
        for split in page_splits:
            chunks.append(
                Chunk(content=split, page_number=page_number, chunk_index=global_index)
            )
            global_index += 1

    return chunks