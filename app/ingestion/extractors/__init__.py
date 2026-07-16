"""
Factory function — main pipeline calls get_extractor(file_type).
PDF-only for now; add new formats here later if scope expands.
"""
from app.ingestion.extractors.base import BaseExtractor
from app.ingestion.extractors.pdf_extractor import PDFExtractor

_EXTRACTOR_MAP: dict[str, type[BaseExtractor]] = {
    "pdf": PDFExtractor,
}


def get_extractor(file_type: str) -> BaseExtractor:
    file_type = file_type.lower().lstrip(".")
    extractor_cls = _EXTRACTOR_MAP.get(file_type)
    if extractor_cls is None:
        raise ValueError(f"Unsupported file type: {file_type}. Only PDF is supported.")
    return extractor_cls()