"""
OCR utility used as a fallback when a PDF page has no extractable text layer
( scanned image), Also used for standalone image uploads that
contain text (e.g., a photographed whiteboard or scanned form)
"""
import io

import pytesseract
from PIL import Image

from app.core.logging_config import get_logger
# top of app/ingestion/ocr.py
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

logger = get_logger(__name__)


def run_ocr(image_bytes: bytes) -> str:
    """Runs Tesseract OCR on raw image bytes and returns extracted text"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)#main ocr call
        return text.strip()
    except Exception as exc:
        logger.warning("ocr_failed", error=str(exc))
        return ""