"""
PDF extractor using PyMuPDF (fitz)

Handles three cases per page:
1) Normal text page -> extract text directly.
2) Scanned page (no text layer, but has a full-page image) -> rasterize
   the page and run OCR.
3) Embedded images within a text page -> extract them separately and
   keep them for captioning (these are diagrams/figures, not OCR targets).
"""
import fitz  # PyMuPDF

from app.ingestion.extractors.base import BaseExtractor, ExtractedContent, ExtractedImage
from app.ingestion.ocr import run_ocr
from app.core.logging_config import get_logger

logger = get_logger(__name__)#logger obj

# If a page has fewer than this many characters of native text,
# treat it as a likely scanned page and fall back to OCR.
MIN_TEXT_LENGTH_THRESHOLD = 20


class PDFExtractor(BaseExtractor):
    def extract(self, file_path: str) -> ExtractedContent:
        doc = fitz.open(file_path)#opens pdf
        text_by_page: dict[int, str] = {}
        images: list[ExtractedImage] = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1

            native_text = page.get_text().strip()

            if len(native_text) < MIN_TEXT_LENGTH_THRESHOLD:
                # Likely scanned page -> rasterize and OCR
                logger.info("scanned_page_detected", page=page_number)
                pix = page.get_pixmap(dpi=200)#converts pg to imgs
                page_image_bytes = pix.tobytes("png")
                ocr_text = run_ocr(page_image_bytes)
                text_by_page[page_number] = ocr_text#save scanned text from ocr
                # Also keep the rasterized page as a visual reference
                images.append(
                    ExtractedImage(image_bytes=page_image_bytes, page_number=page_number)
                )
            else:
                text_by_page[page_number] = native_text

                # Extract embedded images (figures, diagrams, charts)
                for img_ref in page.get_images(full=True):
                    xref = img_ref[0]
                    try:
                        base_image = doc.extract_image(xref)
                        images.append(
                            ExtractedImage(
                                image_bytes=base_image["image"],
                                page_number=page_number,
                                extension=base_image.get("ext", "png"),
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            "image_extraction_failed", page=page_number, error=str(exc)
                        )

        doc.close()
        logger.info(
            "pdf_extraction_complete",
            pages=len(text_by_page),
            images_found=len(images),
        )
        return ExtractedContent(text_by_page=text_by_page, images=images)