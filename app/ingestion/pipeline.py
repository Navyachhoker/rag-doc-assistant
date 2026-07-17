
from sqlalchemy.orm import Session

from app.core.database import Document, DocumentChunk, DocumentImage
from app.core.logging_config import get_logger
from app.embeddings.image_captioner import ImageCaptioner
from app.embeddings.text_embedder import TextEmbedder
from app.ingestion.chunker import chunk_text_by_page
from app.ingestion.extractors import get_extractor
from app.storage.image_store import ImageStore
from app.utils.rate_limiter import RateLimiter
from google.api_core.exceptions import ResourceExhausted

logger = get_logger(__name__)

VISION_RATE_LIMITER = RateLimiter(max_calls=9, period_seconds=60.0)


class IngestionPipeline:
    def __init__(self) -> None:
        self.text_embedder = TextEmbedder()
        self.image_captioner = ImageCaptioner()
        self.image_store = ImageStore()

    def run(self, db: Session, document: Document, file_path: str | None = None) -> None:
        try:
            if document.status == "extracting":
                if file_path is None:
                    raise ValueError("file_path required for a document still in 'extracting' stage")
                self._extract(db, document, file_path)
                document.status = "embedding_text"
                db.commit()

            if document.status == "embedding_text":
                self._embed_text_chunks(db, document)
                document.status = "captioning_images"
                db.commit()

            if document.status == "captioning_images":
                self._caption_images(db, document)
                document.status = "ready"
                db.commit()

            logger.info("ingestion_complete", document_id=document.id)

        except Exception as exc:
            logger.error(
                "ingestion_stage_failed",
                document_id=document.id,
                stage=document.status,
                error=str(exc),
            )
            document.status = "failed"
            db.commit()
            raise

    def _extract(self, db: Session, document: Document, file_path: str) -> None:
        extractor = get_extractor(document.file_type)
        extracted = extractor.extract(file_path)

        chunks = chunk_text_by_page(extracted.text_by_page)
        for chunk in chunks:
            db.add(
                DocumentChunk(
                    document_id=document.id,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    embedding=None,
                )
            )

        for extracted_image in extracted.images:
            if self._is_too_small(extracted_image.image_bytes):
                continue
            file_path_saved = self.image_store.save(
                extracted_image.image_bytes, extracted_image.extension, document.id
            )
            db.add(
                DocumentImage(
                    document_id=document.id,
                    file_path=file_path_saved,
                    caption=None,
                    page_number=extracted_image.page_number,
                    embedding=None,
                )
            )

        db.commit()
        logger.info("extraction_stage_complete", document_id=document.id, chunks=len(chunks))

    def _embed_text_chunks(self, db: Session, document: Document) -> None:
        pending = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document.id, DocumentChunk.embedding.is_(None))
            .all()
        )
        logger.info("embedding_text_pending", document_id=document.id, count=len(pending))

        for chunk in pending:
            chunk.embedding = self.text_embedder.embed(chunk.content, task_type="retrieval_document")
            db.commit()

    

    def _caption_images(self, db: Session, document: Document) -> None:
        pending = (
            db.query(DocumentImage)
            .filter(DocumentImage.document_id == document.id, DocumentImage.caption.is_(None))
            .all()
        )
        logger.info("captioning_images_pending", document_id=document.id, count=len(pending))

        failures = 0
        for image in pending:
            VISION_RATE_LIMITER.wait_if_needed()
            image_bytes = self._read_image_bytes(image.file_path)

            try:
                caption = self.image_captioner.caption(image_bytes)
            except ResourceExhausted:
                # Quota exhausted — every remaining image will fail identically.
                # Stop immediately instead of burning time on guaranteed failures.
                logger.error("quota_exhausted_stopping_batch", document_id=document.id, remaining=len(pending))
                raise RuntimeError("Gemini quota exhausted — resume later once quota resets")
            except Exception as exc:
                logger.warning("image_caption_failed", image_id=image.id, error=str(exc))
                failures += 1
                continue

            if not caption:
                image.caption = ""
                db.commit()
                continue

            image.caption = caption
            image.embedding = self.text_embedder.embed(caption, task_type="retrieval_document")
            db.commit()

        if failures > 0:
            raise RuntimeError(f"{failures} image(s) failed captioning — document not fully ready")
    def _is_too_small(self, image_bytes: bytes) -> bool:
        from io import BytesIO
        from PIL import Image
        try:
            img = Image.open(BytesIO(image_bytes))
            return img.width < 150 or img.height < 150
        except Exception:
            return True

    def _read_image_bytes(self, relative_path: str) -> bytes:
        from pathlib import Path
        from app.config import get_settings
        settings = get_settings()
        filename = relative_path.rsplit("/", 1)[-1]
        return (Path(settings.image_storage_path) / filename).read_bytes()