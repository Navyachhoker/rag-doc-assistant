"""
Orchestrates the full ingestion flow for a single uploaded document:

  extract -> chunk text -> embed text chunks -> caption images ->
  embed image captions -> persist everything to Postgres

This is the single entry point the API route calls. Each stage is
independently swappable/testable because of the modular design in
Phases 2-3.
"""
from sqlalchemy.orm import Session

from app.core.database import Document, DocumentChunk, DocumentImage
from app.core.logging_config import get_logger
from app.embeddings.image_captioner import ImageCaptioner
from app.embeddings.text_embedder import TextEmbedder
from app.ingestion.chunker import chunk_text_by_page
from app.ingestion.extractors import get_extractor
from app.storage.image_store import ImageStore

logger = get_logger(__name__)


class IngestionPipeline:
    def __init__(self) -> None:
        self.text_embedder = TextEmbedder()
        self.image_captioner = ImageCaptioner()
        self.image_store = ImageStore()

    def run(self, db: Session, document: Document, file_path: str) -> None:
        """
        Mutates `document`'s status as it progresses and commits chunks/
        images incrementally. Wrapped in a broad try/except at the top
        level so a failure marks the document 'failed' instead of leaving
        it stuck in 'processing' forever.
        """
        try:
            extractor = get_extractor(document.file_type)
            extracted = extractor.extract(file_path)

            self._process_text(db, document, extracted.text_by_page)
            self._process_images(db, document, extracted.images)

            document.status = "ready"
            db.commit()
            logger.info("ingestion_complete", document_id=document.id)

        except Exception as exc:
            logger.error("ingestion_failed", document_id=document.id, error=str(exc))
            document.status = "failed"
            db.commit()
            raise

    def _process_text(self, db: Session, document: Document, text_by_page: dict[int, str]) -> None:
        chunks = chunk_text_by_page(text_by_page)
        if not chunks:
            logger.warning("no_text_chunks_extracted", document_id=document.id)
            return

        embeddings = self.text_embedder.embed_batch(
            [c.content for c in chunks], task_type="retrieval_document"
        )

        for chunk, embedding in zip(chunks, embeddings):
            db.add(
                DocumentChunk(
                    document_id=document.id,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    embedding=embedding,
                )
            )
        db.flush()
        logger.info("text_chunks_persisted", document_id=document.id, count=len(chunks))

    def _process_images(self, db: Session, document: Document, images) -> None:
        persisted_count = 0

        for extracted_image in images:
            caption = self.image_captioner.caption(extracted_image.image_bytes)

            # Skip images the captioner deemed non-informative (icons, tiny assets)
            if not caption:
                continue

            file_path = self.image_store.save(
                extracted_image.image_bytes,
                extracted_image.extension,
                document.id,
            )

            embedding = self.text_embedder.embed(caption, task_type="retrieval_document")

            db.add(
                DocumentImage(
                    document_id=document.id,
                    file_path=file_path,
                    caption=caption,
                    page_number=extracted_image.page_number,
                    embedding=embedding,
                )
            )
            persisted_count += 1

        db.flush()
        logger.info("images_persisted", document_id=document.id, count=persisted_count)