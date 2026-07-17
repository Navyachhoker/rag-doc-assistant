"""
Performs vector similarity search against both document_chunks and
document_images using pgvector's cosine distance operator, then returns
ranked results for each. The generation layer decides how to
weave these into a final answer.
"""
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.database import DocumentChunk, DocumentImage
from app.core.logging_config import get_logger
from app.embeddings.text_embedder import TextEmbedder

settings = get_settings()
logger = get_logger(__name__)


@dataclass
class RetrievedChunkResult:
    chunk_id: int
    content: str
    page_number: int | None
    score: float  # similarity score, higher = more relevant


@dataclass
class RetrievedImageResult:
    image_id: int
    file_path: str
    caption: str
    page_number: int | None
    score: float


class Retriever:
    def __init__(self) -> None:
        self.text_embedder = TextEmbedder()

    def retrieve(
        self,
        db: Session,
        query: str,
        document_id: int | None = None,
    ) -> tuple[list[RetrievedChunkResult], list[RetrievedImageResult]]:
        """
        Embeds the query once, then searches both tables with it.
        document_id=None searches across the whole corpus; passing a
        specific id scopes search to one uploaded document.
        """
        query_embedding = self.text_embedder.embed(query, task_type="retrieval_query")

        chunks = self._search_chunks(db, query_embedding, document_id)
        images = self._search_images(db, query_embedding, document_id)

        logger.info(
            "retrieval_complete",
            chunks_found=len(chunks),
            images_found=len(images),
        )
        return chunks, images

    def _search_chunks(
        self, db: Session, query_embedding: list[float], document_id: int | None
    ) -> list[RetrievedChunkResult]:
        stmt = select(
            DocumentChunk,
            DocumentChunk.embedding.cosine_distance(query_embedding).label("distance"),
        ).where(DocumentChunk.embedding.isnot(None))

        if document_id is not None:
            stmt = stmt.where(DocumentChunk.document_id == document_id)
        stmt = stmt.order_by("distance").limit(settings.top_k_text)

        results = db.execute(stmt).all()
        return [
            RetrievedChunkResult(
                chunk_id=row.DocumentChunk.id,
                content=row.DocumentChunk.content,
                page_number=row.DocumentChunk.page_number,
                score=1 - row.distance,
            )
            for row in results
        ]

    def _search_images(
        self, db: Session, query_embedding: list[float], document_id: int | None
    ) -> list[RetrievedImageResult]:
        stmt = select(
            DocumentImage,
            DocumentImage.embedding.cosine_distance(query_embedding).label("distance"),
        ).where(DocumentImage.embedding.isnot(None))

        if document_id is not None:
            stmt = stmt.where(DocumentImage.document_id == document_id)
        stmt = stmt.order_by("distance").limit(settings.top_k_images)

        results = db.execute(stmt).all()
        return [
            RetrievedImageResult(
                image_id=row.DocumentImage.id,
                file_path=row.DocumentImage.file_path,
                caption=row.DocumentImage.caption,
                page_number=row.DocumentImage.page_number,
                score=1 - row.distance,
            )
            for row in results
        ]