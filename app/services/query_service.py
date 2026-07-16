"""
Query orchestration logic — retrieval + generation, extracted out of the
route handler. This is also the natural seam for adding things like
query logging/analytics or caching later without touching the router.
"""
from sqlalchemy.orm import Session

from app.generation.answer_generator import AnswerGenerator
from app.models.schemas import QueryResponse, RetrievedChunk, RetrievedImage
from app.retrieval.retriever import Retriever

EMPTY_RESULT_MESSAGE = "I couldn't find any relevant content in the uploaded documents to answer that."


def answer_question(
    db: Session,
    question: str,
    document_id: int | None,
    retriever: Retriever,
    generator: AnswerGenerator,
) -> QueryResponse:
    chunks, images = retriever.retrieve(db, question, document_id)

    if not chunks and not images:
        return QueryResponse(answer=EMPTY_RESULT_MESSAGE, source_chunks=[], source_images=[])

    answer, used_images = generator.generate(question, chunks, images)

    return QueryResponse(
        answer=answer,
        source_chunks=[
            RetrievedChunk(
                chunk_id=c.chunk_id,
                content=c.content,
                page_number=c.page_number,
                relevance_score=round(c.score, 3),
            )
            for c in chunks
        ],
        source_images=[
            RetrievedImage(
                image_id=i.image_id,
                file_path=i.file_path,
                caption=i.caption,
                page_number=i.page_number,
                relevance_score=round(i.score, 3),
            )
            for i in used_images
        ],
    )