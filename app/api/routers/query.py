"""
Query route — thin adapter delegating to query_service.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_answer_generator, get_retriever
from app.core.database import get_db
from app.generation.answer_generator import AnswerGenerator
from app.models.schemas import QueryRequest, QueryResponse
from app.retrieval.retriever import Retriever
from app.services import query_service

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db),
    retriever: Retriever = Depends(get_retriever),
    generator: AnswerGenerator = Depends(get_answer_generator),
):
    return query_service.answer_question(
        db, request.question, request.document_id, retriever, generator
    )