#defines response and request schemas for apisexplain 3

from typing import Optional

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: int
    filename: str
    status: str
    message: str

#represents users query
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)#adds validation rules
    document_id: Optional[int] = None  # None = search across all documents


class RetrievedImage(BaseModel):
    image_id: int
    file_path: str
    caption: str
    page_number: Optional[int]
    relevance_score: float #similarity score


class RetrievedChunk(BaseModel):
    chunk_id: int
    content: str
    page_number: Optional[int]
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str  # markdown, with inline ![caption](url) image refs where relevant
    source_chunks: list[RetrievedChunk]
    source_images: list[RetrievedImage]