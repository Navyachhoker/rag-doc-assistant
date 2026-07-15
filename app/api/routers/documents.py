"""
Document upload/list/delete routes. Pure HTTP adapter layer — no business
logic here, only request/response handling and status code mapping.
"""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.dependencies import get_ingestion_pipeline
from app.core.database import get_db
from app.ingestion.pipeline import IngestionPipeline
from app.models.schemas import DocumentUploadResponse
from app.services import document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):
    try:
        document = await document_service.process_upload(file, db, pipeline)
    except document_service.UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except document_service.FileTooLargeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return DocumentUploadResponse(
        document_id=document.id,
        filename=document.filename,
        status=document.status,
        message="Document processed successfully" if document.status == "ready" else "Document processing failed",
    )


@router.get("", response_model=list[DocumentUploadResponse])
async def list_documents(db: Session = Depends(get_db)):
    documents = document_service.list_all_documents(db)
    return [
        DocumentUploadResponse(document_id=d.id, filename=d.filename, status=d.status, message="")
        for d in documents
    ]


@router.delete("/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    deleted = document_service.delete_document(db, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted"}