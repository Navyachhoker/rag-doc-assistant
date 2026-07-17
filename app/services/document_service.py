"""
All document upload/validation/lifecycle business logic — extracted out
of the route handler so it's independently testable and the router stays
a thin HTTP adapter.
"""
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.database import Document
from app.core.logging_config import get_logger
from app.ingestion.pipeline import IngestionPipeline

settings = get_settings()
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {"pdf"}
UPLOAD_TMP_DIR = Path("./storage/uploads")
UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)


class UnsupportedFileTypeError(Exception):
    pass


class FileTooLargeError(Exception):
    pass


def validate_file(filename: str, contents: bytes) -> str:
    """Returns the validated extension, or raises a domain-specific error."""
    file_ext = Path(filename).suffix.lstrip(".").lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '.{file_ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise FileTooLargeError(f"File too large ({size_mb:.1f}MB). Max is {settings.max_upload_size_mb}MB.")

    return file_ext


def save_temp_file(contents: bytes, file_ext: str) -> Path:
    tmp_filename = f"{uuid.uuid4().hex}.{file_ext}"
    tmp_path = UPLOAD_TMP_DIR / tmp_filename
    tmp_path.write_bytes(contents)
    return tmp_path


async def process_upload(
    file: UploadFile,
    db: Session,
    pipeline: IngestionPipeline,
) -> Document:
    """
    Orchestrates the full upload flow: validate -> persist temp file ->
    create DB record -> run ingestion -> clean up temp file.
    Always returns the Document (status reflects success/failure) rather
    than raising on ingestion failure, since a failed ingestion is a
    valid, displayable outcome — not a system error.
    """
    contents = await file.read()
    file_ext = validate_file(file.filename, contents)
    tmp_path = save_temp_file(contents, file_ext)

    document = Document(filename=file.filename, file_type=file_ext, status="extracting")
    db.add(document)
    db.commit()
    db.refresh(document)

    try:
        pipeline.run(db, document, str(tmp_path))
    except Exception as exc:
        logger.error("upload_ingestion_failed", document_id=document.id, error=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    return document


def list_all_documents(db: Session) -> list[Document]:
    return db.query(Document).order_by(Document.uploaded_at.desc()).all()


def delete_document(db: Session, document_id: int) -> bool:
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        return False
    db.delete(document)
    db.commit()
    return True

def resume_ingestion(db: Session, document_id: int, pipeline: IngestionPipeline) -> Document:
    """
    Re-runs the pipeline against a document that's 'failed' or stuck
    mid-stage. Since 'failed' isn't itself a stage pipeline.run() knows
    how to act on, we inspect the actual DB state first to figure out
    which stage genuinely still has pending work, and reset status to
    that stage before calling run().
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise ValueError(f"Document {document_id} not found")

    if document.status == "ready":
        return document

    if document.status == "failed":
        document.status = _determine_resume_stage(db, document)
        db.commit()

    try:
        pipeline.run(db, document, file_path=None)
    except Exception as exc:
        logger.error("resume_ingestion_failed", document_id=document_id, error=str(exc))

    return document


def _determine_resume_stage(db: Session, document: Document) -> str:
    """
    Inspects actual chunk/image state to figure out which stage still
    has incomplete work, since 'failed' alone doesn't tell us where.
    """
    from app.core.database import DocumentChunk, DocumentImage

    has_pending_chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document.id, DocumentChunk.embedding.is_(None))
        .first()
        is not None
    )
    if has_pending_chunks:
        return "embedding_text"

    has_pending_images = (
        db.query(DocumentImage)
        .filter(DocumentImage.document_id == document.id, DocumentImage.caption.is_(None))
        .first()
        is not None
    )
    if has_pending_images:
        return "captioning_images"

    return "ready"