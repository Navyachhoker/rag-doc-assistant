"""
Handles persisting extracted image bytes to disk and returning a path
that can be served back to the frontend. Kept as its own module so
swapping local disk storage for S3/GCS later only touches this file.
"""
import uuid
from pathlib import Path

from app.config import get_settings
from app.core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


class ImageStore:
    def __init__(self) -> None:
        self.base_path = Path(settings.image_storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, image_bytes: bytes, extension: str, document_id: int) -> str:
        """
        Saves image bytes under a unique filename scoped to the document,
        and returns a relative path suitable for storing in the DB and
        later serving via a static file route.
        """
        extension = extension.lstrip(".").lower() or "png"
        filename = f"{document_id}_{uuid.uuid4().hex[:12]}.{extension}"
        file_path = self.base_path / filename

        try:
            file_path.write_bytes(image_bytes)
        except Exception as exc:
            logger.error("image_save_failed", filename=filename, error=str(exc))
            raise

        # Relative path — main.py will mount this directory as static files
        return f"/static/images/{filename}"