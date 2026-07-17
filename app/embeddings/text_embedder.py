
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)

genai.configure(api_key=settings.google_api_key)


class TextEmbedder:
    def __init__(self) -> None:
        self.model_name = settings.embedding_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed(self, text: str, task_type: str = "retrieval_document") -> list[float]:
        
        if not text.strip():
            # Avoid sending empty strings to the API, return a zero-ish
            # placeholder to keep pipeline flow intact (caller should skip
            # persisting truly empty chunks upstream).
            text = " "

        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type=task_type,
        )
        return result["embedding"]

    def embed_batch(self, texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
        return [self.embed(t, task_type=task_type) for t in texts]