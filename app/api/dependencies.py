
from functools import lru_cache

from app.embeddings.image_captioner import ImageCaptioner
from app.embeddings.text_embedder import TextEmbedder
from app.generation.answer_generator import AnswerGenerator
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.retriever import Retriever
from app.storage.image_store import ImageStore


@lru_cache
def get_text_embedder() -> TextEmbedder:
    return TextEmbedder()


@lru_cache
def get_image_captioner() -> ImageCaptioner:
    return ImageCaptioner()


@lru_cache
def get_image_store() -> ImageStore:
    return ImageStore()


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline()


@lru_cache
def get_retriever() -> Retriever:
    return Retriever()


@lru_cache
def get_answer_generator() -> AnswerGenerator:
    return AnswerGenerator()