#lru_cache stands for Least Recently Used Cache
#It is a decorator that stores the results of function calls 
# so that if the same arguments are used again, 
# Python returns the cached result instead of recomputing it
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
#textembedder loads model "gemini-embedding-001" repeatedly 
#so we use lru_cache to save this


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