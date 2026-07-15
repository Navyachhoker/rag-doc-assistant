from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # extra-> this will ignroe the extra var that are in the env but 
    # not defined in the config file, and will not throw error

    # Database
    database_url: str

    # API key
    google_api_key: str
    groq_api_key: str

    # Storage
    image_storage_path: str = "./storage/images"
    max_upload_size_mb: int = 50

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    top_k_text: int = 5
    top_k_images: int = 3

    # App
    log_level: str = "INFO"
    environment: str = "development"
    #env-> tells us where the application is running
    #log_l -> tells us how info will log prints(debug,info,warning,error)

    # Models (pinned centrally so upgrades happen in one place)
    embedding_model: str = "models/text-embedding-004" 
    vision_model: str = "gemini-1.5-flash"
    llm_model: str = "llama-3.3-70b-versatile"


@lru_cache
def get_settings() -> Settings:
    #Cached settings instance — avoids re-parsing .env on every import
    return Settings()