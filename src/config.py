import os
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

class Settings(BaseModel):
    """Application Settings."""
    gemini_api_key: Optional[str] = Field(default=None)
    log_level: str = Field(default="INFO")
    uploads_dir: str = Field(default="data/uploads")
    vector_store_dir: str = Field(default="data/vector_store")
    gemini_model: str = Field(default="gemini-1.5-flash")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

@lru_cache()
def get_settings() -> Settings:
    """Load settings with caching and directory initialization."""
    uploads = os.getenv("UPLOADS_DIR", "data/uploads")
    vector_store = os.getenv("VECTOR_STORE_DIR", "data/vector_store")
    
    # Ensure required data folders are created locally
    os.makedirs(uploads, exist_ok=True)
    os.makedirs(vector_store, exist_ok=True)

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        uploads_dir=uploads,
        vector_store_dir=vector_store,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
