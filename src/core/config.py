"""
Configuration management for Henry Bot M2.

Centralized configuration with environment variable support
and validation using Pydantic.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Configuration
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        "https://openrouter.ai/api/v1",
        env="OPENROUTER_BASE_URL"
    )
    model_name: str = Field(
        "google/gemini-2.0-flash-exp:free",
        env="MODEL_NAME"
    )

    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    debug: bool = Field(False, env="DEBUG")

    # API Authentication
    api_key: str = Field(..., env="API_KEY")

    # LLM Parameters
    temperature: float = Field(0.7, env="TEMPERATURE", ge=0.0, le=2.0)
    max_tokens: int = Field(500, env="MAX_TOKENS", gt=0, le=2000)
    prompting_technique: str = Field("few_shot", env="PROMPTING_TECHNIQUE")

    # RAG Configuration
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    vector_store_path: str = Field(
        "./data/vector_store.faiss", env="VECTOR_STORE_PATH")
    documents_path: str = Field("./data/documents", env="DOCUMENTS_PATH")
    similarity_top_k: int = Field(3, env="SIMILARITY_TOP_K", gt=0, le=20)
    response_scoring_threshold: float = Field(
        0.7, env="RESPONSE_SCORING_THRESHOLD", ge=0.0, le=2.0)
    vector_db_api_key: Optional[str] = Field(None, env="VECTOR_DB_API_KEY")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("./logs/app.log", env="LOG_FILE")
    metrics_csv: str = Field("./logs/metrics.csv", env="METRICS_CSV")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def validate_api_key(self, provided_key: str) -> bool:
        """Validate provided API key against configured key."""
        return provided_key == self.api_key


# Global settings instance
settings = Settings()
