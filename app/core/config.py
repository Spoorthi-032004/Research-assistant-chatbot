"""
Central application configuration.
All values are overridable via environment variables (.env file).
"""
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # --- App ---
    APP_NAME: str = "AI Research Assistant"
    APP_ENV: str = Field(default="development")
    DEBUG: bool = Field(default=True)

    # --- Mongo ---
    MONGO_URI: str = Field(default="mongodb://localhost:27017")
    MONGO_DB_NAME: str = Field(default="research_assistant")

    # --- FAISS ---
    FAISS_INDEX_DIR: str = Field(default="./data/faiss_indexes")
    FAISS_DIM: int = Field(default=384)  # all-MiniLM-L6-v2 output dim

    # --- Embedding model ---
    EMBEDDING_MODEL_NAME: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # --- Groq LLM ---
    GROQ_API_KEY: str = Field(default="")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")
    GROQ_BASE_URL: str = Field(default="https://api.groq.com/openai/v1")
    LLM_MAX_TOKENS: int = Field(default=1024)
    LLM_TEMPERATURE: float = Field(default=0.2)

    # --- Cost estimation (USD per 1M tokens). Adjust to current Groq pricing. ---
    LLM_INPUT_COST_PER_1M: float = Field(default=0.59)
    LLM_OUTPUT_COST_PER_1M: float = Field(default=0.79)

    # --- File storage ---
    UPLOAD_DIR: str = Field(default="./data/uploads")
    MAX_UPLOAD_SIZE_MB: int = Field(default=50)

    # --- Chunking ---
    CHUNK_SIZE_TOKENS: int = Field(default=500)
    CHUNK_OVERLAP_TOKENS: int = Field(default=60)
    TOP_K_RETRIEVAL: int = Field(default=5)

    # --- Multi-paper upload ---
    MAX_UPLOAD_FILES: int = Field(default=3)
    PAPER_SIMILARITY_THRESHOLD: float = Field(default=0.80)

    # --- Chat sessions / conversational memory ---
    MAX_CHAT_HISTORY_EXCHANGES: int = Field(default=12)  # user+assistant pairs kept per session (spec: 10-15)

    # --- Embedding batching ---
    EMBEDDING_BATCH_SIZE: int = Field(default=32)

    # --- Guardrails ---
    ENABLE_PRESIDIO: bool = Field(default=True)
    ENABLE_NEMO_GUARDRAILS: bool = Field(default=True)
    BLOCKED_ENTITIES: List[str] = Field(
        default=[
            "CREDIT_CARD",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "US_SSN",
            "IBAN_CODE",
            "CRYPTO",
        ]
    )

    # --- External research search APIs ---
    SEMANTIC_SCHOLAR_BASE_URL: str = Field(default="https://api.semanticscholar.org/graph/v1")
    OPENALEX_BASE_URL: str = Field(default="https://api.openalex.org")
    CROSSREF_BASE_URL: str = Field(default="https://api.crossref.org")
    ARXIV_BASE_URL: str = Field(default="http://export.arxiv.org/api/query")
    EXTERNAL_API_TIMEOUT_SECONDS: int = Field(default=15)

    # --- Logging ---
    LOG_LEVEL: str = Field(default="INFO")

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
