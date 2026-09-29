"""Application configuration via environment variables (pydantic-settings).

All settings can be overridden with environment variables of the same name.
See ``.env.example`` for the full list with placeholder values.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for ollama-rag-api."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:1b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    # Vector store
    CHROMA_DIR: str = "./data/chroma"

    # Security — empty string means authentication is DISABLED.
    API_KEY: str = ""

    # Rate limiting
    RATE_LIMIT_PER_MIN: int = 60

    # CORS (comma-separated or JSON list). NoDecode: pass the raw string to the
    # validator below instead of letting pydantic-settings JSON-decode it first
    # (an empty env value is not valid JSON and would crash startup).
    CORS_ORIGINS: Annotated[list[str], NoDecode] = []

    # Documents
    MAX_UPLOAD_MB: int = 25

    # RAG
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 4
    HISTORY_WINDOW: int = 6

    # Logging
    LOG_LEVEL: str = "INFO"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_cors(cls, value: object) -> list[str]:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("["):
                return [str(o).strip() for o in json.loads(value)]
            return [o.strip() for o in value.split(",") if o.strip()]
        return list(value or [])

    @property
    def auth_enabled(self) -> bool:
        """True when an API key is configured (auth enforced on /v1/*)."""
        return bool(self.API_KEY)


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings (reads env / .env once)."""
    return Settings()
