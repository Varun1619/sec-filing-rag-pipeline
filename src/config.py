"""
Central configuration via pydantic-settings.

All settings are read from environment variables (prefix SEC_) or a .env file.
This means every backend — embedder, LLM, vector store, warehouse — is swappable
via a single env-var change, with no code modification required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the directory that contains the `src/` package.
# Resolving paths relative to this means the CLI works regardless of CWD.
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Ingestion ─────────────────────────────────────────────────────────
    user_agent: str = Field(
        default="SEC-RAG-Pipeline/1.0 (contact: user@example.com)",
        description="EDGAR requires a descriptive User-Agent with contact info.",
    )
    edgar_base_url: str = "https://data.sec.gov"
    rate_limit_rps: float = 8.0
    data_dir: Path = _PROJECT_ROOT / "data"
    watermark_date: str = ""  # ISO date; empty = fetch all

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = 512  # approximate tokens
    chunk_overlap: int = 64

    # ── Embedding ─────────────────────────────────────────────────────────
    embedder: Literal["hashing", "fastembed", "sentence_transformers", "openai"] = "hashing"
    embed_dim: int = 384
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # ── Vector store ─────────────────────────────────────────────────────
    qdrant_location: str = str(_PROJECT_ROOT / "qdrant_data")
    qdrant_collection: str = "sec_chunks"

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_provider: Literal["none", "openai", "anthropic", "groq"] = "none"
    llm_model: str = ""

    # ── Warehouse ─────────────────────────────────────────────────────────
    warehouse: Literal["duckdb"] = "duckdb"
    duckdb_path: Path = _PROJECT_ROOT / "warehouse.duckdb"

    # ── Retrieval ─────────────────────────────────────────────────────────
    top_k: int = 5

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: Path = _PROJECT_ROOT / "logs" / "pipeline.jsonl"

    @field_validator("data_dir", "duckdb_path", "log_file", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        return Path(str(v))

    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze"

    def silver_dir(self) -> Path:
        return self.data_dir / "silver"


# Module-level singleton — import this everywhere.
settings = Settings()
