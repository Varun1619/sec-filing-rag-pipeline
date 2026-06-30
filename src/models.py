"""
Pydantic domain models — the schema contract at every stage boundary.

These are the "rows" that flow through the pipeline:
  Filing  -> parsed into  Chunk(s)  -> each Chunk gets an embedding vector
  QueryResult bundles retrieval output for the RAG layer and for logging.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class FormType(str, Enum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EIGHT_K = "8-K"
    DEF_14A = "DEF 14A"
    OTHER = "OTHER"


class Filing(BaseModel):
    """Represents a single SEC filing document (bronze layer)."""

    filing_id: str = Field(default_factory=lambda: str(uuid4()))
    cik: str
    company_name: str
    form_type: str
    filed_date: date
    period_of_report: date | None = None
    accession_number: str  # e.g. "0001234567-23-000001"
    document_url: str
    local_path: str | None = None  # set after download
    file_size_bytes: int | None = None
    is_scanned_pdf: bool = False
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("cik")
    @classmethod
    def pad_cik(cls, v: str) -> str:
        return v.zfill(10)

    @field_validator("accession_number")
    @classmethod
    def normalise_accession(cls, v: str) -> str:
        return v.replace("/", "-")


class Entity(BaseModel):
    """A named entity or financial figure extracted from a chunk."""

    entity_type: str  # e.g. "TICKER", "MONEY", "DATE", "FISCAL_PERIOD", "METRIC"
    value: str
    start_char: int | None = None
    end_char: int | None = None


class Chunk(BaseModel):
    """
    A text segment produced by chunking a Filing (silver layer).

    chunk_index is the zero-based position within the filing so we can
    reconstruct document order for citation.
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    filing_id: str
    cik: str
    company_name: str
    form_type: str
    filed_date: date
    chunk_index: int
    text: str
    char_count: int = 0
    entities: list[Entity] = Field(default_factory=list)
    embedding: list[float] | None = None
    chunked_at: datetime = Field(default_factory=datetime.utcnow)

    def model_post_init(self, __context: Any) -> None:
        if not self.char_count:
            self.char_count = len(self.text)

    @field_validator("text")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk text must not be empty")
        return v


class RetrievedChunk(BaseModel):
    """A Chunk returned from vector search, annotated with its score."""

    chunk: Chunk
    score: float


class QueryResult(BaseModel):
    """Full output of a single RAG query — stored in query_logs."""

    query_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    retrieved_chunks: list[RetrievedChunk]
    answer: str | None = None  # None when llm_provider="none"
    latency_ms: float = 0.0
    queried_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def chunk_ids(self) -> list[str]:
        return [rc.chunk.chunk_id for rc in self.retrieved_chunks]

    @property
    def scores(self) -> list[float]:
        return [rc.score for rc in self.retrieved_chunks]
