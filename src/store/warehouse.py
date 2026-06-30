"""
DuckDB warehouse layer — persists chunks, embeddings, and query logs.

Tables created here are the raw inputs consumed by dbt staging models.
DuckDB is configured as a local file by default; swapping to Snowflake later
requires only changing the dbt profile (the application layer stays identical).

Table layout (bronze → silver boundary):
  raw_filings      — one row per downloaded filing (from Filing objects)
  raw_chunks       — one row per chunk, including serialised embedding
  raw_query_logs   — one row per user query
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from src.config import settings
from src.logging_utils import get_logger
from src.models import Chunk, Filing, QueryResult

logger = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS raw_filings (
    filing_id        VARCHAR PRIMARY KEY,
    cik              VARCHAR NOT NULL,
    company_name     VARCHAR NOT NULL,
    form_type        VARCHAR NOT NULL,
    filed_date       DATE NOT NULL,
    period_of_report DATE,
    accession_number VARCHAR NOT NULL,
    document_url     VARCHAR,
    local_path       VARCHAR,
    file_size_bytes  BIGINT,
    is_scanned_pdf   BOOLEAN DEFAULT FALSE,
    ingested_at      TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_chunks (
    chunk_id     VARCHAR PRIMARY KEY,
    filing_id    VARCHAR NOT NULL,
    cik          VARCHAR NOT NULL,
    company_name VARCHAR NOT NULL,
    form_type    VARCHAR NOT NULL,
    filed_date   DATE NOT NULL,
    chunk_index  INTEGER NOT NULL,
    text         VARCHAR NOT NULL,
    char_count   INTEGER,
    entities_json VARCHAR,
    embedding_json VARCHAR,
    chunked_at   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_query_logs (
    query_id       VARCHAR PRIMARY KEY,
    question       VARCHAR NOT NULL,
    chunk_ids_json VARCHAR,
    scores_json    VARCHAR,
    answer         VARCHAR,
    latency_ms     DOUBLE,
    queried_at     TIMESTAMP
);
"""


class Warehouse:
    """Thin DuckDB wrapper for pipeline writes."""

    def __init__(self, path: Path | str | None = None) -> None:
        db_path = str(path or settings.duckdb_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(db_path)
        self._conn.execute(_DDL)
        logger.info("Warehouse connected", extra={"path": db_path})

    def upsert_filing(self, filing: Filing) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO raw_filings VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                filing.filing_id,
                filing.cik,
                filing.company_name,
                filing.form_type,
                filing.filed_date.isoformat(),
                (filing.period_of_report.isoformat() if filing.period_of_report else None),
                filing.accession_number,
                filing.document_url,
                filing.local_path,
                filing.file_size_bytes,
                filing.is_scanned_pdf,
                filing.ingested_at.isoformat(),
            ],
        )

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        rows = [
            (
                c.chunk_id,
                c.filing_id,
                c.cik,
                c.company_name,
                c.form_type,
                c.filed_date.isoformat(),
                c.chunk_index,
                c.text,
                c.char_count,
                json.dumps([e.model_dump() for e in c.entities]),
                json.dumps(c.embedding) if c.embedding else None,
                c.chunked_at.isoformat(),
            )
            for c in chunks
        ]
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO raw_chunks VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            rows,
        )
        return len(rows)

    def log_query(self, result: QueryResult) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO raw_query_logs VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                result.query_id,
                result.question,
                json.dumps(result.chunk_ids),
                json.dumps(result.scores),
                result.answer,
                result.latency_ms,
                result.queried_at.isoformat(),
            ],
        )

    def row_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for table in ("raw_filings", "raw_chunks", "raw_query_logs"):
            row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = row[0] if row else 0
        return counts

    def close(self) -> None:
        self._conn.close()

    # context-manager support
    def __enter__(self) -> "Warehouse":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
