"""
Seeds a minimal DuckDB warehouse for CI dbt-build testing.

Writes synthetic but schema-valid rows so that all dbt models, tests, and
relationships pass without hitting EDGAR or requiring real embeddings.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb

DB_PATH = os.environ.get("SEC_DUCKDB_PATH", "ci_warehouse.duckdb")

DDL = """
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
    chunk_id       VARCHAR PRIMARY KEY,
    filing_id      VARCHAR NOT NULL,
    cik            VARCHAR NOT NULL,
    company_name   VARCHAR NOT NULL,
    form_type      VARCHAR NOT NULL,
    filed_date     DATE NOT NULL,
    chunk_index    INTEGER NOT NULL,
    text           VARCHAR NOT NULL,
    char_count     INTEGER,
    entities_json  VARCHAR,
    embedding_json VARCHAR,
    chunked_at     TIMESTAMP
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

COMPANIES = [
    ("0000320193", "Apple Inc.", "AAPL"),
    ("0001018724", "Amazon.com Inc.", "AMZN"),
]

NOW = datetime.utcnow().isoformat()
TODAY = date.today().isoformat()


def main() -> None:
    conn = duckdb.connect(DB_PATH)
    conn.execute(DDL)

    filing_ids: list[str] = []
    for cik, name, _ in COMPANIES:
        filing_id = str(uuid4())
        filing_ids.append(filing_id)
        conn.execute(
            "INSERT OR REPLACE INTO raw_filings VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                filing_id,
                cik,
                name,
                "10-K",
                "2023-10-01",
                "2023-09-30",
                f"{cik}-23-000001",
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/000001/filing.htm",
                None,
                None,
                False,
                NOW,
            ],
        )

    for filing_id in filing_ids:
        row = conn.execute(
            "SELECT cik, company_name, form_type, filed_date FROM raw_filings WHERE filing_id=?",
            [filing_id],
        ).fetchone()
        cik, company_name, form_type, filed_date = row  # type: ignore[misc]
        for i in range(3):
            conn.execute(
                "INSERT OR REPLACE INTO raw_chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    str(uuid4()),
                    filing_id,
                    cik,
                    company_name,
                    form_type,
                    filed_date,
                    i,
                    f"Sample text chunk {i} for {company_name} annual report.",
                    50,
                    json.dumps([]),
                    None,
                    NOW,
                ],
            )

    conn.execute(
        "INSERT OR REPLACE INTO raw_query_logs VALUES (?,?,?,?,?,?,?)",
        [
            str(uuid4()),
            "What was Apple revenue?",
            json.dumps([]),
            json.dumps([]),
            None,
            42.0,
            NOW,
        ],
    )

    conn.close()
    print(f"CI warehouse seeded at {DB_PATH}")


if __name__ == "__main__":
    main()
