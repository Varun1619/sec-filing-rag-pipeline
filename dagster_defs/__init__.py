"""
Dagster software-defined assets for the SEC RAG pipeline.

Asset lineage:
  raw_filings -> parsed_text -> chunks -> embedded_chunks
              -> warehouse_filings
                            -> warehouse_chunks

dbt assets (dim_company, dim_filing, fact_chunks, fact_query_logs) are loaded
from the dbt project via dagster-dbt integration and appear in the lineage graph
downstream of the warehouse assets.
"""

from __future__ import annotations

import os
from pathlib import Path

from dagster import (
    AssetExecutionContext,
    Definitions,
    Output,
    asset,
    define_asset_job,
    load_assets_from_modules,
)

from src.config import settings
from src.embed.embedder import get_embedder
from src.ingest.edgar import ingest_companies
from src.ingest.entities import extract_entities
from src.ingest.parse import parse_filing
from src.chunk.chunker import chunk_filing
from src.logging_utils import get_logger, setup_logging
from src.models import Chunk, Filing
from src.store.qdrant_store import QdrantStore
from src.store.warehouse import Warehouse

setup_logging(settings.log_level, settings.log_file)
logger = get_logger(__name__)

# ── Sample company list (override via environment) ────────────────────────
_DEFAULT_CIKS = [
    "0000320193",  # Apple Inc.
    "0001018724",  # Amazon
    "0001652044",  # Alphabet (Google)
]
_FORM_TYPES = ["10-K", "10-Q"]
_MAX_PER_COMPANY = int(os.getenv("SEC_MAX_PER_COMPANY", "3"))


@asset(group_name="ingestion")
def raw_filings(context: AssetExecutionContext) -> list[Filing]:
    """Download filings from EDGAR for the configured company list."""
    ciks = os.getenv("SEC_CIK_LIST", ",".join(_DEFAULT_CIKS)).split(",")
    filings = ingest_companies(
        cik_list=[c.strip() for c in ciks],
        form_types=_FORM_TYPES,
        max_per_company=_MAX_PER_COMPANY,
    )
    context.add_output_metadata({"filings_downloaded": len(filings)})
    return filings


@asset(group_name="ingestion")
def warehouse_filings(context: AssetExecutionContext, raw_filings: list[Filing]) -> None:
    """Persist downloaded filing metadata to DuckDB."""
    with Warehouse() as wh:
        for filing in raw_filings:
            wh.upsert_filing(filing)
        counts = wh.row_counts()
    context.add_output_metadata(counts)


@asset(group_name="processing")
def chunks(
    context: AssetExecutionContext, raw_filings: list[Filing]
) -> list[Chunk]:
    """Parse filings and split into text chunks."""
    from src.ingest.parse import parse_filing as _parse

    all_chunks: list[Chunk] = []
    for filing in raw_filings:
        try:
            text = _parse(filing)
            fc = chunk_filing(filing, text)
            all_chunks.extend(fc)
        except Exception as exc:
            logger.warning("Failed to chunk filing",
                           extra={"filing_id": filing.filing_id, "error": str(exc)})

    # Row-count reconciliation log
    logger.info(
        "Reconciliation",
        extra={"filings_in": len(raw_filings), "chunks_out": len(all_chunks)},
    )
    context.add_output_metadata({
        "filings_in": len(raw_filings),
        "chunks_produced": len(all_chunks),
    })
    return all_chunks


@asset(group_name="processing")
def embedded_chunks(
    context: AssetExecutionContext, chunks: list[Chunk]
) -> list[Chunk]:
    """Embed each chunk and upsert into Qdrant."""
    embedder = get_embedder()
    texts = [c.text for c in chunks]
    vectors = embedder.embed(texts)

    enriched: list[Chunk] = []
    for chunk, vec in zip(chunks, vectors):
        enriched.append(chunk.model_copy(update={"embedding": vec}))

    store = QdrantStore()
    n = store.upsert_chunks(enriched)
    context.add_output_metadata({"vectors_upserted": n, "embed_dim": embedder.dim})
    return enriched


@asset(group_name="processing")
def warehouse_chunks(
    context: AssetExecutionContext, embedded_chunks: list[Chunk]
) -> None:
    """Persist chunks (with embeddings) to DuckDB raw_chunks table."""
    with Warehouse() as wh:
        n = wh.upsert_chunks(embedded_chunks)
        counts = wh.row_counts()
    context.add_output_metadata({"chunks_written": n, **counts})


# ── dbt assets ────────────────────────────────────────────────────────────
# Loaded lazily so the Dagster UI works even without dbt installed.
try:
    from dagster_dbt import DbtCliResource, dbt_assets, DbtProject

    _DBT_PROJECT = DbtProject(project_dir=Path(__file__).parent.parent / "dbt")

    @dbt_assets(manifest=_DBT_PROJECT.manifest_path if _DBT_PROJECT.manifest_path.exists() else None,  # type: ignore[arg-type]
                project=_DBT_PROJECT)
    def sec_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):
        yield from dbt.cli(["build"], context=context).stream()

    _dbt_resource = {"dbt": DbtCliResource(project_dir=str(_DBT_PROJECT.project_dir))}
    _extra_assets = [sec_dbt_assets]
except Exception:
    _extra_assets = []
    _dbt_resource = {}


defs = Definitions(
    assets=[raw_filings, warehouse_filings, chunks, embedded_chunks, warehouse_chunks]
    + _extra_assets,
    resources=_dbt_resource,
    jobs=[
        define_asset_job("full_pipeline", selection="*"),
        define_asset_job("ingest_only", selection=["raw_filings", "warehouse_filings"]),
    ],
)
