"""
Builds demo_assets/warehouse.duckdb from live EDGAR data.

Run this locally before committing to pre-populate the demo database that
Streamlit Community Cloud will serve in read-only DEMO_MODE.

Usage:
    python scripts/build_demo_data.py
    python scripts/build_demo_data.py --embedder sentence_transformers
    python scripts/build_demo_data.py --max-per-company 3 --out demo_assets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_utils import setup_logging

setup_logging("INFO")

COMPANIES = {
    "Apple Inc.": "0000320193",
    "Microsoft Corp.": "0000789019",
    "Amazon.com Inc.": "0001018724",
    "Alphabet Inc.": "0001652044",
    "NVIDIA Corp.": "0001045810",
}

FORM_TYPES = ["10-K", "10-Q"]


def build(embedder_name: str, max_per_company: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "warehouse.duckdb"

    # Override settings for the demo build
    import os

    os.environ["SEC_DUCKDB_PATH"] = str(db_path)
    os.environ["SEC_EMBEDDER"] = embedder_name
    os.environ["SEC_QDRANT_LOCATION"] = ":memory:"
    # Reset watermark so all filings are eligible regardless of prior runs
    os.environ["SEC_WATERMARK_DATE"] = ""

    # Re-import after env override so settings picks up new values
    import importlib

    import src.config as cfg_mod

    importlib.reload(cfg_mod)

    from src.chunk.chunker import chunk_filing
    from src.embed.embedder import get_embedder
    from src.ingest.edgar import download_filing, get_company_filings
    from src.ingest.entities import extract_entities
    from src.ingest.parse import parse_filing
    from src.store.warehouse import Warehouse

    # Remove watermark so all filings are re-eligible (already-downloaded files
    # are still skipped by the local-dir check, so re-download doesn't happen)
    from src.config import settings as _settings

    watermark_path = _settings.data_dir / ".watermark"
    if watermark_path.exists():
        watermark_path.unlink()
        print(f"Cleared watermark at {watermark_path}")

    embedder = get_embedder()
    wh = Warehouse(path=db_path)

    total_chunks = 0
    for company_name, cik in COMPANIES.items():
        print(f"\n{'-'*60}")
        print(f"Ingesting {company_name} (CIK {cik}) ...")
        count = 0
        for filing in get_company_filings(cik, form_types=FORM_TYPES, max_filings=max_per_company):
            try:
                filing = download_filing(filing)
            except Exception as exc:
                print(f"  Download failed: {exc}")
                continue

            text = ""
            if filing.local_path:
                try:
                    text = parse_filing(filing)
                except Exception as exc:
                    print(f"  Parse failed: {exc}")

            wh.upsert_filing(filing)

            if not text.strip():
                print(f"  {filing.form_type} {filing.filed_date} -- empty text, skipping chunks")
                continue

            chunks = chunk_filing(filing, text)
            texts = [c.text for c in chunks]
            vectors = embedder.embed(texts)
            for chunk, vec in zip(chunks, vectors):
                chunk.entities = extract_entities(chunk.text)
                chunk.embedding = vec
            wh.upsert_chunks(chunks)
            total_chunks += len(chunks)
            count += 1
            print(f"  {filing.form_type} {filing.filed_date} -- {len(chunks)} chunks embedded")

        print(f"  -> {count} filings processed for {company_name}")

    wh.close()
    counts = Warehouse(path=db_path).row_counts()
    print(f"\n{'='*60}", flush=True)
    print(f"Demo build complete -> {db_path}")
    print(f"  raw_filings : {counts['raw_filings']}")
    print(f"  raw_chunks  : {counts['raw_chunks']} ({total_chunks} with embeddings)")
    print(f"  embedder    : {embedder_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SEC demo dataset")
    parser.add_argument(
        "--embedder",
        default="sentence_transformers",
        choices=["hashing", "sentence_transformers"],
        help="Embedder backend (must match what app.py uses in DEMO_MODE)",
    )
    parser.add_argument(
        "--max-per-company",
        type=int,
        default=4,
        help="Max filings per company (15–25 total for 5 companies)",
    )
    parser.add_argument(
        "--out",
        default="demo_assets",
        help="Output directory for warehouse.duckdb",
    )
    args = parser.parse_args()

    build(
        embedder_name=args.embedder,
        max_per_company=args.max_per_company,
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
