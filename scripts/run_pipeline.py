"""
CLI entry point for running the pipeline without Dagster.

Usage:
  python -m scripts.run_pipeline ingest --ciks 0000320193,0001018724 --max 3
  python -m scripts.run_pipeline build
  python -m scripts.run_pipeline query "What was Apple's revenue in 2023?"
  python -m scripts.run_pipeline eval
  python -m scripts.run_pipeline all   # ingest + build + eval
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.logging_utils import get_logger, setup_logging

setup_logging(settings.log_level, settings.log_file)
logger = get_logger(__name__)


@click.group()
def cli() -> None:
    """SEC Filing RAG Pipeline CLI."""


@cli.command()
@click.option("--ciks", default="0000320193,0001018724,0001652044",
              help="Comma-separated CIK numbers to ingest.")
@click.option("--forms", default="10-K,10-Q", help="Comma-separated form types.")
@click.option("--max", "max_per", default=3, help="Max filings per company.")
def ingest(ciks: str, forms: str, max_per: int) -> None:
    """Fetch and download SEC filings from EDGAR."""
    from src.ingest.edgar import ingest_companies
    from src.store.warehouse import Warehouse

    cik_list = [c.strip() for c in ciks.split(",")]
    form_list = [f.strip() for f in forms.split(",")]

    click.echo(f"Ingesting {len(cik_list)} companies, up to {max_per} filings each …")
    filings = ingest_companies(cik_list=cik_list, form_types=form_list, max_per_company=max_per)

    with Warehouse() as wh:
        for f in filings:
            wh.upsert_filing(f)
        counts = wh.row_counts()

    click.echo(f"Done. {len(filings)} filings downloaded. DB counts: {counts}")


@cli.command()
def build() -> None:
    """Parse, chunk, embed and index all downloaded filings."""
    import json
    from pathlib import Path

    from src.chunk.chunker import chunk_filing
    from src.embed.embedder import get_embedder
    from src.ingest.parse import parse_filing
    from src.models import Filing
    from src.store.qdrant_store import QdrantStore
    from src.store.warehouse import Warehouse

    bronze = settings.bronze_dir()
    if not bronze.exists():
        click.echo("Bronze dir not found — run `ingest` first.")
        return

    # Load filing metadata from DuckDB
    import duckdb
    conn = duckdb.connect(str(settings.duckdb_path))
    rows = conn.execute("SELECT * FROM raw_filings").fetchall()
    cols = [d[0] for d in conn.description]
    conn.close()

    if not rows:
        click.echo("No filings in warehouse — run `ingest` first.")
        return

    filings = []
    for row in rows:
        d = dict(zip(cols, row))
        filings.append(Filing(**d))

    embedder = get_embedder()
    store = QdrantStore()
    all_chunks = []

    click.echo(f"Processing {len(filings)} filings …")
    with Warehouse() as wh:
        for filing in filings:
            try:
                text = parse_filing(filing)
                fc = chunk_filing(filing, text)
                texts = [c.text for c in fc]
                if texts:
                    vecs = embedder.embed(texts)
                    embedded = [c.model_copy(update={"embedding": v})
                                for c, v in zip(fc, vecs)]
                    store.upsert_chunks(embedded)
                    wh.upsert_chunks(embedded)
                    all_chunks.extend(embedded)
            except Exception as exc:
                logger.warning("Error processing filing",
                               extra={"filing_id": filing.filing_id, "error": str(exc)})

        counts = wh.row_counts()

    logger.info("Reconciliation",
                extra={"filings_in": len(filings), "chunks_out": len(all_chunks)})
    click.echo(f"Built {len(all_chunks)} chunks. DB counts: {counts}")


@cli.command()
@click.argument("question")
@click.option("--top-k", default=None, type=int)
def query(question: str, top_k: int | None) -> None:
    """Run a natural-language query against the indexed filings."""
    from src.embed.embedder import get_embedder
    from src.rag.pipeline import RAGPipeline
    from src.store.qdrant_store import QdrantStore
    from src.store.warehouse import Warehouse

    embedder = get_embedder()
    store = QdrantStore()
    with Warehouse() as wh:
        pipeline = RAGPipeline(embedder, store, wh)
        result = pipeline.query(question, top_k=top_k)

    click.echo(f"\nQuestion: {result.question}")
    click.echo(f"Latency: {result.latency_ms:.1f} ms")
    click.echo(f"Retrieved {len(result.retrieved_chunks)} chunks")
    if result.answer:
        click.echo(f"\nAnswer:\n{result.answer}")
    else:
        click.echo("\n--- Top chunk ---")
        if result.retrieved_chunks:
            click.echo(result.retrieved_chunks[0].chunk.text[:500])


@cli.command("eval")
@click.option("--eval-set", default="eval_data/eval_set.json")
@click.option("--output-dir", default="eval_results")
def run_eval(eval_set: str, output_dir: str) -> None:
    """Run the deterministic retrieval evaluation sweep."""
    from src.embed.embedder import get_embedder
    from src.eval.evaluate import load_eval_set, parameter_sweep
    from src.store.qdrant_store import QdrantStore

    samples = load_eval_set(eval_set)
    embedder = get_embedder()
    store = QdrantStore()

    results_path = parameter_sweep(samples, embedder, store, output_dir=output_dir)
    click.echo(f"Eval results written to {results_path}")


@cli.command("all")
@click.option("--ciks", default="0000320193,0001018724,0001652044")
@click.option("--max", "max_per", default=3)
def run_all(ciks: str, max_per: int) -> None:
    """Convenience: ingest + build + eval in one shot."""
    from click.testing import CliRunner
    runner = CliRunner()
    runner.invoke(ingest, [f"--ciks={ciks}", f"--max={max_per}"], catch_exceptions=False)
    runner.invoke(build, [], catch_exceptions=False)
    runner.invoke(run_eval, [], catch_exceptions=False)
    click.echo("Pipeline complete.")


if __name__ == "__main__":
    cli()
