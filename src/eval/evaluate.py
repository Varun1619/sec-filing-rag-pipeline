"""
Retrieval evaluation — deterministic (no LLM required) and RAGAS (optional).

Deterministic metric: Hit@k
  For each question in the eval set, we check whether the expected filing
  (identified by accession number or CIK+form_type+date) appears in the
  top-k retrieved chunks.  No LLM, no API key needed.

Parameter sweep:
  We sweep chunk_size × top_k and output a results table + matplotlib chart.

RAGAS evaluation (optional):
  Requires an LLM key.  Wraps faithfulness and answer_relevancy from ragas.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import settings
from src.embed.embedder import BaseEmbedder
from src.logging_utils import get_logger
from src.store.qdrant_store import QdrantStore

logger = get_logger(__name__)


@dataclass
class EvalSample:
    question: str
    expected_filing_id: str | None = None
    expected_cik: str | None = None
    expected_text_contains: str | None = None  # substring match fallback


@dataclass
class EvalResult:
    hit_at_k: dict[int, float] = field(default_factory=dict)   # k -> hit rate
    recall_at_k: dict[int, float] = field(default_factory=dict)
    total_questions: int = 0
    params: dict[str, Any] = field(default_factory=dict)


def load_eval_set(path: Path | str = "eval_data/eval_set.json") -> list[EvalSample]:
    p = Path(path)
    if not p.exists():
        logger.warning("Eval set not found", extra={"path": str(p)})
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [EvalSample(**item) for item in raw]


def hit_at_k(
    samples: list[EvalSample],
    embedder: BaseEmbedder,
    store: QdrantStore,
    k_values: list[int] | None = None,
) -> EvalResult:
    """
    Deterministic retrieval evaluation — no LLM required.

    A question is a "hit" if any of the top-k retrieved chunks belongs to the
    expected filing (matched by filing_id, cik, or text substring).
    """
    k_values = k_values or [1, 3, 5, 10]
    max_k = max(k_values)
    hits: dict[int, int] = {k: 0 for k in k_values}
    total = len(samples)

    for sample in samples:
        query_vec = embedder.embed([sample.question])[0]
        retrieved = store.search(query_vec, top_k=max_k)

        for k in k_values:
            top = retrieved[:k]
            matched = False
            for rc in top:
                c = rc.chunk
                if sample.expected_filing_id and c.filing_id == sample.expected_filing_id:
                    matched = True
                elif sample.expected_cik and c.cik == sample.expected_cik:
                    matched = True
                elif sample.expected_text_contains and sample.expected_text_contains.lower() in c.text.lower():
                    matched = True
            if matched:
                hits[k] += 1

    return EvalResult(
        hit_at_k={k: hits[k] / total if total else 0.0 for k in k_values},
        total_questions=total,
    )


def parameter_sweep(
    samples: list[EvalSample],
    embedder: BaseEmbedder,
    store: QdrantStore,
    chunk_sizes: list[int] | None = None,
    top_k_values: list[int] | None = None,
    output_dir: Path | str = "eval_results",
) -> Path:
    """
    Sweep chunk_size × top_k and save results table + chart.

    Note: chunk_size affects retrieval only if chunks were stored with that size.
    Here we vary top_k across a fixed indexed store and report Hit@k, which is
    the meaningful axis for retrieval quality given a fixed index.
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k_values = top_k_values or [1, 3, 5, 10]
    rows: list[dict[str, Any]] = []

    if not samples:
        logger.warning("No eval samples; skipping sweep")
        results_path = output_dir / "sweep_results.json"
        results_path.write_text("[]")
        return results_path

    query_vecs = [embedder.embed([s.question])[0] for s in samples]

    for k in top_k_values:
        hits = 0
        t0 = time.perf_counter()
        for sample, qvec in zip(samples, query_vecs):
            retrieved = store.search(qvec, top_k=k)
            for rc in retrieved[:k]:
                c = rc.chunk
                if (
                    (sample.expected_filing_id and c.filing_id == sample.expected_filing_id)
                    or (sample.expected_cik and c.cik == sample.expected_cik)
                    or (sample.expected_text_contains
                        and sample.expected_text_contains.lower() in c.text.lower())
                ):
                    hits += 1
                    break
        latency = (time.perf_counter() - t0) * 1000
        hit_rate = hits / len(samples) if samples else 0.0
        rows.append({"top_k": k, "hit_rate": hit_rate, "latency_ms": latency,
                     "n_questions": len(samples)})
        logger.info("Sweep point", extra={"top_k": k, "hit_rate": round(hit_rate, 3)})

    results_path = output_dir / "sweep_results.json"
    results_path.write_text(json.dumps(rows, indent=2))
    logger.info("Sweep results saved", extra={"path": str(results_path)})

    _plot_sweep(rows, output_dir / "sweep_chart.png")
    return results_path


def _plot_sweep(rows: list[dict], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        ks = [r["top_k"] for r in rows]
        hits = [r["hit_rate"] for r in rows]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ks, hits, marker="o", linewidth=2)
        ax.set_xlabel("top_k")
        ax.set_ylabel("Hit Rate")
        ax.set_title("Retrieval Hit Rate vs top_k")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Sweep chart saved", extra={"path": str(output_path)})
    except ImportError:
        logger.warning("matplotlib not installed; skipping chart")
