"""
Evaluate the demo TF-IDF retriever against eval_data/eval_set.json.

This is the retriever that actually runs on Streamlit Community Cloud:
  - TfidfVectorizer (max_features=50k, ngram_range=(1,2), sublinear_tf=True)
  - Company name prepended to each indexed chunk (same as app.py)
  - Source: demo_assets/warehouse.duckdb

Writes eval_results/sweep_results.json (overwrites hashing-embedder numbers).

Usage:
    python scripts/eval_tfidf.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_REPO_ROOT = Path(__file__).parent.parent
_DEMO_DB = _REPO_ROOT / "demo_assets" / "warehouse.duckdb"
_EVAL_SET = _REPO_ROOT / "eval_data" / "eval_set.json"
_OUT_DIR = _REPO_ROOT / "eval_results"

# Match exactly what app.py does in _get_demo_searcher()
_TFIDF_PARAMS = dict(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)


def _build_index():
    from src.models import Chunk

    conn = duckdb.connect(str(_DEMO_DB), read_only=True)
    rows = conn.execute(
        "SELECT chunk_id, filing_id, cik, company_name, form_type, "
        "filed_date, chunk_index, text, char_count FROM raw_chunks "
        "WHERE length(text) > 0"
    ).fetchall()
    conn.close()

    chunks = [
        Chunk(
            chunk_id=r[0],
            filing_id=r[1],
            cik=r[2],
            company_name=r[3],
            form_type=r[4],
            filed_date=r[5],
            chunk_index=r[6],
            text=r[7],
            char_count=r[8],
        )
        for r in rows
    ]

    indexed_texts = [
        f"{c.company_name} {c.form_type} {c.filed_date.year}: {c.text}" for c in chunks
    ]

    vec = TfidfVectorizer(**_TFIDF_PARAMS)
    mat = vec.fit_transform(indexed_texts)
    return chunks, vec, mat


def _search(question, chunks, vec, mat, top_k):
    from src.models import RetrievedChunk

    q = vec.transform([question])
    scores = cosine_similarity(q, mat)[0]
    idx = np.argsort(scores)[::-1][:top_k]
    return [RetrievedChunk(chunk=chunks[i], score=float(scores[i])) for i in idx if scores[i] > 0.0]


def _is_hit(sample: dict, retrieved) -> bool:
    for rc in retrieved:
        c = rc.chunk
        if sample.get("expected_filing_id") and c.filing_id == sample["expected_filing_id"]:
            return True
        if sample.get("expected_cik") and c.cik == sample["expected_cik"]:
            return True
        if sample.get("expected_text_contains"):
            if sample["expected_text_contains"].lower() in c.text.lower():
                return True
    return False


def run(top_k_values: list[int] | None = None) -> list[dict]:
    top_k_values = top_k_values or [1, 3, 5, 10]
    samples = json.loads(_EVAL_SET.read_text(encoding="utf-8"))
    n = len(samples)

    print(f"Loading demo DB: {_DEMO_DB}")
    chunks, vec, mat = _build_index()
    print(f"Index ready: {len(chunks):,} chunks\n")

    rows = []
    for k in top_k_values:
        hits = 0
        t0 = time.perf_counter()
        for s in samples:
            retrieved = _search(s["question"], chunks, vec, mat, top_k=k)
            if _is_hit(s, retrieved[:k]):
                hits += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000
        hit_rate = hits / n
        rows.append(
            {
                "top_k": k,
                "hit_rate": round(hit_rate, 4),
                "latency_ms": round(elapsed_ms, 3),
                "n_questions": n,
            }
        )
        print(f"  top_k={k:2d}  hit_rate={hit_rate:.0%}  ({hits}/{n})  {elapsed_ms:.0f} ms")

    _OUT_DIR.mkdir(exist_ok=True)
    out = _OUT_DIR / "sweep_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nResults written to {out}")
    return rows


if __name__ == "__main__":
    run()
