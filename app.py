"""
Streamlit dashboard — two tabs:
  1. Ask a Question  — RAG query interface with cited source chunks
  2. Pipeline Analytics — filing counts, query stats from warehouse tables

DEMO_MODE (SEC_DEMO_MODE=true):
  - Loads warehouse from demo_assets/warehouse.duckdb (committed to repo)
  - Builds TF-IDF search index in memory at startup (no model download)
  - Calls Anthropic claude-haiku for generation (ANTHROPIC_API_KEY in secrets)
  - No ingestion, dbt, or Dagster at runtime
  - Read-only; banner shown at top
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

_DEMO_MODE = os.environ.get("SEC_DEMO_MODE", "").lower() in ("1", "true", "yes")

# In demo mode, point config at the pre-built demo warehouse before importing settings
_PROJECT_ROOT = Path(__file__).parent
_DEMO_DB = _PROJECT_ROOT / "demo_assets" / "warehouse.duckdb"

if _DEMO_MODE:
    os.environ.setdefault("SEC_DUCKDB_PATH", str(_DEMO_DB))
    os.environ.setdefault("SEC_EMBEDDER", "hashing")
    os.environ.setdefault("SEC_QDRANT_LOCATION", ":memory:")
    os.environ.setdefault("SEC_LLM_PROVIDER", "anthropic")
    os.environ.setdefault("SEC_LLM_MODEL", "claude-haiku-4-5-20251001")
    # Surface the Anthropic key from Streamlit secrets if present
    try:
        import streamlit as _st

        if hasattr(_st, "secrets") and "ANTHROPIC_API_KEY" in _st.secrets:
            os.environ.setdefault("ANTHROPIC_API_KEY", _st.secrets["ANTHROPIC_API_KEY"])
    except Exception:
        pass

from src.config import settings  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402

setup_logging(settings.log_level)

st.set_page_config(
    page_title="SEC Filing RAG",
    page_icon="📄",
    layout="wide",
)

# ── Demo banner ───────────────────────────────────────────────────────────

if _DEMO_MODE:
    st.info(
        "**Demo mode** — this app is running read-only against a pre-built dataset of "
        "~20 SEC filings (Apple, Microsoft, Amazon, Alphabet, NVIDIA). "
        "Ingestion, dbt, and Dagster are disabled. "
        "[View source on GitHub](https://github.com/varunsingh09/sec-filing-rag-pipeline)",
        icon="ℹ️",
    )

# ── Demo: TF-IDF search index (built once, cached for the session) ────────

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the user's question using ONLY "
    "the provided SEC filing excerpts. For each claim, cite the filing (company name, "
    "form type, and date) in parentheses. If the answer cannot be found in the "
    "provided excerpts, respond with: 'The answer is not found in the provided filings.'"
)


@st.cache_resource(show_spinner="Building search index ...")
def _get_demo_searcher():
    """
    Load chunk texts from DuckDB, fit a TF-IDF vectorizer, return a search callable.
    Pure scikit-learn — no model download, no binary dependencies.
    """
    import duckdb
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    from src.models import Chunk, RetrievedChunk

    conn = duckdb.connect(str(settings.duckdb_path), read_only=True)
    rows = conn.execute(
        "SELECT chunk_id, filing_id, cik, company_name, form_type, "
        "filed_date, chunk_index, text, char_count FROM raw_chunks "
        "WHERE length(text) > 0"
    ).fetchall()
    conn.close()

    if not rows:
        return None

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
    texts = [c.text for c in chunks]

    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)

    def search(question: str, top_k: int = 5) -> list[RetrievedChunk]:
        q_vec = vectorizer.transform([question])
        scores = cosine_similarity(q_vec, matrix)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(chunk=chunks[i], score=float(scores[i]))
            for i in top_idx
            if scores[i] > 0.0
        ]

    return search


# ── Non-demo: embedder + vector store ─────────────────────────────────────


@st.cache_resource(show_spinner="Loading embedder ...")
def _get_embedder():
    from src.embed.embedder import get_embedder

    return get_embedder()


@st.cache_resource(show_spinner="Loading vector index ...")
def _get_store():
    from src.store.qdrant_store import QdrantStore

    return QdrantStore()


# ── Analytics helper ──────────────────────────────────────────────────────


@st.cache_data(ttl=300, show_spinner=False)
def _load_analytics() -> dict:
    """Read warehouse analytics — cached 5 min to avoid repeated DuckDB opens."""
    import duckdb

    try:
        conn = duckdb.connect(str(settings.duckdb_path), read_only=True)
        counts = conn.execute(
            """
            SELECT 'raw_filings' as tbl, COUNT(*) as rows FROM raw_filings
            UNION ALL SELECT 'raw_chunks', COUNT(*) FROM raw_chunks
            UNION ALL SELECT 'raw_query_logs', COUNT(*) FROM raw_query_logs
            """
        ).df()
        filings_df = conn.execute(
            """
            SELECT company_name, year(filed_date) as year,
                   upper(form_type) as form_type, COUNT(*) as n
            FROM raw_filings
            GROUP BY 1, 2, 3 ORDER BY 2 DESC, 1
            """
        ).df()
        recent_q = conn.execute(
            """
            SELECT question, latency_ms, queried_at
            FROM raw_query_logs ORDER BY queried_at DESC LIMIT 10
            """
        ).df()
        conn.close()
        return {"counts": counts, "filings": filings_df, "recent_queries": recent_q}
    except Exception as exc:
        return {"error": str(exc)}


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    if _DEMO_MODE:
        st.caption("Embedder: `tfidf (demo)`")
    else:
        st.caption(f"Embedder: `{settings.embedder}`")
    st.caption(f"LLM: `{settings.llm_provider}`")
    st.caption(
        f"Vector store: `{'in-memory TF-IDF (demo)' if _DEMO_MODE else settings.qdrant_location}`"
    )
    top_k = st.slider("Top-K chunks", 1, 20, settings.top_k)
    if _DEMO_MODE:
        st.caption("🔒 Read-only demo mode")

# ── Tabs ──────────────────────────────────────────────────────────────────

tab_query, tab_analytics = st.tabs(["🔍 Ask a Question", "📊 Pipeline Analytics"])

# ── Tab 1: Query ──────────────────────────────────────────────────────────

with tab_query:
    st.header("Ask a question about SEC filings")
    question = st.text_input(
        "Question",
        placeholder="What was Apple's total revenue in fiscal 2023?",
    )
    search_btn = st.button("Search", type="primary")

    if search_btn and question.strip():
        if _DEMO_MODE:
            # ── Demo path: TF-IDF retrieval + Anthropic generation ────────
            searcher = _get_demo_searcher()
            if searcher is None:
                st.error("No chunks found in demo warehouse — index is empty.")
            else:
                with st.spinner("Searching and generating answer ..."):
                    t0 = time.perf_counter()
                    retrieved = searcher(question, top_k=top_k)

                    answer: str | None = None
                    if retrieved:
                        context_parts: list[str] = []
                        for i, rc in enumerate(retrieved, 1):
                            c = rc.chunk
                            context_parts.append(
                                f"[{i}] {c.company_name} ({c.form_type}, {c.filed_date}):\n{c.text}"
                            )
                        context = "\n\n---\n\n".join(context_parts)

                        if settings.llm_provider == "anthropic" and os.environ.get(
                            "ANTHROPIC_API_KEY"
                        ):
                            import anthropic

                            model = settings.llm_model or "claude-haiku-4-5-20251001"
                            client = anthropic.Anthropic()
                            resp = client.messages.create(
                                model=model,
                                max_tokens=1024,
                                system=_SYSTEM_PROMPT,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                                    }
                                ],
                            )
                            answer = resp.content[0].text if resp.content else ""
                        else:
                            answer = None
                    else:
                        answer = "No relevant filings found in the index."

                    latency_ms = (time.perf_counter() - t0) * 1000

                st.caption(
                    f"Latency: {latency_ms:.0f} ms  |  " f"{len(retrieved)} chunks retrieved"
                )

                if answer:
                    st.subheader("Answer")
                    st.write(answer)

                st.subheader("Source Chunks")
                for i, rc in enumerate(retrieved, 1):
                    c = rc.chunk
                    with st.expander(
                        f"[{i}] {c.company_name} — {c.form_type} ({c.filed_date})"
                        f"  score={rc.score:.3f}"
                    ):
                        st.text(c.text[:1000] + ("..." if len(c.text) > 1000 else ""))
                        st.caption(f"chunk_id: {c.chunk_id}  |  filing_id: {c.filing_id}")

        else:
            # ── Full pipeline path ────────────────────────────────────────
            embedder = _get_embedder()
            store = _get_store()

            from src.rag.pipeline import RAGPipeline
            from src.store.warehouse import Warehouse

            wh = Warehouse()
            with st.spinner("Searching ..."):
                pipeline = RAGPipeline(embedder, store, wh)
                result = pipeline.query(question, top_k=top_k)
            wh.close()

            st.caption(
                f"Latency: {result.latency_ms:.0f} ms  |  "
                f"{len(result.retrieved_chunks)} chunks retrieved"
            )

            if result.answer:
                st.subheader("Answer")
                st.write(result.answer)

            st.subheader("Source Chunks")
            for i, rc in enumerate(result.retrieved_chunks, 1):
                c = rc.chunk
                with st.expander(
                    f"[{i}] {c.company_name} — {c.form_type} ({c.filed_date})"
                    f"  score={rc.score:.3f}"
                ):
                    st.text(c.text[:1000] + ("..." if len(c.text) > 1000 else ""))
                    st.caption(f"chunk_id: {c.chunk_id}  |  filing_id: {c.filing_id}")

# ── Tab 2: Analytics ──────────────────────────────────────────────────────

with tab_analytics:
    st.header("Pipeline Analytics")

    data = _load_analytics()
    if "error" in data:
        st.info(f"Warehouse not yet populated. ({data['error']})")
    else:
        st.subheader("Row Counts")
        st.dataframe(data["counts"], use_container_width=True)

        if not data["filings"].empty:
            st.subheader("Filings by Company & Year")
            st.dataframe(data["filings"], use_container_width=True)

        if not data["recent_queries"].empty:
            st.subheader("Recent Queries")
            st.dataframe(data["recent_queries"], use_container_width=True)

        sweep_path = _PROJECT_ROOT / "eval_results" / "sweep_results.json"
        if sweep_path.exists():
            import pandas as pd
            import plotly.express as px

            sweep = pd.DataFrame(json.loads(sweep_path.read_text()))
            st.subheader("Retrieval Eval: Hit Rate vs top_k")
            fig = px.line(
                sweep,
                x="top_k",
                y="hit_rate",
                markers=True,
                labels={"hit_rate": "Hit Rate", "top_k": "top_k"},
            )
            st.plotly_chart(fig, use_container_width=True)
