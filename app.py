"""
Streamlit dashboard — two tabs:
  1. Ask a Question  — RAG query interface with cited source chunks
  2. Pipeline Analytics — filing counts, query stats, eval metrics from gold tables
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on path when run as `streamlit run app.py`
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from src.config import settings
from src.logging_utils import setup_logging

setup_logging(settings.log_level, settings.log_file)

st.set_page_config(
    page_title="SEC Filing RAG",
    page_icon="📄",
    layout="wide",
)

# ── Cached resource initialisation ────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedder …")
def _get_embedder():
    from src.embed.embedder import get_embedder
    return get_embedder()


@st.cache_resource(show_spinner="Connecting to vector store …")
def _get_store():
    from src.store.qdrant_store import QdrantStore
    return QdrantStore()


def _get_warehouse():
    from src.store.warehouse import Warehouse
    return Warehouse()


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    st.caption(f"Embedder: `{settings.embedder}`")
    st.caption(f"LLM: `{settings.llm_provider}`")
    st.caption(f"Vector store: `{settings.qdrant_location}`")
    top_k = st.slider("Top-K chunks", 1, 20, settings.top_k)

# ── Tabs ─────────────────────────────────────────────────────────────────

tab_query, tab_analytics = st.tabs(["🔍 Ask a Question", "📊 Pipeline Analytics"])

# ── Tab 1: Query ─────────────────────────────────────────────────────────

with tab_query:
    st.header("Ask a question about SEC filings")
    question = st.text_input(
        "Question",
        placeholder="What was Apple's total revenue in fiscal 2023?",
    )
    search_btn = st.button("Search", type="primary")

    if search_btn and question.strip():
        from src.rag.pipeline import RAGPipeline

        embedder = _get_embedder()
        store = _get_store()
        wh = _get_warehouse()

        with st.spinner("Searching …"):
            pipeline = RAGPipeline(embedder, store, wh)
            result = pipeline.query(question, top_k=top_k)
        wh.close()

        st.caption(f"Latency: {result.latency_ms:.0f} ms  |  "
                   f"{len(result.retrieved_chunks)} chunks retrieved")

        if result.answer:
            st.subheader("Answer")
            st.write(result.answer)

        st.subheader("Source Chunks")
        for i, rc in enumerate(result.retrieved_chunks, 1):
            c = rc.chunk
            with st.expander(
                f"[{i}] {c.company_name} — {c.form_type} ({c.filed_date})  "
                f"score={rc.score:.3f}"
            ):
                st.text(c.text[:1000] + ("…" if len(c.text) > 1000 else ""))
                st.caption(f"chunk_id: {c.chunk_id}  |  filing_id: {c.filing_id}")

# ── Tab 2: Analytics ─────────────────────────────────────────────────────

with tab_analytics:
    st.header("Pipeline Analytics")

    try:
        import duckdb
        import pandas as pd

        conn = duckdb.connect(str(settings.duckdb_path), read_only=True)

        # Row counts
        counts_q = """
            SELECT 'raw_filings' as tbl, COUNT(*) as rows FROM raw_filings
            UNION ALL
            SELECT 'raw_chunks', COUNT(*) FROM raw_chunks
            UNION ALL
            SELECT 'raw_query_logs', COUNT(*) FROM raw_query_logs
        """
        counts = conn.execute(counts_q).df()
        st.subheader("Row Counts")
        st.dataframe(counts, use_container_width=True)

        # Filings by company and year
        try:
            filings_df = conn.execute("""
                SELECT company_name, year(filed_date) as year,
                       upper(form_type) as form_type, COUNT(*) as n
                FROM raw_filings
                GROUP BY 1, 2, 3
                ORDER BY 2 DESC, 1
            """).df()
            if not filings_df.empty:
                st.subheader("Filings by Company & Year")
                st.dataframe(filings_df, use_container_width=True)
        except Exception:
            pass

        # Query stats
        try:
            qlog_df = conn.execute("""
                SELECT
                    COUNT(*) as total_queries,
                    AVG(latency_ms) as avg_latency_ms,
                    MAX(latency_ms) as max_latency_ms
                FROM raw_query_logs
            """).df()
            st.subheader("Query Statistics")
            st.dataframe(qlog_df, use_container_width=True)

            recent = conn.execute("""
                SELECT question, latency_ms, queried_at
                FROM raw_query_logs
                ORDER BY queried_at DESC
                LIMIT 10
            """).df()
            if not recent.empty:
                st.subheader("Recent Queries")
                st.dataframe(recent, use_container_width=True)
        except Exception:
            pass

        # Eval sweep results
        sweep_path = Path("eval_results/sweep_results.json")
        if sweep_path.exists():
            import plotly.express as px
            sweep = pd.DataFrame(json.loads(sweep_path.read_text()))
            st.subheader("Retrieval Eval: Hit Rate vs top_k")
            fig = px.line(sweep, x="top_k", y="hit_rate", markers=True,
                          labels={"hit_rate": "Hit Rate", "top_k": "top_k"})
            st.plotly_chart(fig, use_container_width=True)

        conn.close()

    except Exception as exc:
        st.info(f"Warehouse not yet populated — run the pipeline first. ({exc})")
