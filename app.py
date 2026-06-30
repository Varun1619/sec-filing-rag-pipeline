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

import html as _html
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
    page_title="FilingLens · SEC Filing RAG",
    page_icon="📄",
    layout="wide",
)

# ── Ticker helper ─────────────────────────────────────────────────────────

_TICKER_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "nvidia": "NVDA",
}


def _ticker(company_name: str) -> str:
    low = company_name.lower()
    for key, tk in _TICKER_MAP.items():
        if key in low:
            return tk
    return company_name[:4].upper()


# ── Global styles + IBM Plex fonts ────────────────────────────────────────

st.html(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,500;1,400&display=swap" rel="stylesheet">
<style>
:root {
  --brand:#163A36; --brand-bright:#1F574F;
  --hl:#F7DA74; --hl-soft:#FCEDB8; --hl-edge:#D9AE3A;
  --good:#2C7A6B;
  --ink:#15191E; --muted:#5E6873; --faint:#929BA6;
  --canvas:#E7EAEE; --surface:#FFFFFF; --surface-2:#F3F5F7;
  --line:#D9DEE4; --line-2:#E7EBEF;
  --radius:14px; --radius-sm:9px;
  --shadow:0 1px 2px rgba(20,25,30,.05),0 8px 28px rgba(20,25,30,.06);
  --shadow-soft:0 1px 2px rgba(20,25,30,.04);
  --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;
  --serif:"IBM Plex Serif",Georgia,serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}

/* ── shell ── */
.stApp { background:var(--canvas) !important; }
html,body,[class*="css"],.stMarkdown { font-family:var(--sans) !important; }
header[data-testid="stHeader"] { display:none !important; }
.block-container {
  padding-top:0 !important;
  padding-bottom:56px !important;
  max-width:1140px !important;
}

/* ── sidebar ── */
[data-testid="stSidebar"] { background:var(--canvas) !important; border-right:1px solid var(--line) !important; }
[data-testid="stSidebar"] > div:first-child { padding:16px 14px 24px !important; }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background:transparent !important;
  border-bottom:1px solid var(--line) !important;
  gap:2px !important;
  padding:0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family:var(--sans) !important; font-weight:600 !important;
  font-size:.9rem !important; color:var(--faint) !important;
  border-bottom:2px solid transparent !important;
  padding:15px 16px 13px !important;
  background:transparent !important; border-radius:0 !important;
}
.stTabs [data-baseweb="tab"]:hover { color:var(--muted) !important; }
.stTabs [aria-selected="true"] { color:var(--brand) !important; border-bottom-color:var(--brand) !important; }
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display:none !important; }

/* ── text input ── */
.stTextInput > div > div {
  border:1px solid var(--line) !important; border-radius:var(--radius) !important;
  background:var(--surface) !important; box-shadow:var(--shadow) !important;
  padding:4px 8px !important;
}
.stTextInput > div > div > input {
  font-family:var(--sans) !important; font-size:1.02rem !important;
  color:var(--ink) !important; border:none !important;
  background:transparent !important; box-shadow:none !important;
  padding:10px 8px !important;
}
.stTextInput > div > div:focus-within {
  border-color:var(--brand) !important;
  box-shadow:0 0 0 4px rgba(22,58,54,.10),var(--shadow) !important;
}

/* ── buttons ── */
.stButton > button {
  font-family:var(--sans) !important; font-weight:600 !important;
  border-radius:9px !important; transition:all .15s !important;
}
.stButton > button[kind="primary"] {
  background:var(--brand) !important; border:none !important;
  color:#fff !important; font-size:.92rem !important;
  padding:0 20px !important; height:44px !important;
}
.stButton > button[kind="primary"]:hover { background:var(--brand-bright) !important; border:none !important; }
.stButton > button[kind="secondary"] {
  border:1px solid var(--line) !important; border-radius:999px !important;
  font-weight:500 !important; font-size:.82rem !important;
  color:var(--muted) !important; background:var(--surface) !important;
}
.stButton > button[kind="secondary"]:hover {
  border-color:var(--brand) !important; color:var(--ink) !important; background:#fff !important;
}

/* ── slider ── */
[data-testid="stSlider"] { padding:4px 0 !important; }

/* ── metrics ── */
[data-testid="metric-container"] {
  background:var(--surface) !important; border:1px solid var(--line) !important;
  border-radius:var(--radius-sm) !important; padding:16px 18px !important;
  box-shadow:var(--shadow-soft) !important;
}
[data-testid="stMetricValue"] { font-family:var(--mono) !important; color:var(--ink) !important; }
[data-testid="stMetricLabel"] { font-family:var(--sans) !important; color:var(--muted) !important; font-size:.78rem !important; }

/* ── dataframe ── */
[data-testid="stDataFrame"] { border-radius:var(--radius-sm) !important; overflow:hidden !important; }

/* ── spinner ── */
[data-testid="stSpinner"] p { font-family:var(--mono) !important; font-size:.82rem !important; color:var(--muted) !important; }

/* ═══ FL custom components ═══════════════════════════════════════════════ */

/* demo strip */
.fl-demostrip {
  display:flex; align-items:center; gap:10px; flex-wrap:wrap;
  background:var(--brand); color:#CFE3DE;
  font-size:.79rem; border-radius:var(--radius-sm);
  padding:10px 16px; margin:14px 0 0;
}
.fl-demostrip .fl-dot {
  width:7px; height:7px; border-radius:50%; background:var(--hl); flex:none;
  box-shadow:0 0 0 3px rgba(247,218,116,.18);
}
.fl-demostrip b { color:#fff; font-weight:600; }
.fl-demostrip code {
  font-family:var(--mono); font-size:.74rem; color:#EAF3F0;
  background:rgba(255,255,255,.08); padding:1px 6px; border-radius:5px;
}
.fl-demostrip a { color:#fff; text-decoration:underline; text-underline-offset:2px; }

/* hero */
.fl-hero { max-width:760px; margin:22px 0 28px; }
.fl-eyebrow {
  font:600 .72rem/1 var(--mono); letter-spacing:.14em; text-transform:uppercase;
  color:var(--good); display:flex; align-items:center; gap:8px; margin-bottom:14px;
}
.fl-eyebrow::before { content:""; width:22px; height:1px; background:var(--good); display:block; }
.fl-hero h1 {
  font:600 clamp(1.9rem,3.6vw,2.55rem)/1.05 var(--sans);
  letter-spacing:-.025em; margin:0 0 12px; color:var(--ink);
}
.fl-hl {
  background:linear-gradient(180deg,transparent 12%,var(--hl-soft) 12%,var(--hl) 92%,transparent 92%);
  border-radius:2px; padding:0 .06em;
}
.fl-hero p { margin:0; color:var(--muted); font-size:1.02rem; max-width:60ch; }

/* answer card */
.fl-answer-card {
  background:var(--surface); border:1px solid var(--line);
  border-radius:var(--radius); box-shadow:var(--shadow); overflow:hidden;
  animation:fl-rise .45s cubic-bezier(.2,.7,.2,1) both; margin-bottom:8px;
}
@keyframes fl-rise { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:none} }
.fl-answer-top {
  display:flex; align-items:center; justify-content:space-between; gap:12px;
  padding:14px 22px; border-bottom:1px solid var(--line-2); background:var(--surface-2);
}
.fl-answer-eyebrow {
  font:600 .68rem/1 var(--mono); letter-spacing:.16em; text-transform:uppercase;
  color:var(--brand); display:flex; align-items:center; gap:9px;
}
.fl-pin { width:6px; height:6px; border-radius:50%; background:var(--good); display:inline-block; }
.fl-answer-source-tag { font:500 .76rem/1 var(--mono); color:var(--muted); }
.fl-answer-source-tag b { color:var(--ink); font-weight:600; }
.fl-answer-body {
  font-family:var(--serif); font-size:1.11rem; line-height:1.65;
  color:var(--ink); padding:22px 24px 8px;
}
.fl-answer-meta {
  display:flex; flex-wrap:wrap; gap:18px;
  padding:14px 24px 18px; border-top:1px solid var(--line-2);
  margin-top:8px; color:var(--muted);
}
.fl-m { display:flex; align-items:center; gap:7px; font:400 .76rem/1 var(--mono); color:var(--muted); }
.fl-m b { color:var(--ink); font-weight:600; }

/* sources */
.fl-sources-head {
  display:flex; align-items:baseline; justify-content:space-between;
  margin:28px 0 14px;
}
.fl-sources-head h2 { font:600 .92rem/1 var(--sans); letter-spacing:-.01em; margin:0; color:var(--ink); }
.fl-sources-head .fl-hint { font:400 .78rem/1 var(--sans); color:var(--faint); }
.fl-sources { display:flex; flex-direction:column; gap:10px; }
.fl-source {
  background:var(--surface); border:1px solid var(--line);
  border-radius:var(--radius-sm); padding:14px 16px 15px;
  display:grid; grid-template-columns:34px 1fr; gap:14px;
  animation:fl-rise .4s ease both;
}
.fl-source-rank {
  width:34px; height:34px; border-radius:8px;
  background:var(--surface-2); border:1px solid var(--line-2);
  display:grid; place-items:center;
  font:600 .82rem/1 var(--mono); color:var(--brand);
}
.fl-source-head { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:9px; }
.fl-source-ticker {
  font:600 .76rem/1 var(--mono); letter-spacing:.04em; color:#fff;
  background:var(--brand); padding:4px 8px; border-radius:6px;
}
.fl-source-doc { font:500 .8rem/1 var(--sans); color:var(--ink); }
.fl-source-date { font:400 .76rem/1 var(--mono); color:var(--faint); }
.fl-relevance { margin-left:auto; display:flex; align-items:center; gap:8px; }
.fl-rv { font:500 .73rem/1 var(--mono); color:var(--muted); }
.fl-relbar { width:64px; height:5px; border-radius:3px; background:var(--line-2); overflow:hidden; }
.fl-relbar i { display:block; height:100%; background:var(--good); }
.fl-source-text {
  font-family:var(--serif); font-size:.94rem; line-height:1.55;
  color:var(--muted); border-left:2px solid var(--line); padding-left:13px;
}

/* placeholder */
.fl-placeholder {
  border:1px dashed var(--line); border-radius:var(--radius);
  background:rgba(255,255,255,.45); padding:34px 26px;
  text-align:center; color:var(--faint); margin-top:28px;
}
.fl-placeholder p { margin:0; font-size:.92rem; }

/* sidebar panels */
.fl-panel {
  background:var(--surface); border:1px solid var(--line);
  border-radius:var(--radius); box-shadow:var(--shadow-soft);
  margin-bottom:14px; overflow:hidden;
}
.fl-panel-head {
  display:flex; align-items:center; gap:8px;
  padding:13px 16px; border-bottom:1px solid var(--line-2);
}
.fl-panel-head h3 {
  margin:0; font:600 .72rem/1 var(--mono);
  letter-spacing:.14em; text-transform:uppercase; color:var(--muted);
}
.fl-panel-body { padding:6px 16px 14px; }
.fl-row {
  display:flex; align-items:center; justify-content:space-between;
  gap:10px; padding:9px 0; border-bottom:1px solid var(--line-2);
}
.fl-row:last-child { border-bottom:0; }
.fl-row .fl-k { font:400 .8rem/1.2 var(--sans); color:var(--muted); }
.fl-row .fl-v { font:500 .76rem/1 var(--mono); color:var(--ink); display:inline-flex; align-items:center; gap:6px; }
.fl-badge {
  font:500 .66rem/1 var(--mono); color:var(--good);
  background:rgba(44,122,107,.12); padding:3px 6px; border-radius:5px;
}
.fl-corpus { display:flex; flex-wrap:wrap; gap:6px; padding-top:4px; }
.fl-corpus .fl-tk {
  font:600 .72rem/1 var(--mono); letter-spacing:.03em; color:var(--brand);
  background:var(--surface-2); border:1px solid var(--line-2);
  padding:5px 8px; border-radius:6px;
}
.fl-demo-lock {
  display:flex; align-items:center; gap:8px;
  font:400 .76rem/1.4 var(--sans); color:var(--faint);
  padding:11px 16px; background:var(--surface-2);
  border-radius:var(--radius); border:1px solid var(--line-2);
}

/* slider label inside panel */
.fl-slider-head {
  display:flex; align-items:baseline; justify-content:space-between;
  padding:10px 0 2px;
}
.fl-slider-head .fl-k { font:400 .8rem/1 var(--sans); color:var(--muted); }

/* pipeline */
.fl-pipe-intro { max-width:680px; margin:18px 0 22px; }
.fl-pipe-intro h2 { font:600 1.45rem/1.1 var(--sans); letter-spacing:-.02em; margin:0 0 8px; color:var(--ink); }
.fl-pipe-intro p { margin:0; color:var(--muted); font-size:.95rem; }

.fl-pipe-grid {
  display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:0; margin:8px 0 28px;
  background:var(--surface); border:1px solid var(--line);
  border-radius:var(--radius); overflow:hidden; box-shadow:var(--shadow-soft);
}
.fl-stage { padding:18px; border-right:1px solid var(--line-2); position:relative; }
.fl-stage:last-child { border-right:0; }
.fl-stage .fl-num { font:600 .68rem/1 var(--mono); color:var(--faint); letter-spacing:.1em; }
.fl-stage h4 { margin:8px 0 4px; font:600 .92rem/1.2 var(--sans); color:var(--ink); }
.fl-stage p { margin:0; font:400 .78rem/1.45 var(--sans); color:var(--muted); }
.fl-stage .fl-tool {
  margin-top:10px; font:500 .7rem/1 var(--mono); color:var(--brand);
  background:var(--surface-2); border:1px solid var(--line-2);
  display:inline-block; padding:4px 7px; border-radius:5px;
}
.fl-stage.fl-off h4 { color:var(--faint); }
.fl-stage.fl-off::after {
  content:"paused"; position:absolute; top:14px; right:14px;
  font:500 .6rem/1 var(--mono); letter-spacing:.08em; text-transform:uppercase;
  color:var(--faint); background:var(--surface-2); border:1px solid var(--line-2);
  padding:3px 6px; border-radius:4px;
}

.fl-metrics {
  display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:12px; margin-bottom:28px;
}
.fl-metric { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius-sm); padding:16px 18px; box-shadow:var(--shadow-soft); }
.fl-metric .fl-mv { font:600 1.5rem/1 var(--mono); letter-spacing:-.01em; color:var(--ink); font-variant-numeric:tabular-nums; }
.fl-metric .fl-mv small { font-size:.9rem; color:var(--muted); font-weight:500; }
.fl-metric .fl-ml { margin-top:6px; font:400 .78rem/1.3 var(--sans); color:var(--muted); }

/* section heading reuse */
.fl-section-head { font:600 .92rem/1 var(--sans); letter-spacing:-.01em; color:var(--ink); margin:24px 0 12px; }
</style>
""",
)

# ── Demo banner ───────────────────────────────────────────────────────────

if _DEMO_MODE:
    st.html(
        """
    <div class="fl-demostrip">
      <span class="fl-dot"></span>
      <span><b>Demo mode.</b> Read-only over a frozen corpus of ~20 SEC filings
      (Apple, Microsoft, Amazon, Alphabet, NVIDIA).
      Ingestion, <code>dbt</code>, and <code>Dagster</code> are paused.
      <a href="https://github.com/varunsingh09/sec-filing-rag-pipeline">View source on GitHub</a></span>
    </div>
    """,
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
    # Prepend company name + form type + year so company-name query terms
    # (e.g. "Apple") reliably anchor retrieval to the right company.
    # SEC filing bodies use "we"/"the Company" throughout, so without this
    # the company name never appears in the chunk text.
    indexed_texts = [
        f"{c.company_name} {c.form_type} {c.filed_date.year}: {c.text}" for c in chunks
    ]

    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(indexed_texts)

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


# ── Example questions ─────────────────────────────────────────────────────

_EXAMPLES = [
    (
        "MSFT",
        "Microsoft income before taxes",
        "What was Microsoft's income before taxes in fiscal 2025?",
    ),
    ("NVDA", "NVIDIA total revenue", "What was NVIDIA's total revenue in fiscal 2025?"),
    ("AAPL", "Apple risk factors", "What are Apple's main risk factors?"),
]

# ── Sidebar (Engine Rail) ─────────────────────────────────────────────────

with st.sidebar:
    embedder_display = "TF-IDF" if _DEMO_MODE else settings.embedder
    vector_store_display = "In-memory" if _DEMO_MODE else settings.qdrant_location
    badge = '<span class="fl-badge">demo</span>' if _DEMO_MODE else ""

    st.html(
        f"""
    <div class="fl-panel">
      <div class="fl-panel-head">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="3"/>
          <path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2"/>
        </svg>
        <h3>Retrieval engine</h3>
      </div>
      <div class="fl-panel-body">
        <div class="fl-row"><span class="fl-k">Embedder</span><span class="fl-v">{embedder_display} {badge}</span></div>
        <div class="fl-row"><span class="fl-k">Answer model</span><span class="fl-v">Claude</span></div>
        <div class="fl-row"><span class="fl-k">Vector store</span><span class="fl-v">{vector_store_display}</span></div>
      </div>
    </div>

    <div class="fl-panel">
      <div class="fl-panel-head">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="4" y1="8" x2="20" y2="8"/><circle cx="9" cy="8" r="2.4" fill="white"/>
          <line x1="4" y1="16" x2="20" y2="16"/><circle cx="15" cy="16" r="2.4" fill="white"/>
        </svg>
        <h3>Passages per query</h3>
      </div>
      <div class="fl-panel-body">
    """,
    )

    top_k = st.slider("Top-K chunks", 1, 20, settings.top_k, label_visibility="collapsed")

    st.html(
        """
      </div>
    </div>

    <div class="fl-panel">
      <div class="fl-panel-head">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="4" width="18" height="16" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/>
        </svg>
        <h3>Corpus</h3>
      </div>
      <div class="fl-panel-body">
        <div class="fl-corpus">
          <span class="fl-tk">AAPL</span>
          <span class="fl-tk">MSFT</span>
          <span class="fl-tk">AMZN</span>
          <span class="fl-tk">GOOGL</span>
          <span class="fl-tk">NVDA</span>
        </div>
      </div>
    </div>
    """,
    )

    if _DEMO_MODE:
        st.html(
            """
        <div class="fl-demo-lock">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="5" y="11" width="14" height="9" rx="2"/>
            <path d="M8 11V8a4 4 0 018 0v3"/>
          </svg>
          Read-only demo. Connect a backend to query the full corpus.
        </div>
        """,
        )

# ── Tabs ──────────────────────────────────────────────────────────────────

tab_query, tab_analytics = st.tabs(["🔍  Ask a question", "📊  Pipeline analytics"])

# ── Tab 1: Query ──────────────────────────────────────────────────────────

with tab_query:
    st.html(
        """
    <div class="fl-hero">
      <div class="fl-eyebrow">Grounded retrieval</div>
      <h1>Ask the filings. Get answers you can <span class="fl-hl">trace to the source</span>.</h1>
      <p>Natural-language questions across SEC 10-K and 10-Q documents.
         Every answer is grounded in the exact passages it was retrieved from.</p>
    </div>
    """,
    )

    # Initialise session state for the question input
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""

    question = st.text_input(
        "Your question",
        value=st.session_state["question_input"],
        placeholder="e.g. What was Apple's revenue in fiscal 2025?",
        label_visibility="collapsed",
    )

    # Example chips
    ex_cols = st.columns(len(_EXAMPLES))
    run_example = False
    for col, (tk, label, q) in zip(ex_cols, _EXAMPLES):
        if col.button(f"{tk} · {label}", use_container_width=True):
            question = q
            st.session_state["question_input"] = q
            run_example = True

    st.write("")
    search_btn = st.button("Ask →", type="primary", use_container_width=False)

    should_run = (search_btn or run_example) and question.strip()

    if not should_run:
        st.html(
            """
        <div class="fl-placeholder">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="1.6" style="margin-bottom:10px;opacity:.6">
            <rect x="4" y="3" width="13" height="18" rx="2"/>
            <line x1="7" y1="8" x2="14" y2="8"/>
            <line x1="7" y1="12" x2="14" y2="12"/>
            <line x1="7" y1="16" x2="11" y2="16"/>
          </svg>
          <p>Ask a question to retrieve cited passages from the filing corpus.</p>
        </div>
        """,
        )

    if should_run:
        if _DEMO_MODE:
            # ── Demo path: TF-IDF retrieval + Anthropic generation ────────
            searcher = _get_demo_searcher()
            if searcher is None:
                st.error("No chunks found in demo warehouse — index is empty.")
            else:
                with st.spinner("Retrieving passages and synthesizing a cited answer…"):
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

                # ── Answer card ───────────────────────────────────────────
                if answer:
                    top_chunk = retrieved[0].chunk if retrieved else None
                    source_tag = ""
                    if top_chunk:
                        tk_label = _ticker(top_chunk.company_name)
                        source_tag = (
                            f"grounded in <b>{tk_label} {top_chunk.form_type}</b>"
                            f" · {top_chunk.filed_date}"
                        )

                    answer_html = _html.escape(answer).replace("\n", "<br>")
                    st.html(
                        f"""
                    <div class="fl-answer-card">
                      <div class="fl-answer-top">
                        <span class="fl-answer-eyebrow">
                          <span class="fl-pin"></span>Answer
                        </span>
                        <span class="fl-answer-source-tag">{source_tag}</span>
                      </div>
                      <div class="fl-answer-body">{answer_html}</div>
                      <div class="fl-answer-meta">
                        <span class="fl-m">
                          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2" stroke-linecap="round"/>
                          </svg>
                          <b>{latency_ms:.0f}</b> ms
                        </span>
                        <span class="fl-m">
                          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="4" y1="7" x2="20" y2="7"/>
                            <line x1="4" y1="12" x2="20" y2="12"/>
                            <line x1="4" y1="17" x2="14" y2="17"/>
                          </svg>
                          <b>{len(retrieved)}</b> passages
                        </span>
                        <span class="fl-m">
                          model <b>Claude</b>
                        </span>
                      </div>
                    </div>
                    """,
                    )

                # ── Source chunks ─────────────────────────────────────────
                if retrieved:
                    st.html(
                        f"""
                    <div class="fl-sources-head">
                      <h2>Retrieved passages</h2>
                      <span class="fl-hint">{len(retrieved)} chunk{"s" if len(retrieved) != 1 else ""} · ranked by relevance</span>
                    </div>
                    """,
                    )

                    cards_html = '<div class="fl-sources">'
                    for i, rc in enumerate(retrieved, 1):
                        c = rc.chunk
                        pct = min(int(rc.score * 100), 100)
                        delay = (i - 1) * 70
                        snippet = _html.escape(c.text[:600]) + ("…" if len(c.text) > 600 else "")
                        cards_html += f"""
                        <article class="fl-source" style="animation-delay:{delay}ms">
                          <div class="fl-source-rank">{i}</div>
                          <div>
                            <div class="fl-source-head">
                              <span class="fl-source-ticker">{_ticker(c.company_name)}</span>
                              <span class="fl-source-doc">{_html.escape(c.form_type)}</span>
                              <span class="fl-source-date">{c.filed_date}</span>
                              <span class="fl-relevance">
                                <span class="fl-rv">{rc.score:.2f}</span>
                                <span class="fl-relbar"><i style="width:{pct}%"></i></span>
                              </span>
                            </div>
                            <div class="fl-source-text">{snippet}</div>
                          </div>
                        </article>
                        """
                    cards_html += "</div>"
                    st.html(cards_html)

        else:
            # ── Full pipeline path ────────────────────────────────────────
            embedder = _get_embedder()
            store = _get_store()

            from src.rag.pipeline import RAGPipeline
            from src.store.warehouse import Warehouse

            wh = Warehouse()
            with st.spinner("Searching…"):
                pipeline = RAGPipeline(embedder, store, wh)
                result = pipeline.query(question, top_k=top_k)
            wh.close()

            if result.answer:
                top_chunk = result.retrieved_chunks[0].chunk if result.retrieved_chunks else None
                source_tag = ""
                if top_chunk:
                    tk_label = _ticker(top_chunk.company_name)
                    source_tag = (
                        f"grounded in <b>{tk_label} {top_chunk.form_type}</b>"
                        f" · {top_chunk.filed_date}"
                    )

                answer_html = _html.escape(result.answer).replace("\n", "<br>")
                st.html(
                    f"""
                <div class="fl-answer-card">
                  <div class="fl-answer-top">
                    <span class="fl-answer-eyebrow">
                      <span class="fl-pin"></span>Answer
                    </span>
                    <span class="fl-answer-source-tag">{source_tag}</span>
                  </div>
                  <div class="fl-answer-body">{answer_html}</div>
                  <div class="fl-answer-meta">
                    <span class="fl-m">
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2" stroke-linecap="round"/>
                      </svg>
                      <b>{result.latency_ms:.0f}</b> ms
                    </span>
                    <span class="fl-m">
                      <b>{len(result.retrieved_chunks)}</b> passages
                    </span>
                    <span class="fl-m">model <b>Claude</b></span>
                  </div>
                </div>
                """,
                )

            if result.retrieved_chunks:
                st.html(
                    f"""
                <div class="fl-sources-head">
                  <h2>Retrieved passages</h2>
                  <span class="fl-hint">{len(result.retrieved_chunks)} chunks · ranked by relevance</span>
                </div>
                """,
                )

                cards_html = '<div class="fl-sources">'
                for i, rc in enumerate(result.retrieved_chunks, 1):
                    c = rc.chunk
                    pct = min(int(rc.score * 100), 100)
                    delay = (i - 1) * 70
                    snippet = _html.escape(c.text[:600]) + ("…" if len(c.text) > 600 else "")
                    cards_html += f"""
                    <article class="fl-source" style="animation-delay:{delay}ms">
                      <div class="fl-source-rank">{i}</div>
                      <div>
                        <div class="fl-source-head">
                          <span class="fl-source-ticker">{_ticker(c.company_name)}</span>
                          <span class="fl-source-doc">{_html.escape(c.form_type)}</span>
                          <span class="fl-source-date">{c.filed_date}</span>
                          <span class="fl-relevance">
                            <span class="fl-rv">{rc.score:.2f}</span>
                            <span class="fl-relbar"><i style="width:{pct}%"></i></span>
                          </span>
                        </div>
                        <div class="fl-source-text">{snippet}</div>
                      </div>
                    </article>
                    """
                cards_html += "</div>"
                st.html(cards_html)

# ── Tab 2: Analytics ──────────────────────────────────────────────────────

with tab_analytics:
    st.html(
        """
    <div class="fl-pipe-intro">
      <h2>Pipeline analytics</h2>
      <p>The retrieval layer sits on top of an ingestion pipeline that pulls filings from EDGAR,
         models them with dbt, and orchestrates runs with Dagster.
         In demo mode the upstream stages are frozen.</p>
    </div>
    """,
    )

    # Pipeline stages
    st.html(
        """
    <div class="fl-pipe-grid">
      <div class="fl-stage fl-off">
        <div class="fl-num">01</div>
        <h4>Ingest</h4>
        <p>Pull 10-K and 10-Q filings from the SEC EDGAR feed and stage raw HTML.</p>
        <span class="fl-tool">EDGAR API</span>
      </div>
      <div class="fl-stage fl-off">
        <div class="fl-num">02</div>
        <h4>Model</h4>
        <p>Parse sections, normalize financials, and build clean tables.</p>
        <span class="fl-tool">dbt</span>
      </div>
      <div class="fl-stage fl-off">
        <div class="fl-num">03</div>
        <h4>Orchestrate</h4>
        <p>Schedule and monitor the ingest and model jobs end to end.</p>
        <span class="fl-tool">Dagster</span>
      </div>
      <div class="fl-stage">
        <div class="fl-num">04</div>
        <h4>Chunk + embed</h4>
        <p>Split passages and build the TF-IDF index served to retrieval.</p>
        <span class="fl-tool">TF-IDF</span>
      </div>
      <div class="fl-stage">
        <div class="fl-num">05</div>
        <h4>Retrieve + answer</h4>
        <p>Rank passages by query similarity and synthesize a cited answer.</p>
        <span class="fl-tool">Claude</span>
      </div>
    </div>
    """,
    )

    data = _load_analytics()
    if "error" in data:
        st.info(f"Warehouse not yet populated. ({data['error']})")
    else:
        # Metrics grid from warehouse counts
        counts = data["counts"]

        def _count(tbl: str) -> int:
            rows = counts[counts["tbl"] == tbl]["rows"]
            return int(rows.values[0]) if len(rows) else 0

        n_filings = _count("raw_filings")
        n_chunks = _count("raw_chunks")
        n_queries = _count("raw_query_logs")

        st.html(
            f"""
        <div class="fl-metrics">
          <div class="fl-metric">
            <div class="fl-mv">{n_filings}</div>
            <div class="fl-ml">Filings indexed</div>
          </div>
          <div class="fl-metric">
            <div class="fl-mv">5</div>
            <div class="fl-ml">Companies covered</div>
          </div>
          <div class="fl-metric">
            <div class="fl-mv">{n_chunks:,}</div>
            <div class="fl-ml">Passages embedded</div>
          </div>
          <div class="fl-metric">
            <div class="fl-mv">{n_queries}</div>
            <div class="fl-ml">Queries logged</div>
          </div>
        </div>
        """,
        )

        if not data["filings"].empty:
            st.html(
                '<p class="fl-section-head">Filings by company &amp; year</p>',
            )
            st.dataframe(data["filings"], use_container_width=True, hide_index=True)

        if not data["recent_queries"].empty:
            st.html('<p class="fl-section-head">Recent queries</p>')
            st.dataframe(data["recent_queries"], use_container_width=True, hide_index=True)

        sweep_path = _PROJECT_ROOT / "eval_results" / "sweep_results.json"
        if sweep_path.exists():
            import pandas as pd
            import plotly.express as px

            sweep = pd.DataFrame(json.loads(sweep_path.read_text()))
            st.html(
                '<p class="fl-section-head">Retrieval eval: hit rate vs top_k</p>',
            )
            fig = px.line(
                sweep,
                x="top_k",
                y="hit_rate",
                markers=True,
                labels={"hit_rate": "Hit Rate", "top_k": "top_k"},
                color_discrete_sequence=["#163A36"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_family="IBM Plex Sans",
                margin=dict(l=0, r=0, t=8, b=0),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#D9DEE4", zeroline=False)
            fig.update_yaxes(showgrid=True, gridcolor="#D9DEE4", zeroline=False)
            st.plotly_chart(fig, use_container_width=True)
