<div align="center">

<img src="docs/header.svg" alt="FilingLens — SEC Filing RAG Pipeline" width="900"/>

<br/>

[![CI](https://img.shields.io/github/actions/workflow/status/Varun1619/sec-filing-rag-pipeline/ci.yml?branch=main&style=flat&label=CI&color=163A36)](https://github.com/Varun1619/sec-filing-rag-pipeline/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/demo-live%20on%20Streamlit-163A36?style=flat&logo=streamlit&logoColor=F7DA74)](https://sec-filing-rag-pipeline.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-163A36?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![dbt](https://img.shields.io/badge/dbt-semantic%20layer-163A36?style=flat&logo=dbt&logoColor=F7DA74)](https://www.getdbt.com)
[![Dagster](https://img.shields.io/badge/Dagster-orchestration-163A36?style=flat)](https://dagster.io)

</div>

---

A production-grade data engineering pipeline that ingests SEC EDGAR filings, parses and chunks them, embeds with pluggable backends, and makes them queryable in natural language — built as a proper data engineering project with a DuckDB warehouse, dbt semantic layer, Dagster orchestration, evaluation harness, and Streamlit dashboard.

## Live Demo

**[Try the live demo on Streamlit Community Cloud](https://sec-filing-rag-pipeline.streamlit.app)**

Ask natural-language questions across 20 SEC filings (Apple, Microsoft, Amazon, Alphabet, NVIDIA — 2025–2026). The demo uses a TF-IDF retrieval index built from 2,342 chunks at startup and Anthropic claude-haiku for answer generation. No login required.

Example questions to try:
- *What was Microsoft's income before taxes in fiscal 2025?*
- *What was NVIDIA's total revenue in fiscal 2025?*
- *What are Apple's main risk factors?*

## Live Demo & Deployment

### Streamlit Community Cloud

The app runs in **read-only demo mode** backed by a pre-built dataset of 20 filings
(Apple, Microsoft, Amazon, Alphabet, NVIDIA) committed to `demo_assets/warehouse.duckdb`.
At startup it builds an in-memory TF-IDF index over 2,342 chunks — no model download.

**Deploy steps:**

1. Fork / push this repo to a public GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app".
3. Select this repo, branch `main`, main file `app.py`.
4. In **Advanced settings → Python version**, set **3.11**.
5. In **Advanced settings → Secrets**, add:
   ```toml
   SEC_DEMO_MODE = "true"
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
6. Click Deploy — the app starts serving immediately with no ingestion step.

**To rebuild the demo dataset locally:**
```bash
python scripts/build_demo_data.py          # writes demo_assets/warehouse.duckdb
git add demo_assets/warehouse.duckdb
git commit -m "chore: rebuild demo dataset"
git push
```
The script reads already-downloaded filings from `data/bronze/` (sorted newest-first) and falls back to fetching from EDGAR if the cache is empty. Use `--embedder hashing` (default) for speed — stored embeddings are not used by the TF-IDF demo mode.

### Self-hosted (Docker + Qdrant)

```bash
# Run Qdrant locally
docker compose up -d qdrant

# Full pipeline
SEC_EMBEDDER=sentence_transformers
SEC_QDRANT_LOCATION=http://localhost:6333
python -m scripts.run_pipeline ingest && python -m scripts.run_pipeline build
streamlit run app.py
```

### CI (GitHub Actions)

Push to `main` triggers `.github/workflows/ci.yml`:
- **black --check** + **ruff check** — code quality gates
- **pytest** (21 tests, offline, in-memory Qdrant + hashing embedder)
- **dbt build** — seeds a minimal warehouse and runs all models + 30 data tests

## Architecture

```mermaid
flowchart LR
    EDGAR["SEC EDGAR\nAPI"] -->|rate-limited\nidempotent| BRONZE["Bronze\nraw filings"]
    BRONZE -->|parse + chunk\n+ entities| SILVER["Silver\nchunks + embeddings"]
    SILVER -->|dbt| GOLD["Gold\nstar schema"]
    SILVER -->|upsert| QDRANT["Qdrant\nvector store"]
    QDRANT -->|retrieve| RAG["RAG pipeline"]
    GOLD --> DASH["Streamlit\ndashboard"]
    RAG --> DASH
```

See [docs/architecture.md](docs/architecture.md) for the full Mermaid diagram and ERD.

## Quickstart (offline — no API keys, no model downloads)

```bash
# 1. Install (Python 3.11+)
pip install -e ".[st,eval,dev]"

# 2. Configure (copy and edit as needed — defaults work offline)
cp .env.example .env

# 3. Ingest 3 companies × 3 filings from EDGAR
python -m scripts.run_pipeline ingest

# 4. Parse, chunk, embed (uses offline HashingVectorizer), index in Qdrant
python -m scripts.run_pipeline build

# 5. Run dbt transformations
cd dbt && dbt build --profiles-dir . --project-dir . && cd ..

# 6. Run the deterministic retrieval eval
python -m scripts.run_pipeline eval

# 7. Run tests
pytest

# 8. Launch the Streamlit dashboard
streamlit run app.py
```

Or use `make e2e` to run steps 3–7 in one shot.

## Scaling to 500+ Filings

The `--ciks` and `--max` parameters control the scope:

```bash
# Fetch up to 20 filings each for 30 companies
python -m scripts.run_pipeline ingest \
  --ciks 0000320193,0001018724,0001652044,...  \
  --max 20

# Rebuild the index (idempotent — skips already-downloaded filings)
python -m scripts.run_pipeline build
```

The pipeline is **incremental**: a watermark file (`data/.watermark`) records the last-ingested filing date; subsequent runs only fetch newer filings.

## Switching Backends via Environment Variables

All backends are swapped in `.env` — no code changes:

| What | Variable | Options |
|------|----------|---------|
| Embedder | `SEC_EMBEDDER` | `hashing` (default, offline), `fastembed` (ONNX, no extra deps), `sentence_transformers`, `openai` |
| LLM | `SEC_LLM_PROVIDER` | `none` (default), `openai`, `anthropic`, `groq` |
| Vector store | `SEC_QDRANT_LOCATION` | `./qdrant_data` (default), `:memory:`, `http://localhost:6333` |
| Warehouse | `SEC_DUCKDB_PATH` | `./warehouse.duckdb` (swap dbt profile for Snowflake) |

### Example: switch to semantic embeddings + Claude

```bash
SEC_EMBEDDER=sentence_transformers
SEC_LLM_PROVIDER=anthropic
SEC_LLM_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=sk-ant-...
```

## Design Decisions & Tradeoffs

**Why DuckDB instead of Snowflake?**  
DuckDB runs locally with zero infrastructure and is fast enough for millions of rows. The dbt profile is the only change needed to point at Snowflake — the SQL models are identical.

**Why embedded Qdrant instead of Qdrant Cloud?**  
`qdrant-client` supports an embedded persistent mode (`path=./qdrant_data`) with no server process. The application code is identical to the Docker/Cloud mode — swap `SEC_QDRANT_LOCATION` and you're done.

**Why TF-IDF for the Streamlit demo instead of a neural embedder?**  
`fastembed` (ONNX) and `sentence-transformers` both failed to install reliably on Streamlit Community Cloud due to binary dependency conflicts (`onnxruntime`, `pyarrow`). A scikit-learn `TfidfVectorizer` fitted on the corpus at startup is guaranteed to work, has IDF weighting (common financial boilerplate is downweighted), and adds zero install-time dependencies. Company names are prepended to each indexed chunk so company-specific queries anchor correctly. For local use, switch to `sentence_transformers` for full semantic retrieval.

**Why HashingVectorizer as the default offline embedder?**  
It has zero cold-start (no model download), works fully offline, and is deterministic. It is not semantically meaningful but makes the pipeline runnable immediately. Switch to `sentence_transformers` or `fastembed` for real retrieval quality.

**Why is generation optional?**  
`SEC_LLM_PROVIDER=none` (the default) returns ranked chunks without generation. This keeps the pipeline runnable without any API key and makes retrieval quality independently measurable.

**Idempotency & incrementality**  
Re-running `ingest` skips already-downloaded accession numbers (checked by local directory). A watermark file persists the last-seen `filed_date` so only new filings are fetched on subsequent runs.

**Data quality**  
dbt `schema.yml` tests (`not_null`, `unique`, `relationships`) run on every `dbt build`. Custom staging WHERE clauses reject empty chunks. Row-count reconciliation (filings in → chunks out) is logged on every pipeline run.

## What Requires an API Key

| Feature | Key needed |
|---------|-----------|
| Ingest, parse, chunk, embed, index | None |
| Deterministic retrieval eval (Hit@k) | None |
| Streamlit dashboard (analytics tab) | None |
| Sentence-transformers / fastembed embeddings | None (downloads model on first run) |
| LLM-generated answers | OpenAI / Anthropic / Groq key |
| RAGAS evaluation (faithfulness) | OpenAI key |

## Project Structure

```
sec-filing-rag-pipeline/
├── src/
│   ├── config.py           # pydantic-settings, all backends swappable via .env
│   ├── models.py           # Pydantic: Filing, Chunk, Entity, QueryResult
│   ├── logging_utils.py    # structured JSON logging
│   ├── ingest/             # EDGAR downloader, HTML/PDF parser, entity extractor
│   ├── chunk/              # fixed-window chunker with overlap
│   ├── embed/              # pluggable embedder (hashing / fastembed / SBERT / OpenAI)
│   ├── store/              # Qdrant wrapper + DuckDB warehouse
│   ├── rag/                # query pipeline (retrieve + optional generate)
│   └── eval/               # Hit@k sweep + RAGAS wiring
├── dagster_defs/           # Dagster software-defined assets
├── dbt/                    # dbt project: staging + marts, schema tests
├── scripts/                # CLI entry point, demo data builder, eval set generator
├── tests/                  # pytest unit tests (21 tests, fully offline)
├── demo_assets/            # pre-built warehouse.duckdb (20 filings, 2,342 chunks)
├── eval_data/              # labeled QA set
├── docs/                   # architecture + ERD diagrams
├── app.py                  # Streamlit dashboard
├── Makefile
├── pyproject.toml
├── requirements.txt        # full pipeline deps (local dev / CI)
├── requirements-app.txt    # slim deps for Streamlit Cloud
├── .env.example
└── docker-compose.yml      # optional Qdrant server
```
