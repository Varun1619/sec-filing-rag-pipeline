.PHONY: install ingest build dbt eval test app clean

# ── Setup ─────────────────────────────────────────────────────────────────
install:
	pip install -e ".[st,eval,dev]"

install-all:
	pip install -e ".[st,openai,anthropic,groq,spacy,sentence-transformers,eval,dev]"

# ── Pipeline stages ────────────────────────────────────────────────────────
# Fetch 3 filings per company from 3 well-known companies (offline-safe sample)
ingest:
	python -m scripts.run_pipeline ingest \
		--ciks 0000320193,0001018724,0001652044 \
		--forms 10-K,10-Q \
		--max 3

# Parse, chunk, embed, index (uses offline hashing embedder by default)
build:
	python -m scripts.run_pipeline build

# Run a sample query
query:
	python -m scripts.run_pipeline query "What was Apple's total revenue?"

# ── dbt ───────────────────────────────────────────────────────────────────
dbt:
	cd dbt && dbt build --profiles-dir . --project-dir .

dbt-docs:
	cd dbt && dbt docs generate --profiles-dir . --project-dir . && dbt docs serve --profiles-dir . --project-dir .

dbt-test:
	cd dbt && dbt test --profiles-dir . --project-dir .

# ── Evaluation ────────────────────────────────────────────────────────────
eval-set:
	python scripts/make_eval_set.py

eval:
	python -m scripts.run_pipeline eval

# ── Tests ─────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Streamlit app ─────────────────────────────────────────────────────────
app:
	streamlit run app.py

# ── Dagster UI ────────────────────────────────────────────────────────────
dagster-ui:
	dagster dev -f dagster_defs/__init__.py

# ── Utility ───────────────────────────────────────────────────────────────
clean:
	rm -rf data/ qdrant_data/ warehouse.duckdb logs/ eval_results/ dbt/target/ dbt/dbt_packages/

# Full end-to-end offline run (no API keys required)
e2e: ingest build dbt eval test
	@echo "✓ End-to-end pipeline complete."
