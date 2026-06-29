# Architecture

## System Diagram

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion (Bronze)"]
        EDGAR["SEC EDGAR API\n(submissions + filings)"] --> DL["edgar.py\nRate-limited downloader\nIdempotent + Incremental"]
        DL --> BRONZE["data/bronze/\n{cik}/{accession}/"]
        DL --> RAW_F["DuckDB: raw_filings"]
    end

    subgraph Processing["Processing (Silver)"]
        BRONZE --> PARSE["parse.py\nHTML → text\nPDF → text"]
        PARSE --> CHUNK["chunker.py\nFixed-size windows\nwith overlap"]
        CHUNK --> ENTITY["entities.py\nRegex NER\n(tickers, money, dates)"]
        ENTITY --> EMBED["embedder.py\nHashing (offline)\nor SBERT / OpenAI"]
        EMBED --> QDRANT["Qdrant\nVector Store\n(embedded or server)"]
        EMBED --> RAW_C["DuckDB: raw_chunks"]
    end

    subgraph Transform["Transformation (Gold — dbt)"]
        RAW_F --> STG_F["stg_filings"]
        RAW_C --> STG_C["stg_chunks"]
        STG_F --> DIM_CO["dim_company"]
        STG_F --> DIM_FI["dim_filing"]
        STG_C --> FACT_C["fact_chunks"]
        QLOGS["DuckDB: raw_query_logs"] --> STG_Q["stg_query_logs"]
        STG_Q --> FACT_Q["fact_query_logs"]
    end

    subgraph Query["Query Layer"]
        Q["User Question"] --> EMBED2["Embed query\n(same backend)"]
        EMBED2 --> QDRANT
        QDRANT --> RETRIEVE["Top-K chunks"]
        RETRIEVE --> LLM["LLM (optional)\nnone / openai / anthropic / groq"]
        LLM --> ANSWER["Answer + Citations"]
        RETRIEVE --> QLOGS
    end

    subgraph Observability["Observability"]
        DAGSTER["Dagster\nAsset lineage\n+ schedules"]
        STREAMLIT["Streamlit\nQuery UI\n+ Analytics"]
        EVAL["evaluate.py\nHit@k sweep\n+ RAGAS (optional)"]
        FACT_Q --> STREAMLIT
        FACT_C --> STREAMLIT
    end
```

## ERD (DuckDB tables)

```mermaid
erDiagram
    raw_filings {
        VARCHAR filing_id PK
        VARCHAR cik
        VARCHAR company_name
        VARCHAR form_type
        DATE filed_date
        DATE period_of_report
        VARCHAR accession_number
        VARCHAR document_url
        VARCHAR local_path
        BIGINT file_size_bytes
        BOOLEAN is_scanned_pdf
        TIMESTAMP ingested_at
    }

    raw_chunks {
        VARCHAR chunk_id PK
        VARCHAR filing_id FK
        VARCHAR cik
        VARCHAR company_name
        VARCHAR form_type
        DATE filed_date
        INTEGER chunk_index
        VARCHAR text
        INTEGER char_count
        VARCHAR entities_json
        VARCHAR embedding_json
        TIMESTAMP chunked_at
    }

    raw_query_logs {
        VARCHAR query_id PK
        VARCHAR question
        VARCHAR chunk_ids_json
        VARCHAR scores_json
        VARCHAR answer
        DOUBLE latency_ms
        TIMESTAMP queried_at
    }

    raw_filings ||--o{ raw_chunks : "has"
```

## dbt Lineage

```
raw_filings ──► stg_filings ──► dim_company ──┐
                          └──► dim_filing  ──┤
                                              ├──► (dashboards)
raw_chunks  ──► stg_chunks  ──► fact_chunks ──┤
raw_query_logs ► stg_query_logs ► fact_query_logs ─┘
```

## Medallion Layers

| Layer  | Location | Contents |
|--------|----------|----------|
| Bronze | `data/bronze/{cik}/{accession}/` + `raw_filings` | Raw HTML/PDF files + filing metadata |
| Silver | `raw_chunks` → `stg_chunks` → `fact_chunks` | Parsed, chunked, entity-enriched, embedded text |
| Gold   | dbt marts (`dim_*`, `fact_*`) | Analytics-ready star schema |
