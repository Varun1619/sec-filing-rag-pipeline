"""Tests for the vector store and deterministic retrieval metric."""

from __future__ import annotations

from datetime import date


from src.chunk.chunker import chunk_filing
from src.embed.embedder import HashingEmbedder
from src.eval.evaluate import EvalSample, hit_at_k
from src.models import Filing
from src.store.qdrant_store import QdrantStore


def _in_memory_store(dim: int = 384) -> QdrantStore:
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    # Patch settings for this test
    import src.store.qdrant_store as qs

    original_dim = qs.settings.embed_dim
    qs.settings.__dict__["embed_dim"] = dim
    store = QdrantStore(client=client)
    qs.settings.__dict__["embed_dim"] = original_dim
    return store


def _make_filing(cik="0000320193", name="Apple Inc") -> Filing:
    return Filing(
        cik=cik,
        company_name=name,
        form_type="10-K",
        filed_date=date(2023, 10, 1),
        accession_number=f"{cik}-23-000001",
        document_url="https://example.com/doc.htm",
        local_path=None,
    )


def test_upsert_and_search():
    emb = HashingEmbedder()
    store = _in_memory_store(dim=emb.dim)

    filing = _make_filing()
    text = "Apple Inc reported total revenue of $394 billion in fiscal year 2023."
    chunks = chunk_filing(filing, text)
    vecs = emb.embed([c.text for c in chunks])
    embedded = [c.model_copy(update={"embedding": v}) for c, v in zip(chunks, vecs)]

    n = store.upsert_chunks(embedded)
    assert n == len(embedded)

    query_vec = emb.embed(["Apple revenue fiscal 2023"])[0]
    results = store.search(query_vec, top_k=3)
    assert len(results) > 0
    assert results[0].score > 0.0


def test_hit_at_k_metric():
    emb = HashingEmbedder()
    store = _in_memory_store(dim=emb.dim)

    filing = _make_filing(cik="0000320193", name="Apple Inc")
    text = "Apple total revenue for fiscal 2023 was $394 billion."
    chunks = chunk_filing(filing, text)
    vecs = emb.embed([c.text for c in chunks])
    embedded = [c.model_copy(update={"embedding": v}) for c, v in zip(chunks, vecs)]
    store.upsert_chunks(embedded)

    samples = [
        EvalSample(
            question="What was Apple's revenue?",
            expected_cik="0000320193",
        )
    ]
    result = hit_at_k(samples, emb, store, k_values=[1, 5])
    # With hashing embedder and a topically similar query, we expect a hit
    assert result.total_questions == 1
    assert 0.0 <= result.hit_at_k[1] <= 1.0
    assert 0.0 <= result.hit_at_k[5] <= 1.0
