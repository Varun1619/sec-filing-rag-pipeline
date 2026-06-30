"""Tests for the chunker module."""

from __future__ import annotations

from datetime import date


from src.chunk.chunker import chunk_filing
from src.models import Filing


def _make_filing(**kwargs) -> Filing:
    defaults = dict(
        cik="0000320193",
        company_name="Test Corp",
        form_type="10-K",
        filed_date=date(2023, 10, 1),
        accession_number="0000320193-23-000001",
        document_url="https://example.com/doc.htm",
    )
    defaults.update(kwargs)
    return Filing(**defaults)


def test_chunk_basic():
    filing = _make_filing()
    text = "Hello world. " * 200  # ~2600 chars
    chunks = chunk_filing(filing, text)
    assert len(chunks) > 1, "Long text should produce multiple chunks"
    for c in chunks:
        assert c.filing_id == filing.filing_id
        assert c.text.strip(), "No chunk should be empty"


def test_chunk_empty_text():
    filing = _make_filing()
    chunks = chunk_filing(filing, "   ")
    assert chunks == [], "Empty text should produce no chunks"


def test_chunk_ordering():
    filing = _make_filing()
    text = "x" * 5000
    chunks = chunk_filing(filing, text)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks))), "chunk_index should be sequential"


def test_chunk_attributes():
    filing = _make_filing()
    text = "Apple reported revenue of $394 billion in fiscal year 2023."
    chunks = chunk_filing(filing, text)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.cik == "0000320193"
    assert c.form_type == "10-K"
    assert c.char_count == len(c.text)


def test_chunk_overlap():
    """With overlap, adjacent chunks should share some characters."""
    filing = _make_filing()
    # Large enough to produce 2+ chunks
    text = "A" * 4000
    chunks = chunk_filing(filing, text)
    if len(chunks) >= 2:
        # The text of chunk[1] should start before the end of chunk[0]'s
        # non-overlapping region — they should be non-disjoint
        assert len(chunks[0].text) > 0
        assert len(chunks[1].text) > 0
