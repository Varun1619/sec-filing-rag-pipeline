"""
Text chunker — splits filing text into overlapping windows (silver layer).

Design: fixed-size character windows with overlap.  "Tokens" are approximated
as chars/4 to avoid a tokenizer dependency on the offline path.  The chunker
is stateless so it can be parallelised trivially.
"""

from __future__ import annotations

from src.config import settings
from src.ingest.entities import extract_entities
from src.logging_utils import get_logger
from src.models import Chunk, Filing

logger = get_logger(__name__)

# chars per approximate token (good enough for English SEC text)
_CHARS_PER_TOKEN = 4


def chunk_filing(filing: Filing, text: str) -> list[Chunk]:
    """
    Split *text* into overlapping Chunk objects attributed to *filing*.

    Returns an empty list if text is blank (e.g. scanned PDF).
    """
    if not text.strip():
        logger.warning("Empty text for filing", extra={"filing_id": filing.filing_id})
        return []

    chunk_chars = settings.chunk_size * _CHARS_PER_TOKEN
    overlap_chars = settings.chunk_overlap * _CHARS_PER_TOKEN
    step = max(chunk_chars - overlap_chars, 1)

    chunks: list[Chunk] = []
    idx = 0
    start = 0

    while start < len(text):
        end = min(start + chunk_chars, len(text))
        window = text[start:end].strip()
        if window:
            entities = extract_entities(window)
            chunk = Chunk(
                filing_id=filing.filing_id,
                cik=filing.cik,
                company_name=filing.company_name,
                form_type=filing.form_type,
                filed_date=filing.filed_date,
                chunk_index=idx,
                text=window,
                entities=entities,
            )
            chunks.append(chunk)
            idx += 1
        start += step

    logger.info(
        "Chunked filing",
        extra={
            "filing_id": filing.filing_id,
            "company": filing.company_name,
            "text_chars": len(text),
            "chunks": len(chunks),
        },
    )
    return chunks
