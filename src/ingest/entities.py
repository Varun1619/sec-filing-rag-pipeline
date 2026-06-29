"""
Entity extraction from SEC filing text.

Uses regex/heuristics tuned for SEC documents. An optional spaCy NER path
is available behind the SEC_USE_SPACY flag (requires: pip install sec-filing-rag-pipeline[spacy]).

Extracted entity types:
  TICKER        — stock symbols like $AAPL or (NASDAQ: GOOGL)
  CIK           — SEC registrant identifiers
  MONEY         — dollar amounts ($1.2B, $500 million, etc.)
  FISCAL_PERIOD — Q1/Q2/Q3/Q4 + year references
  DATE          — standalone dates in the text
  METRIC        — common financial KPIs (revenue, EPS, EBITDA …)
"""

from __future__ import annotations

import os
import re

from src.logging_utils import get_logger
from src.models import Entity

logger = get_logger(__name__)

# ── Regex patterns ─────────────────────────────────────────────────────────

_TICKER_RE = re.compile(
    r"""
    (?:\$([A-Z]{1,5}))                       # $AAPL
    |(?:\((?:NYSE|NASDAQ|AMEX):\s*([A-Z]{1,5})\))  # (NYSE: AAPL)
    """,
    re.VERBOSE,
)

_CIK_RE = re.compile(r"\bCIK[:\s#]*(\d{7,10})\b", re.IGNORECASE)

_MONEY_RE = re.compile(
    r"""
    \$\s*                            # leading dollar sign
    (\d{1,3}(?:,\d{3})*(?:\.\d+)?)  # numeric part
    \s*
    (billion|million|thousand|B|M|K)?  # optional scale
    """,
    re.VERBOSE | re.IGNORECASE,
)

_FISCAL_PERIOD_RE = re.compile(
    r"\b(Q[1-4])\s*(?:of\s*)?(\d{4})\b"
    r"|fiscal\s+(?:year\s+)?(\d{4})\b",
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    re.IGNORECASE,
)

_METRIC_KEYWORDS = {
    "revenue", "net income", "earnings per share", "eps", "ebitda",
    "gross profit", "operating income", "cash flow", "total assets",
    "total liabilities", "stockholders equity", "shareholders equity",
    "diluted eps", "basic eps", "operating margin", "net margin",
    "return on equity", "roe", "return on assets", "roa",
}
_METRIC_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_METRIC_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def extract_entities(text: str, use_spacy: bool | None = None) -> list[Entity]:
    """
    Extract named entities from filing text.

    use_spacy defaults to the SEC_USE_SPACY env var if not passed explicitly.
    """
    if use_spacy is None:
        use_spacy = os.getenv("SEC_USE_SPACY", "false").lower() == "true"

    entities: list[Entity] = []

    if use_spacy:
        entities.extend(_extract_spacy(text))
    else:
        entities.extend(_extract_regex(text))

    return entities


def _extract_regex(text: str) -> list[Entity]:
    entities: list[Entity] = []

    for m in _TICKER_RE.finditer(text):
        ticker = m.group(1) or m.group(2)
        entities.append(Entity(entity_type="TICKER", value=ticker,
                               start_char=m.start(), end_char=m.end()))

    for m in _CIK_RE.finditer(text):
        entities.append(Entity(entity_type="CIK", value=m.group(1),
                               start_char=m.start(), end_char=m.end()))

    for m in _MONEY_RE.finditer(text):
        scale = m.group(2) or ""
        entities.append(Entity(entity_type="MONEY",
                               value=f"${m.group(1)} {scale}".strip(),
                               start_char=m.start(), end_char=m.end()))

    for m in _FISCAL_PERIOD_RE.finditer(text):
        val = m.group(0).strip()
        entities.append(Entity(entity_type="FISCAL_PERIOD", value=val,
                               start_char=m.start(), end_char=m.end()))

    for m in _DATE_RE.finditer(text):
        entities.append(Entity(entity_type="DATE", value=m.group(0),
                               start_char=m.start(), end_char=m.end()))

    for m in _METRIC_RE.finditer(text):
        entities.append(Entity(entity_type="METRIC", value=m.group(1).lower(),
                               start_char=m.start(), end_char=m.end()))

    return entities


def _extract_spacy(text: str) -> list[Entity]:
    """spaCy NER path — only loaded if SEC_USE_SPACY=true."""
    try:
        import spacy  # noqa: PLC0415
        nlp = spacy.load("en_core_web_sm")
    except Exception as exc:
        logger.warning("spaCy unavailable, falling back to regex", extra={"error": str(exc)})
        return _extract_regex(text)

    doc = nlp(text[:100_000])  # spaCy has a practical token limit
    entities: list[Entity] = []
    for ent in doc.ents:
        entities.append(Entity(
            entity_type=ent.label_,
            value=ent.text,
            start_char=ent.start_char,
            end_char=ent.end_char,
        ))
    # Also run regex for SEC-specific patterns spaCy misses
    entities.extend(_extract_regex(text))
    return entities
