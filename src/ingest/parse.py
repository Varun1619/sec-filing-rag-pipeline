"""
Filing parser — HTML (BeautifulSoup) and PDF (pypdf).

Design decisions:
- HTML is the primary format for modern EDGAR filings; we strip boilerplate
  navigation/header/footer elements before extracting text.
- PDF extraction is text-layer only; scanned PDFs are flagged and skipped
  (OCR is documented as a future enhancement).
- The output is raw text per filing — chunking is a separate concern.
"""

from __future__ import annotations

from pathlib import Path

from src.logging_utils import get_logger
from src.models import Filing

logger = get_logger(__name__)

# Tags that never contain meaningful SEC filing text
_STRIP_TAGS = {"script", "style", "header", "footer", "nav", "aside"}


def parse_filing(filing: Filing) -> str:
    """
    Extract raw text from a downloaded filing.

    Returns the full document text as a single string.
    Raises ValueError if the filing has no local_path or is a scanned PDF.
    """
    if not filing.local_path:
        raise ValueError(f"Filing {filing.filing_id} has no local_path; download it first.")

    path = Path(filing.local_path)
    if not path.exists():
        raise FileNotFoundError(f"Local path not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".htm", ".html", ".xml"}:
        return _parse_html(path)
    elif suffix == ".pdf":
        if filing.is_scanned_pdf:
            logger.warning(
                "Scanned PDF flagged for OCR (skipped)",
                extra={"path": str(path), "filing_id": filing.filing_id},
            )
            return ""
        return _parse_pdf(path)
    else:
        # Attempt HTML parsing as a fallback for unknown extensions
        logger.warning("Unknown extension; attempting HTML parse", extra={"path": str(path)})
        return _parse_html(path)


def _parse_html(path: Path) -> str:
    """Parse an HTML/HTM filing and return cleaned plain text."""
    from bs4 import BeautifulSoup

    raw = path.read_bytes()
    soup = BeautifulSoup(raw, "lxml")

    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    # SEC EDGAR wraps the actual document in <DOCUMENT> tags in SGML-style submissions;
    # BeautifulSoup treats these as custom tags — just grab all text.
    text = soup.get_text(separator="\n")

    # Collapse excessive blank lines
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)

    logger.debug("Parsed HTML", extra={"path": str(path), "chars": len(cleaned)})
    return cleaned


def _parse_pdf(path: Path) -> str:
    """Extract text from a text-layer PDF."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as exc:
            logger.warning("PDF page extraction error", extra={"page": i, "error": str(exc)})

    text = "\n".join(pages_text)
    logger.debug(
        "Parsed PDF",
        extra={"path": str(path), "pages": len(reader.pages), "chars": len(text)},
    )
    return text
