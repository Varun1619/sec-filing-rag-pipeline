"""
EDGAR ingestion — idempotent, incremental, rate-limited.

Design decisions:
- Idempotency: we skip any accession number already present in the bronze dir.
- Incrementality: a watermark file stores the last-seen filing date so subsequent
  runs only fetch newer filings.
- Rate limiting: tenacity retry + a per-request sleep respects EDGAR's 10 req/s cap.
- User-Agent: EDGAR policy requires a descriptive string with contact info.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Iterator

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.logging_utils import get_logger
from src.models import Filing

logger = get_logger(__name__)

_WATERMARK_FILE = Path(settings.data_dir) / ".watermark"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": settings.user_agent, "Accept-Encoding": "gzip"})
    return s


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def _get(session: requests.Session, url: str) -> requests.Response:
    resp = session.get(url, timeout=30)
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", 10))
        logger.warning("Rate limited by EDGAR", extra={"retry_after": retry_after, "url": url})
        time.sleep(retry_after)
        resp.raise_for_status()
    resp.raise_for_status()
    time.sleep(1.0 / settings.rate_limit_rps)
    return resp


def _load_watermark() -> date | None:
    if settings.watermark_date:
        return date.fromisoformat(settings.watermark_date)
    if _WATERMARK_FILE.exists():
        raw = _WATERMARK_FILE.read_text().strip()
        return date.fromisoformat(raw) if raw else None
    return None


def _save_watermark(d: date) -> None:
    _WATERMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _WATERMARK_FILE.write_text(d.isoformat())


def get_company_filings(
    cik: str,
    form_types: list[str] | None = None,
    max_filings: int = 10,
) -> Iterator[Filing]:
    """
    Yield Filing objects for a given CIK from EDGAR's submissions API.

    Only filings newer than the current watermark are yielded (incremental).
    Already-downloaded accession numbers are skipped (idempotent).
    """
    form_types = form_types or ["10-K", "10-Q"]
    session = _session()
    watermark = _load_watermark()

    padded_cik = cik.zfill(10)
    url = f"{settings.edgar_base_url}/submissions/CIK{padded_cik}.json"
    logger.info("Fetching submissions", extra={"cik": padded_cik, "url": url})

    data = _get(session, url).json()
    company_name: str = data.get("name", "Unknown")
    filings_data: dict = data.get("filings", {}).get("recent", {})

    accessions: list[str] = filings_data.get("accessionNumber", [])
    forms: list[str] = filings_data.get("form", [])
    filed_dates: list[str] = filings_data.get("filingDate", [])
    periods: list[str] = filings_data.get("reportDate", [])
    primary_docs: list[str] = filings_data.get("primaryDocument", [])

    bronze_dir = settings.bronze_dir()
    bronze_dir.mkdir(parents=True, exist_ok=True)

    yielded = 0
    for acc, form, filed_str, period_str, doc in zip(
        accessions, forms, filed_dates, periods, primary_docs
    ):
        if yielded >= max_filings:
            break
        if form not in form_types:
            continue

        filed = date.fromisoformat(filed_str)
        if watermark and filed <= watermark:
            logger.debug(
                "Skipping (before watermark)",
                extra={"accession": acc, "filed": filed_str},
            )
            continue

        norm_acc = acc.replace("-", "")
        local_dir = bronze_dir / padded_cik / norm_acc
        if local_dir.exists() and any(local_dir.iterdir()):
            logger.debug("Skipping (already downloaded)", extra={"accession": acc})
            continue

        doc_url = (
            f"{settings.edgar_base_url}/Archives/edgar/full-index/"
            if not doc
            else f"https://www.sec.gov/Archives/edgar/data/{int(padded_cik)}" f"/{norm_acc}/{doc}"
        )

        filing = Filing(
            cik=padded_cik,
            company_name=company_name,
            form_type=form,
            filed_date=filed,
            period_of_report=date.fromisoformat(period_str) if period_str else None,
            accession_number=acc,
            document_url=doc_url,
        )
        yield filing
        yielded += 1


def download_filing(filing: Filing) -> Filing:
    """
    Download the raw filing document to the bronze layer.

    Returns the Filing with local_path populated. Skips if already present.
    """
    session = _session()
    norm_acc = filing.accession_number.replace("-", "")
    local_dir = settings.bronze_dir() / filing.cik / norm_acc
    local_dir.mkdir(parents=True, exist_ok=True)

    # Derive filename from URL
    url = filing.document_url
    filename = url.split("/")[-1] or "filing.htm"
    local_path = local_dir / filename

    if local_path.exists():
        logger.info("Already downloaded", extra={"path": str(local_path)})
        filing = filing.model_copy(
            update={
                "local_path": str(local_path),
                "file_size_bytes": local_path.stat().st_size,
            }
        )
        return filing

    logger.info("Downloading filing", extra={"url": url, "dest": str(local_path)})
    resp = _get(session, url)
    local_path.write_bytes(resp.content)

    # Detect scanned PDF: try to extract text; if none → likely scanned
    is_scanned = False
    if filename.endswith(".pdf"):
        is_scanned = _is_scanned_pdf(local_path)

    # Persist metadata alongside the raw file
    meta = filing.model_dump(mode="json")
    meta["local_path"] = str(local_path)
    meta["file_size_bytes"] = local_path.stat().st_size
    meta["is_scanned_pdf"] = is_scanned
    (local_dir / "meta.json").write_text(json.dumps(meta, default=str), encoding="utf-8")

    return filing.model_copy(
        update={
            "local_path": str(local_path),
            "file_size_bytes": local_path.stat().st_size,
            "is_scanned_pdf": is_scanned,
        }
    )


def _is_scanned_pdf(path: Path) -> bool:
    """Heuristic: a PDF with fewer than 100 chars of extractable text is likely scanned."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "".join(p.extract_text() or "" for p in reader.pages[:3])
        return len(text.strip()) < 100
    except Exception:
        return False


def ingest_companies(
    cik_list: list[str],
    form_types: list[str] | None = None,
    max_per_company: int = 3,
) -> list[Filing]:
    """
    Top-level ingestion entry point.

    Fetches + downloads filings for each CIK and updates the watermark.
    Returns the list of downloaded Filing objects.
    """
    all_filings: list[Filing] = []
    latest_date: date | None = None

    for cik in cik_list:
        for filing in get_company_filings(cik, form_types=form_types, max_filings=max_per_company):
            downloaded = download_filing(filing)
            all_filings.append(downloaded)
            if latest_date is None or downloaded.filed_date > latest_date:
                latest_date = downloaded.filed_date
            logger.info(
                "Ingested filing",
                extra={
                    "cik": filing.cik,
                    "company": filing.company_name,
                    "form": filing.form_type,
                    "filed": filing.filed_date.isoformat(),
                    "accession": filing.accession_number,
                },
            )

    if latest_date:
        _save_watermark(latest_date)
        logger.info("Watermark updated", extra={"watermark": latest_date.isoformat()})

    logger.info(
        "Ingestion complete",
        extra={"total_filings": len(all_filings), "companies": len(cik_list)},
    )
    return all_filings
