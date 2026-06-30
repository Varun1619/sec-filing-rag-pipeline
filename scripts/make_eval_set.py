"""
Generates a minimal labeled evaluation set from the indexed chunks.

Run after `build` to auto-generate questions from the first few chunks of each
filing.  You can then manually curate eval_data/eval_set.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb  # noqa: E402

from src.config import settings
from src.logging_utils import setup_logging

setup_logging(settings.log_level)


def make_eval_set(output_path: str = "eval_data/eval_set.json", n: int = 20) -> None:
    db = str(settings.duckdb_path)
    try:
        conn = duckdb.connect(db)
        rows = conn.execute(
            f"""
            SELECT chunk_id, filing_id, cik, company_name, form_type, filed_date, text
            FROM raw_chunks
            ORDER BY random()
            LIMIT {n}
            """
        ).fetchall()
        conn.close()
    except Exception as e:
        print(f"Could not read warehouse: {e}")
        rows = []

    samples = []
    for chunk_id, filing_id, cik, company, form, date, text in rows:
        # Create a simple factual question from the chunk text
        snippet = text[:200].replace("\n", " ").strip()
        question = (
            f"What information does {company} provide in its {form} filing about: {snippet[:80]}?"
        )
        samples.append(
            {
                "question": question,
                "expected_filing_id": filing_id,
                "expected_cik": cik,
                "expected_text_contains": snippet[:40],
            }
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} eval samples to {output_path}")


if __name__ == "__main__":
    make_eval_set()
