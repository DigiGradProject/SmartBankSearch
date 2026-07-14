"""Run batch ingestion from documents.json or scrape source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.pipeline import run_ingestion  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NBE AI Search ingestion")
    parser.add_argument(
        "--source",
        choices=["documents_json", "scrape", "cleaned_jsonl"],
        default="cleaned_jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    status = run_ingestion(source=args.source, limit=args.limit)
    print(
        f"run_id={status.run_id} status={status.status} "
        f"documents={status.documents_processed} upserted={status.chunks_upserted} "
        f"skipped={status.chunks_skipped}"
    )
    if status.errors:
        print("errors:")
        for error in status.errors[:10]:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
