"""Export scraped NBE pages into the unified documents.json contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.document_processing.processor import load_documents_from_scrape  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export scrape data to documents.json")
    parser.add_argument("--scrape-root", type=Path, default=ROOT.parent / "nbe_complete_scrape")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "documents.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    documents = load_documents_from_scrape(args.scrape_root, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = [doc.model_dump() for doc in documents]
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Exported {len(payload)} documents to {args.output}")


if __name__ == "__main__":
    main()
