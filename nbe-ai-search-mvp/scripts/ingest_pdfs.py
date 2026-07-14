"""Ingest PDF documents discovered in scrape."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.pdf_processing.pipeline import ingest_pdfs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and embed NBE PDFs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PDFs to process")
    args = parser.parse_args()

    downloaded, documents, upserted = ingest_pdfs(limit=args.limit)
    print(f"downloaded={downloaded} documents={documents} upserted_chunks={upserted}")


if __name__ == "__main__":
    main()
