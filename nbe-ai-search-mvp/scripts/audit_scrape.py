"""Audit scrape coverage vs indexed content."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.document_processing.processor import load_documents_from_scrape  # noqa: E402
from ingestion.embedding.vector_store import VectorStore  # noqa: E402
from ingestion.pdf_processing.catalog import collect_pdf_links  # noqa: E402
from shared.config import settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit scrape and index coverage")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "coverage_report.json")
    args = parser.parse_args()

    scrape_root = settings.scrape_root
    html_pages = list(scrape_root.glob("**/pages/**/data.json"))
    valid_docs = load_documents_from_scrape(scrape_root)
    pdf_links = collect_pdf_links(scrape_root)
    store = VectorStore()

    report = {
        "scrape_html_pages": len(html_pages),
        "valid_documents_after_cleaning": len(valid_docs),
        "pdf_links_discovered": len(pdf_links),
        "indexed_chunks": store.count(),
        "sample_pdf_links": [
            {"title": meta["title"], "url": url}
            for url, meta in list(pdf_links.items())[:20]
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
