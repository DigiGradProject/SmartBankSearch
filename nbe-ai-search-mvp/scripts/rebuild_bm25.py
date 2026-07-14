"""Rebuild BM25 index from corpus without re-embedding vectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.chunking.chunker import chunk_document  # noqa: E402
from ingestion.document_processing.processor import (  # noqa: E402
    load_documents_jsonl,
    load_product_stubs,
)
from ingestion.lexical.bm25_index import rebuild_bm25_index  # noqa: E402
from shared.config import settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild BM25 lexical index")
    parser.add_argument("--source", choices=["cleaned_jsonl"], default="cleaned_jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.source == "cleaned_jsonl":
        documents = load_documents_jsonl(settings.cleaned_jsonl_path, limit=args.limit)
        documents.extend(load_product_stubs(settings.project_root))

    all_chunks = []
    for document in documents:
        all_chunks.extend(chunk_document(document))

    index = rebuild_bm25_index(all_chunks)
    print(f"bm25_chunks={index.size} path={settings.bm25_index_path}")


if __name__ == "__main__":
    main()
