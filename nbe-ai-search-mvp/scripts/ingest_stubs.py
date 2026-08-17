"""Ingest active curated documents only (fast path for reviewed updates)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.chunking.chunker import chunk_document  # noqa: E402
from ingestion.document_processing.processor import load_curated_documents  # noqa: E402
from ingestion.embedding.vector_store import VectorStore  # noqa: E402


def main() -> None:
    docs = load_curated_documents(ROOT)
    store = VectorStore()
    upserted = 0
    for document in docs:
        chunks = chunk_document(document)
        count, _ = store.upsert_chunks(chunks)
        upserted += count
    print(f"curated_documents={len(docs)} upserted_chunks={upserted} total_chunks={store.count()}")


if __name__ == "__main__":
    main()
