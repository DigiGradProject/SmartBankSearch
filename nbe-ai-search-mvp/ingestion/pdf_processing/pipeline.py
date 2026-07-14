"""Download PDFs referenced in scrape and ingest extracted text."""

from __future__ import annotations

from pathlib import Path

from ingestion.chunking.chunker import chunk_document
from ingestion.embedding.vector_store import VectorStore
from ingestion.pdf_processing.catalog import collect_pdf_links, pdf_catalog_to_documents
from ingestion.pdf_processing.extractor import download_pdf, extract_pdf_text
from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)


def ingest_pdfs(limit: int | None = None) -> tuple[int, int, int]:
    catalog = collect_pdf_links(settings.scrape_root)
    pdf_dir = settings.project_root / "data" / "pdfs"
    texts: dict[str, str] = {}
    downloaded = 0
    extracted = 0

    urls = sorted(catalog.keys())
    if limit:
        urls = urls[:limit]

    for url in urls:
        filename = url.rsplit("/", 1)[-1].split("?")[0] or "document.pdf"
        local_path = pdf_dir / filename
        if download_pdf(url, local_path):
            downloaded += 1
            text = extract_pdf_text(local_path)
            if text:
                texts[url] = text
                extracted += 1

    documents = pdf_catalog_to_documents(catalog, texts)
    store = VectorStore()
    upserted_total = 0
    for document in documents:
        chunks = chunk_document(document)
        upserted, _ = store.upsert_chunks(chunks)
        upserted_total += upserted

    logger.info(
        "pdf_ingestion_completed",
        discovered=len(catalog),
        downloaded=downloaded,
        extracted=extracted,
        documents=len(documents),
        upserted=upserted_total,
    )
    return downloaded, len(documents), upserted_total
