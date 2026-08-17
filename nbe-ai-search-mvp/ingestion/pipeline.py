import uuid
from datetime import UTC, datetime

from ingestion.chunking.chunker import chunk_document, indexable_chunks
from ingestion.document_processing.processor import (
    load_curated_documents,
    load_documents_from_scrape,
    load_documents_json,
    load_documents_jsonl,
    load_merged_corpus,
)
from ingestion.embedding.vector_store import VectorStore
from ingestion.lexical.bm25_index import rebuild_bm25_index
from shared.config import settings
from shared.logging import get_logger
from shared.schemas import IngestRunStatus

logger = get_logger(__name__)

RUNS: dict[str, IngestRunStatus] = {}


def run_ingestion(source: str = "documents_json", limit: int | None = None) -> IngestRunStatus:
    run_id = str(uuid.uuid4())
    status = IngestRunStatus(
        run_id=run_id,
        status="running",
        started_at=datetime.now(UTC),
    )
    RUNS[run_id] = status

    try:
        if source == "scrape":
            documents = load_documents_from_scrape(settings.scrape_root, limit=limit)
            documents.extend(load_curated_documents(settings.project_root))
        elif source == "cleaned_jsonl":
            if not settings.cleaned_jsonl_path.exists():
                raise FileNotFoundError(f"Cleaned JSONL not found: {settings.cleaned_jsonl_path}")
            documents = load_documents_jsonl(settings.cleaned_jsonl_path, limit=limit)
            documents.extend(load_curated_documents(settings.project_root))
        elif source == "merged":
            documents = load_merged_corpus(settings.project_root, limit=limit)
        else:
            if not settings.documents_path.exists():
                documents = load_documents_from_scrape(settings.scrape_root, limit=limit)
            else:
                documents = load_documents_json(settings.documents_path)
                if limit:
                    documents = documents[:limit]
            documents.extend(load_curated_documents(settings.project_root))

        store = VectorStore()
        upserted_total = 0
        skipped_total = 0
        errors: list[str] = []
        all_chunks = []

        for document in documents:
            try:
                chunks = indexable_chunks(chunk_document(document))
                all_chunks.extend(chunks)
                upserted, skipped = store.upsert_chunks(chunks)
                upserted_total += upserted
                skipped_total += skipped
                status.documents_processed += 1
            except Exception as exc:  # noqa: BLE001
                message = f"{document.id}: {exc}"
                errors.append(message)
                logger.error("ingest_document_failed", document_id=document.id, error=str(exc))

        if settings.bm25_enabled and all_chunks:
            rebuild_bm25_index(all_chunks)

        pruned_total = 0
        if not errors and limit is None:
            pruned_total = store.delete_chunks_not_in(
                {chunk.chunk_id for chunk in all_chunks}
            )

        status.chunks_upserted = upserted_total
        status.chunks_skipped = skipped_total
        status.errors = errors
        status.status = "completed" if not errors else "completed"
        status.finished_at = datetime.now(UTC)
        logger.info(
            "ingestion_completed",
            run_id=run_id,
            documents=status.documents_processed,
            upserted=upserted_total,
            skipped=skipped_total,
            pruned=pruned_total,
            bm25_chunks=len(all_chunks),
        )
        return status
    except Exception as exc:  # noqa: BLE001
        status.status = "failed"
        status.errors = [str(exc)]
        status.finished_at = datetime.now(UTC)
        logger.error("ingestion_failed", run_id=run_id, error=str(exc))
        return status
