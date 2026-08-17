from __future__ import annotations

import json
from dataclasses import dataclass, replace

import chromadb
from chromadb.config import Settings as ChromaSettings

from ingestion.embedding.bge_m3 import get_embedder, sparse_score, sparse_score_against_text
from shared.config import settings
from shared.logging import get_logger
from shared.schemas import ChunkRecord

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    title: str
    url: str
    language: str
    text: str
    score: float
    lexical_weights: dict[str, float] | None = None
    doc_type: str = "general"
    category: str = "general"
    is_stub: bool = False
    canonical_url_slug: str = ""
    chunk_level: str = "child"
    parent_chunk_id: str = ""
    section_heading: str = ""
    page_type: str = "web_page"
    subcategory: str = ""
    product_name: str = ""
    keywords: str = ""


class VectorStore:
    def __init__(self) -> None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine", "embedding_model": settings.embedding_model},
        )
        self._embedder = get_embedder()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_dense(texts)

    def get_existing_hashes(self, chunk_ids: list[str]) -> dict[str, str]:
        if not chunk_ids:
            return {}
        result = self._collection.get(ids=chunk_ids, include=["metadatas"])
        hashes: dict[str, str] = {}
        for chunk_id, metadata in zip(result.get("ids") or [], result.get("metadatas") or []):
            if metadata:
                hashes[chunk_id] = metadata.get("content_hash", "")
        return hashes

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> tuple[int, int]:
        if not chunks:
            return 0, 0

        existing = self.get_existing_hashes([chunk.chunk_id for chunk in chunks])
        to_upsert = [chunk for chunk in chunks if existing.get(chunk.chunk_id) != chunk.content_hash]
        skipped = len(chunks) - len(to_upsert)
        if not to_upsert:
            return 0, skipped

        texts = [chunk.text for chunk in to_upsert]
        dense_vectors, sparse_weights = self._embedder.embed_hybrid(texts)
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in to_upsert],
            documents=[chunk.text for chunk in to_upsert],
            embeddings=dense_vectors,
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "title": chunk.title,
                    "url": chunk.url,
                    "language": chunk.language,
                    "content_hash": chunk.content_hash,
                    "doc_type": chunk.doc_type,
                    "category": chunk.category,
                    "is_stub": chunk.is_stub,
                    "canonical_url_slug": chunk.canonical_url_slug,
                    "quality_score": chunk.quality_score,
                    "chunk_level": chunk.chunk_level,
                    "parent_chunk_id": chunk.parent_chunk_id,
                    "section_heading": chunk.section_heading[:200],
                    "page_type": chunk.page_type,
                    "subcategory": chunk.subcategory,
                    "product_name": chunk.product_name[:180],
                    "service_name": chunk.service_name[:120],
                    "document_type": chunk.document_type,
                    "intent": chunk.intent,
                    "keywords": chunk.keywords[:300],
                    "last_updated": chunk.last_updated[:64],
                    "lexical_weights": json.dumps(_top_lexical(weights), ensure_ascii=False),
                }
                for chunk, weights in zip(to_upsert, sparse_weights)
            ],
        )
        return len(to_upsert), skipped

    def delete_chunks_not_in(self, chunk_ids: set[str]) -> int:
        """Remove stale chunks after a successful, unbounded corpus rebuild."""
        result = self._collection.get(include=[])
        existing_ids = set(result.get("ids") or [])
        stale_ids = sorted(existing_ids - chunk_ids)
        if stale_ids:
            self._collection.delete(ids=stale_ids)
        return len(stale_ids)

    def query(
        self,
        query_text: str,
        top_k: int,
        *,
        language: str | None = None,
        doc_types: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if self._collection.count() == 0:
            return []

        dense, sparse = self._embedder.embed_hybrid([query_text])
        query_embedding = dense[0]
        query_sparse = sparse[0]
        where = _build_where_filter(language=language, doc_types=doc_types)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        chunks: list[RetrievedChunk] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        for chunk_id, text, metadata, distance in zip(ids, docs, metas, distances):
            dense_score = max(0.0, 1.0 - float(distance))
            lexical = _parse_lexical(metadata.get("lexical_weights"))
            if lexical and query_sparse:
                sparse = sparse_score(query_sparse, lexical)
                # Normalize sparse roughly into 0..1 relative range later in hybrid layer.
                hybrid = settings.hybrid_alpha * dense_score + (1.0 - settings.hybrid_alpha) * min(1.0, sparse)
            elif query_sparse:
                sparse = sparse_score_against_text(query_sparse, text or "")
                hybrid = settings.hybrid_alpha * dense_score + (1.0 - settings.hybrid_alpha) * min(1.0, sparse / 5.0)
            else:
                hybrid = dense_score

            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    title=metadata.get("title", ""),
                    url=metadata.get("url", ""),
                    language=metadata.get("language", "en"),
                    text=text or "",
                    score=hybrid,
                    lexical_weights=lexical,
                    doc_type=metadata.get("doc_type", "general"),
                    category=metadata.get("category", "general"),
                    is_stub=bool(metadata.get("is_stub", False)),
                    canonical_url_slug=metadata.get("canonical_url_slug", ""),
                    chunk_level=metadata.get("chunk_level", "child"),
                    parent_chunk_id=metadata.get("parent_chunk_id", ""),
                    section_heading=metadata.get("section_heading", ""),
                    page_type=metadata.get("page_type", "web_page"),
                    subcategory=metadata.get("subcategory", ""),
                    product_name=metadata.get("product_name", ""),
                    keywords=metadata.get("keywords", ""),
                )
            )

        chunks.sort(key=lambda item: item.score, reverse=True)
        return chunks

    def count(self) -> int:
        return self._collection.count()

    def health(self) -> str:
        try:
            self._collection.count()
            return "ok"
        except Exception:
            return "down"


def _build_where_filter(
    *,
    language: str | None = None,
    doc_types: list[str] | None = None,
) -> dict | None:
    clauses: list[dict] = []
    if language:
        clauses.append({"language": {"$eq": language}})
    if doc_types:
        if len(doc_types) == 1:
            clauses.append({"doc_type": {"$eq": doc_types[0]}})
        else:
            clauses.append({"doc_type": {"$in": doc_types}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _top_lexical(weights: dict[str, float], limit: int = 64) -> dict[str, float]:
    if not weights:
        return {}
    ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:limit]
    return {token: round(float(score), 5) for token, score in ranked}


def _parse_lexical(raw: object) -> dict[str, float]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    try:
        parsed = json.loads(str(raw))
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items()}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return {}
