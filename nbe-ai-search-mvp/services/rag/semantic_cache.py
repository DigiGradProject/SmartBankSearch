"""Semantic query cache backed by a dedicated Chroma collection (BGE-M3)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from ingestion.embedding.bge_m3 import get_embedder
from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SemanticCacheHit:
    query: str
    similarity: float
    payload: dict[str, Any]
    cache_id: str


class SemanticCache:
    """Similarity cache: hit when cosine similarity ≥ threshold."""

    def __init__(self, collection_name: str | None = None) -> None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        name = collection_name or settings.semantic_cache_collection
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", "purpose": "semantic_query_cache"},
        )
        self._embedder = get_embedder()

    def lookup(self, query: str, language: str) -> SemanticCacheHit | None:
        if not settings.semantic_cache_enabled:
            return None
        text = f"{language}:{query.strip()}"
        embedding = self._embedder.embed_dense([text])[0]
        try:
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["metadatas", "distances", "documents"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("semantic_cache_lookup_failed", error=str(exc))
            return None

        ids = (result.get("ids") or [[]])[0]
        if not ids:
            return None
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distance = float(distances[0]) if distances else 1.0
        # Chroma cosine distance: similarity ≈ 1 - distance
        similarity = 1.0 - distance
        meta = metadatas[0] or {}
        expires_at = float(meta.get("expires_at", 0) or 0)
        if expires_at and time.time() > expires_at:
            try:
                self._collection.delete(ids=[ids[0]])
            except Exception:  # noqa: BLE001
                pass
            return None
        if similarity < settings.semantic_cache_threshold:
            return None
        try:
            payload = json.loads(meta.get("payload_json", "{}"))
        except json.JSONDecodeError:
            return None
        if not payload:
            return None
        logger.info("semantic_cache_hit", similarity=round(similarity, 4), cache_id=ids[0])
        return SemanticCacheHit(
            query=str(meta.get("query", query)),
            similarity=similarity,
            payload=payload,
            cache_id=ids[0],
        )

    def store(
        self,
        query: str,
        language: str,
        payload: dict[str, Any],
        *,
        retrieved_urls: list[str] | None = None,
    ) -> str | None:
        if not settings.semantic_cache_enabled:
            return None
        # Never cache unanswered / empty answers
        if not payload.get("answered") or not payload.get("answer"):
            return None
        if float(payload.get("confidence") or 0) < settings.confidence_threshold:
            return None

        cache_id = f"sc_{uuid.uuid4().hex[:24]}"
        text = f"{language}:{query.strip()}"
        embedding = self._embedder.embed_dense([text])[0]
        now = time.time()
        metadata = {
            "query": query[:500],
            "language": language,
            "created_at": now,
            "expires_at": now + settings.semantic_cache_ttl_seconds,
            "urls": json.dumps((retrieved_urls or [])[:10], ensure_ascii=False)[:1500],
            "payload_json": json.dumps(payload, ensure_ascii=False)[:48000],
        }
        try:
            self._collection.upsert(
                ids=[cache_id],
                embeddings=[embedding],
                documents=[text[:2000]],
                metadatas=[metadata],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("semantic_cache_store_failed", error=str(exc))
            return None
        logger.info("semantic_cache_store", cache_id=cache_id)
        return cache_id
