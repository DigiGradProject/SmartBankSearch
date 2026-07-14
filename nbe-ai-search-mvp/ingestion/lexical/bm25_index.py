"""BM25 lexical index for hybrid retrieval."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from ingestion.embedding.vector_store import RetrievedChunk
from shared.arabic_normalize import normalize_arabic
from shared.config import settings
from shared.logging import get_logger
from shared.schemas import ChunkRecord

logger = get_logger(__name__)

AR_TOKEN = re.compile(r"[\u0600-\u06FF]{2,}")
EN_TOKEN = re.compile(r"[a-zA-Z]{2,}")


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    document_id: str
    title: str
    url: str
    language: str
    text: str
    doc_type: str
    category: str
    is_stub: bool
    canonical_url_slug: str


def tokenize_text(text: str, language: str) -> list[str]:
    if language == "ar":
        return AR_TOKEN.findall(normalize_arabic(text))
    return [token.lower() for token in EN_TOKEN.findall(text.lower())]


def _to_indexed(chunk: ChunkRecord) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        title=chunk.title,
        url=chunk.url,
        language=chunk.language,
        text=chunk.text,
        doc_type=chunk.doc_type,
        category=chunk.category,
        is_stub=chunk.is_stub,
        canonical_url_slug=chunk.canonical_url_slug,
    )


class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[IndexedChunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunk_records: list[ChunkRecord]) -> None:
        self._chunks = [_to_indexed(chunk) for chunk in chunk_records]
        self._tokenized = [
            tokenize_text(f"{chunk.title} {chunk.text}", chunk.language) for chunk in self._chunks
        ]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        logger.info("bm25_index_built", chunks=len(self._chunks))

    def save(self, path: Path | None = None) -> None:
        target = path or settings.bm25_index_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "collection": settings.chroma_collection,
            "chunks": self._chunks,
            "tokenized": self._tokenized,
        }
        with target.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("bm25_index_saved", path=str(target), chunks=len(self._chunks))

    def load(self, path: Path | None = None) -> bool:
        target = path or settings.bm25_index_path
        if not target.exists():
            return False
        with target.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("collection") != settings.chroma_collection:
            logger.warning(
                "bm25_index_collection_mismatch",
                expected=settings.chroma_collection,
                found=payload.get("collection"),
            )
            return False
        self._chunks = payload["chunks"]
        self._tokenized = payload["tokenized"]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        logger.info("bm25_index_loaded", path=str(target), chunks=len(self._chunks))
        return True

    def query(
        self,
        query_text: str,
        language: str,
        top_k: int = 50,
        *,
        doc_types: list[str] | None = None,
    ) -> list[tuple[IndexedChunk, float]]:
        if not self._bm25 or not self._chunks:
            return []

        query_tokens = tokenize_text(query_text, language)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

        results: list[tuple[IndexedChunk, float]] = []
        for idx in ranked_indices:
            chunk = self._chunks[idx]
            if chunk.language != language:
                continue
            if doc_types and chunk.doc_type not in doc_types:
                continue
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append((chunk, score))
            if len(results) >= top_k:
                break
        return results

    def to_retrieved_chunk(self, chunk: IndexedChunk, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.title,
            url=chunk.url,
            language=chunk.language,
            text=chunk.text,
            score=score,
            doc_type=chunk.doc_type,
            category=chunk.category,
            is_stub=chunk.is_stub,
            canonical_url_slug=chunk.canonical_url_slug,
        )


_bm25_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        index = BM25Index()
        if not index.load():
            logger.warning("bm25_index_not_found", path=str(settings.bm25_index_path))
        _bm25_index = index
    return _bm25_index


def rebuild_bm25_index(chunk_records: list[ChunkRecord]) -> BM25Index:
    global _bm25_index
    index = BM25Index()
    index.build(chunk_records)
    index.save()
    _bm25_index = index
    return index
