"""Hybrid retrieval: BGE-M3 dense + BM25 lexical fused with RRF."""

from __future__ import annotations

from ingestion.embedding.vector_store import RetrievedChunk, VectorStore
from ingestion.lexical.bm25_index import BM25Index, get_bm25_index
from services.search_service.intent_classifier import QueryIntent
from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    k: int | None = None,
) -> dict[str, float]:
    """Fuse ranked chunk-id lists using Reciprocal Rank Fusion."""
    rrf_k = k or settings.rrf_k
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return scores


def _chunk_map(chunks: list[RetrievedChunk]) -> dict[str, RetrievedChunk]:
    return {chunk.chunk_id: chunk for chunk in chunks}


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index

    def _get_bm25(self) -> BM25Index | None:
        if not settings.bm25_enabled:
            return None
        if self.bm25_index is not None:
            return self.bm25_index if self.bm25_index.size > 0 else None
        index = get_bm25_index()
        return index if index.size > 0 else None

    def retrieve(
        self,
        query_text: str,
        language: str,
        *,
        candidate_k: int,
        intent: QueryIntent,
        apply_filter: bool,
        doc_types: list[str] | None,
    ) -> tuple[list[RetrievedChunk], bool]:
        dense_k = settings.dense_top_k
        bm25_k = settings.bm25_top_k
        filter_language = language if apply_filter else None
        filter_doc_types = doc_types if apply_filter else None

        dense_chunks = self.vector_store.query(
            query_text,
            dense_k,
            language=filter_language,
            doc_types=filter_doc_types,
        )
        filter_applied = apply_filter and bool(dense_chunks)

        bm25 = self._get_bm25()
        bm25_chunks: list[RetrievedChunk] = []
        if bm25:
            bm25_hits = bm25.query(
                query_text,
                language,
                bm25_k,
                doc_types=filter_doc_types,
            )
            bm25_chunks = [bm25.to_retrieved_chunk(chunk, score) for chunk, score in bm25_hits]

        if apply_filter and len(dense_chunks) < 3 and len(bm25_chunks) < 3:
            logger.info(
                "intent_filter_fallback_broad",
                intent=intent.intent,
                dense_count=len(dense_chunks),
                bm25_count=len(bm25_chunks),
            )
            dense_chunks = self.vector_store.query(query_text, dense_k, language=language)
            if bm25:
                bm25_hits = bm25.query(query_text, language, bm25_k)
                bm25_chunks = [bm25.to_retrieved_chunk(chunk, score) for chunk, score in bm25_hits]
            filter_applied = False

        if not bm25 or not bm25_chunks:
            return dense_chunks[:candidate_k], filter_applied

        dense_ranking = [chunk.chunk_id for chunk in dense_chunks]
        bm25_ranking = [chunk.chunk_id for chunk in bm25_chunks]
        fused_scores = reciprocal_rank_fusion([dense_ranking, bm25_ranking])

        merged = _chunk_map(dense_chunks)
        for chunk in bm25_chunks:
            merged.setdefault(chunk.chunk_id, chunk)

        fused_chunks: list[RetrievedChunk] = []
        for chunk_id, rrf_score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True):
            source = merged[chunk_id]
            fused_chunks.append(
                RetrievedChunk(
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    title=source.title,
                    url=source.url,
                    language=source.language,
                    text=source.text,
                    score=min(1.0, rrf_score * 30.0),
                    lexical_weights=source.lexical_weights,
                    doc_type=source.doc_type,
                    category=source.category,
                    is_stub=source.is_stub,
                    canonical_url_slug=source.canonical_url_slug,
                )
            )

        logger.info(
            "hybrid_retrieval",
            dense=len(dense_chunks),
            bm25=len(bm25_chunks),
            fused=len(fused_chunks),
            filter_applied=filter_applied,
        )
        return fused_chunks[:candidate_k], filter_applied
