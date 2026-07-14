"""Hybrid retrieval: BGE-M3 dense + BM25 lexical fused with RRF."""

from __future__ import annotations

from dataclasses import replace

from ingestion.embedding.vector_store import RetrievedChunk, VectorStore
from ingestion.lexical.bm25_index import BM25Index, get_bm25_index
from services.rag.metadata_filter import related_doc_types, should_broaden
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

    def _query_channels(
        self,
        query_text: str,
        language: str,
        *,
        dense_k: int,
        bm25_k: int,
        filter_language: str | None,
        filter_doc_types: list[str] | None,
        bm25: BM25Index | None,
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        dense_chunks = self.vector_store.query(
            query_text,
            dense_k,
            language=filter_language,
            doc_types=filter_doc_types,
        )
        bm25_chunks: list[RetrievedChunk] = []
        if bm25:
            bm25_hits = bm25.query(
                query_text,
                language,
                bm25_k,
                doc_types=filter_doc_types,
            )
            bm25_chunks = [bm25.to_retrieved_chunk(chunk, score) for chunk, score in bm25_hits]
        return dense_chunks, bm25_chunks

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
        bm25 = self._get_bm25()

        dense_chunks, bm25_chunks = self._query_channels(
            query_text,
            language,
            dense_k=dense_k,
            bm25_k=bm25_k,
            filter_language=filter_language,
            filter_doc_types=filter_doc_types,
            bm25=bm25,
        )
        filter_applied = apply_filter and bool(dense_chunks or bm25_chunks)
        filter_stage = "L0" if apply_filter else "none"

        if apply_filter and should_broaden(
            intent,
            dense_count=len(dense_chunks),
            bm25_count=len(bm25_chunks),
        ):
            # L1/L2: expand to related parent types before full broad.
            related = related_doc_types(intent)
            exact = list(doc_types or [])
            if related and related != exact:
                logger.info(
                    "intent_filter_related_expand",
                    intent=intent.intent,
                    from_types=exact,
                    to_types=related,
                    dense_count=len(dense_chunks),
                    bm25_count=len(bm25_chunks),
                )
                related_dense, related_bm25 = self._query_channels(
                    query_text,
                    language,
                    dense_k=dense_k,
                    bm25_k=bm25_k,
                    filter_language=language,
                    filter_doc_types=related,
                    bm25=bm25,
                )
                if related_dense or related_bm25:
                    dense_chunks, bm25_chunks = related_dense, related_bm25
                    filter_applied = True
                    filter_stage = "L1_related"

            if should_broaden(
                intent,
                dense_count=len(dense_chunks),
                bm25_count=len(bm25_chunks),
            ):
                logger.info(
                    "intent_filter_fallback_broad",
                    intent=intent.intent,
                    dense_count=len(dense_chunks),
                    bm25_count=len(bm25_chunks),
                    stage="L4",
                )
                dense_chunks = self.vector_store.query(query_text, dense_k, language=language)
                if bm25:
                    bm25_hits = bm25.query(query_text, language, bm25_k)
                    bm25_chunks = [bm25.to_retrieved_chunk(chunk, score) for chunk, score in bm25_hits]
                filter_applied = False
                filter_stage = "L4_broad"
        elif apply_filter:
            logger.info(
                "intent_filter_kept",
                intent=intent.intent,
                dense_count=len(dense_chunks),
                bm25_count=len(bm25_chunks),
                stage=filter_stage,
            )

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
            fused_chunks.append(replace(source, score=min(1.0, rrf_score * 30.0)))

        logger.info(
            "hybrid_retrieval",
            dense=len(dense_chunks),
            bm25=len(bm25_chunks),
            fused=len(fused_chunks),
            filter_applied=filter_applied,
            filter_stage=filter_stage,
        )
        return fused_chunks[:candidate_k], filter_applied
