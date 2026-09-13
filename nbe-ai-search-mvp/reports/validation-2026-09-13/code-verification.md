# Requested implementation evidence

Snapshot of commit `607cb249b6bb716ec210211b25651135bf2d24c1`, inspected 2026-09-13. Runtime code was not changed for these experiments.

## services/search_service/search.py

```python
from dataclasses import dataclass

from services.search_service.hybrid_retriever import HybridRetriever
from services.rag.business_rules import apply_business_rule_scoring
from services.search_service.product_detail import prioritize_product_chunks
from services.search_service.card_catalog import prioritize_credit_card_chunks
from services.search_service.account_rank import prioritize_account_chunks
from services.rag.hybrid_rank import apply_metadata_ranking
from services.rag.confidence import compute_confidence
from services.rag.decision_engine import decide
from services.rag.query_understanding import QueryUnderstanding, understand_query
from services.search_service.intent_classifier import QueryIntent, should_apply_metadata_filter
from services.search_service.keyword_rank import extract_query_terms, keyword_overlap_score, rerank_chunks
from services.search_service.reranker import get_reranker
from shared.config import settings
from shared.retrieval_mode import RetrievalMode, business_rules_active, parse_retrieval_mode
from shared.url_canonical import canonical_url_key
from shared.document_quality import (
    is_junk_document,
    is_low_value_document,
    is_menu_heavy_text,
    is_official_product_category_url,
)
from shared.logging import get_logger
from ingestion.embedding.vector_store import RetrievedChunk

logger = get_logger(__name__)

ACCOUNT_INTENTS = frozenset({"account_open"})
CARD_INTENTS = frozenset({"credit_card", "card_types", "debit_card"})


@dataclass
class RetrievalResult:
    query: str
    language: str
    chunks: list[RetrievedChunk]
    confidence: float
    should_answer: bool
    abstention_reason: str | None
    intent: str = "general_faq"
    category: str = "general"
    intent_confidence: float = 0.0
    filter_applied: bool = False
    hybrid_used: bool = False
    citation_chunks: list[RetrievedChunk] | None = None
    rewritten_query: str = ""
    entities: list[str] | None = None
    decision: str = "ANSWER"
    confidence_reason: str = ""
    original_query: str = ""
    entity_payload: list[dict] | None = None
    understand: QueryUnderstanding | None = None
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    retrieval_mode: str = RetrievalMode.ENTERPRISE.value
    business_rules_applied: bool = False


def _should_drop_menu_heavy_chunk(chunk: RetrievedChunk) -> bool:
    if not is_menu_heavy_text(chunk.text):
        return False
    return not is_official_product_category_url(chunk.url or "")


def _keep_chunk_for_citation_pool(chunk: RetrievedChunk) -> bool:
    url = chunk.url or ""
    if is_low_value_document(chunk.document_id):
        return False
    if "ProductDetails" in url and any(
        marker in url for marker in ("CreditCardsID", "DepitCardsID", "PrepaidCardsID")
    ):
        return not is_junk_document(chunk.document_id)
    return (
        not is_junk_document(chunk.document_id)
        and not _should_drop_menu_heavy_chunk(chunk)
    )


def _apply_family_prioritizers(
    query: str,
    language: str,
    chunks: list[RetrievedChunk],
    intent: QueryIntent,
) -> list[RetrievedChunk]:
    """Apply product/card/account prioritizers only for matching intents."""
    chunks = prioritize_product_chunks(query, language, chunks)
    if intent.intent in CARD_INTENTS:
        chunks = prioritize_credit_card_chunks(query, language, chunks)
    if intent.intent in ACCOUNT_INTENTS:
        chunks = prioritize_account_chunks(query, language, chunks, intent=intent.intent)
    return chunks


def _apply_enterprise_ranking(
    chunks: list[RetrievedChunk],
    *,
    query: str,
    language: str,
    intent: QueryIntent,
    rules_on: bool,
) -> list[RetrievedChunk]:
    """Soft business-rule ranking. Never injects documents into the pool."""
    if not rules_on or not chunks:
        return chunks
    chunks = apply_business_rule_scoring(
        chunks, intent, query, language, enabled=True
    )
    chunks = apply_metadata_ranking(chunks, intent)
    chunks = _apply_family_prioritizers(query, language, chunks, intent)
    return chunks


def _build_citation_pool(
    chunks: list[RetrievedChunk],
    query: str,
    language: str,
    intent: QueryIntent,
    *,
    rules_on: bool,
) -> list[RetrievedChunk]:
    pool = [chunk for chunk in chunks if _keep_chunk_for_citation_pool(chunk)]
    if not pool:
        return []

    pool = rerank_chunks(pool, query, language)
    pool = _apply_enterprise_ranking(
        pool, query=query, language=language, intent=intent, rules_on=rules_on
    )

    if rules_on and intent.intent in {"credit_card", "card_types"}:
        card_pool = [
            chunk
            for chunk in pool
            if any(
                marker in (chunk.url or "")
                for marker in (
                    "CreditCardsID",
                    "DepitCardsID",
                    "PrepaidCardsID",
                    "#/AR/CreditCards",
                    "#/EN/CreditCards",
                )
            )
        ]
        # Prefer card URLs already retrieved — never pull external BM25 injects.
        if card_pool:
            pool = card_pool

    deduped: list[RetrievedChunk] = []
    seen_urls: set[str] = set()
    for chunk in pool:
        url_key = canonical_url_key(chunk.url or "")
        if not url_key or url_key in seen_urls:
            continue
        seen_urls.add(url_key)
        deduped.append(chunk)

    return deduped[: max(settings.retrieval_top_k * 3, 20)]


def _compute_confidence(chunks: list[RetrievedChunk], query: str, language: str) -> float:
    if not chunks:
        return 0.0
    top_scores = [chunk.score for chunk in chunks[:3]]
    semantic = sum(top_scores) / len(top_scores)
    terms = extract_query_terms(query, language)
    if not terms:
        return semantic
    keyword = max(keyword_overlap_score(chunk.text, terms, language) for chunk in chunks[:5])
    return min(1.0, semantic * 0.55 + keyword * 0.45)


class SearchService:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
        vector_store=None,  # noqa: ANN001 — test seam
    ) -> None:
        if hybrid_retriever is not None:
            self.hybrid_retriever = hybrid_retriever
        elif vector_store is not None:
            self.hybrid_retriever = HybridRetriever(vector_store=vector_store)
        else:
            self.hybrid_retriever = HybridRetriever()

    def retrieve(
        self,
        query: str,
        language: str = "auto",
        *,
        understanding: QueryUnderstanding | None = None,
        retrieval_mode: str | RetrievalMode | None = None,
        business_rules: bool | None = None,
    ) -> RetrievalResult:
        import time

        t0 = time.perf_counter()
        mode = parse_retrieval_mode(retrieval_mode or settings.retrieval_mode)
        rules_flag = (
            settings.business_rules_enabled if business_rules is None else business_rules
        )
        rules_on = business_rules_active(mode, business_rules_enabled=rules_flag)

        qu = understanding or understand_query(query, language)
        resolved_language = qu.language
        intent = qu.intent
        # Keep original user query; retrieve with optimized search_query only.
        original_query = qu.original_query
        normalized_query = qu.normalized_query or original_query
        expanded_query = qu.search_query
        entity_values = [e.value for e in qu.entities]
        entity_payload = qu.entity_dicts()
        embed_query = expanded_query

        candidate_k = max(
            settings.retrieval_top_k * settings.retrieval_candidate_multiplier,
            settings.rerank_pool_size,
        )
        # PURE_SEMANTIC: no intent metadata filter — independent semantic measurement.
        apply_filter = (
            rules_on
            and settings.intent_filter_enabled
            and should_apply_metadata_filter(intent, settings.intent_filter_confidence)
        )
        doc_types = list(intent.allowed_doc_types) if apply_filter else None

        raw_chunks, filter_applied = self.hybrid_retriever.retrieve(
            embed_query,
            resolved_language,
            candidate_k=candidate_k,
            intent=intent,
            apply_filter=apply_filter,
            doc_types=doc_types,
        )
        hybrid_used = settings.bm25_enabled
        retrieve_ms = (time.perf_counter() - t0) * 1000.0

        citation_source = list(raw_chunks)
        if rules_on and intent.intent in {"credit_card", "card_types"}:
            broad_chunks, _ = self.hybrid_retriever.retrieve(
                embed_query,
                resolved_language,
                candidate_k=max(candidate_k * 2, 80),
                intent=intent,
                apply_filter=False,
                doc_types=None,
            )
            citation_source.extend(broad_chunks)

        chunks = [
            chunk
            for chunk in raw_chunks
            if not is_junk_document(chunk.document_id)
            and not is_low_value_document(chunk.document_id)
            and not _should_drop_menu_heavy_chunk(chunk)
        ]
        if not chunks:
            chunks = [
                chunk
                for chunk in raw_chunks
                if not is_junk_document(chunk.document_id)
                and not is_low_value_document(chunk.document_id)
            ]

        citation_chunks = _build_citation_pool(
            citation_source,
            expanded_query,
            resolved_language,
            intent,
            rules_on=rules_on,
        )

        chunks = rerank_chunks(chunks, expanded_query, resolved_language)
        chunks = _apply_enterprise_ranking(
            chunks,
            query=expanded_query,
            language=resolved_language,
            intent=intent,
            rules_on=rules_on,
        )

        t_rerank = time.perf_counter()
        if settings.reranker_enabled and chunks:
            try:
                rerank_pool = max(settings.rerank_pool_size, settings.retrieval_top_k * 2)
                rerank_keep = min(settings.rerank_keep_size, settings.retrieval_top_k)
                chunks = get_reranker().rerank(
                    expanded_query,
                    chunks[:rerank_pool],
                    top_k=rerank_keep,
                )
                # Soft re-apply enterprise weights after cross-encoder (still no inject).
                chunks = _apply_enterprise_ranking(
                    chunks,
                    query=expanded_query,
                    language=resolved_language,
                    intent=intent,
                    rules_on=rules_on,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("reranker_failed_fallback_keyword", error=str(exc))
                chunks = chunks[: settings.rerank_keep_size]
        else:
            chunks = chunks[: settings.rerank_keep_size]
        rerank_ms = (time.perf_counter() - t_rerank) * 1000.0

        breakdown = compute_confidence(chunks, llm_confidence=0.0, intent=intent)
        legacy = _compute_confidence(chunks, expanded_query, resolved_language)
        confidence = round(min(1.0, breakdown.final * 0.75 + legacy * 0.25), 3)
        confidence_reason = breakdown.reason

        if mode == RetrievalMode.PURE_SEMANTIC:
            # Eval path: return ranked hybrid results without enterprise gate.
            decision_label = "ANSWER" if chunks else "NO_ANSWER"
            should_answer = bool(chunks)
            abstention_reason = None if should_answer else "no_relevant_chunks"
        else:
            decision = decide(
                intent,
                chunks,
                confidence=confidence,
                threshold=settings.confidence_threshold,
                canonical_candidates=None,
            )
            decision_label = decision.decision.value
            should_answer = (
                decision.decision.value == "ANSWER"
                and confidence >= settings.confidence_threshold
                and len(chunks) > 0
            )
            abstention_reason = None
            if not chunks:
                abstention_reason = "no_relevant_chunks"
            elif decision.decision.value == "NO_ANSWER":
                abstention_reason = decision.reason
            elif decision.decision.value == "RETRY_RELATED":
                abstention_reason = decision.reason
                should_answer = False
            elif confidence < settings.confidence_threshold:
                abstention_reason = "low_retrieval_confidence"

        logger.info(
            "retrieval_completed",
            language=resolved_language,
            intent=intent.intent,
            category=intent.category,
            intent_confidence=intent.confidence,
            filter_applied=filter_applied,
            hybrid_used=hybrid_used,
            chunk_count=len(chunks),
            confidence=confidence,
            should_answer=should_answer,
            decision=decision_label,
            embedding_model=settings.embedding_model,
            reranker=settings.reranker_enabled,
            rewritten_len=len(expanded_query),
            retrieval_mode=mode.value,
            business_rules_applied=rules_on,
        )

        return RetrievalResult(
            query=normalized_query,
            language=resolved_language,
            chunks=chunks,
            confidence=confidence,
            should_answer=should_answer,
            abstention_reason=abstention_reason,
            intent=intent.intent,
            category=intent.category,
            intent_confidence=intent.confidence,
            filter_applied=filter_applied,
            hybrid_used=hybrid_used,
            citation_chunks=citation_chunks,
            rewritten_query=expanded_query,
            entities=entity_values,
            decision=decision_label,
            confidence_reason=confidence_reason,
            original_query=original_query,
            entity_payload=entity_payload,
            understand=qu,
            retrieve_ms=retrieve_ms,
            rerank_ms=rerank_ms,
            retrieval_mode=mode.value,
            business_rules_applied=rules_on,
        )

```

## services/search_service/hybrid_retriever.py

```python
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

```

## ingestion/lexical/bm25_index.py

```python
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
from shared.url_canonical import canonical_url_key

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

    def match_urls(
        self,
        *needles: str,
        language: str | None = None,
        limit: int = 20,
    ) -> list[IndexedChunk]:
        """Return indexed chunks whose URL contains all needle substrings."""
        results: list[IndexedChunk] = []
        seen_urls: set[str] = set()
        for chunk in self._chunks:
            if language and chunk.language != language:
                continue
            url = chunk.url or ""
            if not url or not all(needle in url for needle in needles):
                continue
            url_key = canonical_url_key(url)
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)
            results.append(chunk)
            if len(results) >= limit:
                break
        return results


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

```

## services/rag/decision_engine.py

```python
"""Retrieval decision engine — ANSWER / RETRY / NO_ANSWER.

FORCE_CANONICAL is retained as a deprecated enum value for API/log compatibility
but is never emitted. Business rules soft-boost ranking earlier in the pipeline;
they must never pin or inject a document here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ingestion.embedding.vector_store import RetrievedChunk
from services.rag.metadata_filter import (
    preferred_categories_for_intent,
    soft_url_markers_for_intent,
)
from services.search_service.intent_classifier import QueryIntent


class RetrievalDecision(str, Enum):
    ANSWER = "ANSWER"
    RETRY_RELATED = "RETRY_RELATED"
    NO_ANSWER = "NO_ANSWER"
    # Deprecated: hard pin removed. Kept so older clients/log parsers do not crash.
    FORCE_CANONICAL = "FORCE_CANONICAL"


@dataclass(frozen=True)
class DecisionResult:
    decision: RetrievalDecision
    reason: str
    pinned_chunk: RetrievedChunk | None = None


def _top_matches_intent(chunk: RetrievedChunk | None, intent: QueryIntent) -> bool:
    if chunk is None:
        return False
    doc_type = (getattr(chunk, "doc_type", "general") or "general").lower()
    if doc_type in {d.lower() for d in intent.allowed_doc_types}:
        return True
    url = chunk.url or ""
    for marker in soft_url_markers_for_intent(intent.intent):
        if marker in url:
            return True
    preferred = preferred_categories_for_intent(intent.intent)
    category = (getattr(chunk, "category", "") or "").lower()
    if preferred and any(token in category or token in doc_type for token in preferred):
        return True
    return False


def decide(
    intent: QueryIntent,
    chunks: list[RetrievedChunk],
    *,
    confidence: float,
    threshold: float,
    canonical_candidates: list[RetrievedChunk] | None = None,
) -> DecisionResult:
    """Decide answer eligibility without injecting documents.

    ``canonical_candidates`` is accepted for backward compatibility and ignored
    for injection. Soft boosts already re-ranked ``chunks`` when enterprise rules
    were enabled.
    """
    _ = canonical_candidates  # retained signature; never used to force-inject
    top = chunks[0] if chunks else None

    if not chunks:
        return DecisionResult(RetrievalDecision.NO_ANSWER, "no_chunks")

    if intent.confidence >= 0.9 and not _top_matches_intent(top, intent):
        # Ask hybrid path / planner to broaden — never force a URL.
        return DecisionResult(RetrievalDecision.RETRY_RELATED, "top1_out_of_family_retry")

    if confidence < threshold:
        return DecisionResult(RetrievalDecision.NO_ANSWER, "low_confidence")

    if _top_matches_intent(top, intent) or intent.intent == "general_faq":
        return DecisionResult(RetrievalDecision.ANSWER, "top1_matches_intent")

    return DecisionResult(RetrievalDecision.NO_ANSWER, "unresolved_family_mismatch")


def _best_canonical(
    intent: QueryIntent,
    ranked: list[RetrievedChunk],
    candidates: list[RetrievedChunk],
) -> RetrievedChunk | None:
    """Locate preferred URL among already-retrieved candidates (no index inject)."""
    markers = soft_url_markers_for_intent(intent.intent)
    if not markers:
        return None
    pool = list(ranked) + list(candidates)
    for chunk in pool:
        url = chunk.url or ""
        if any(marker in url for marker in markers):
            return chunk
    return None


def pin_chunk_first(chunks: list[RetrievedChunk], pinned: RetrievedChunk) -> list[RetrievedChunk]:
    """Utility retained for tests; production search no longer pins via decision."""
    rest = [chunk for chunk in chunks if chunk.chunk_id != pinned.chunk_id]
    return [pinned, *rest]

```

## First 20 validation cases

```jsonl
{"query": "كام سعر الدولار النهاردة في الأهلي؟", "language": "ar", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter", "forbidden_url_contains": "CertificatesRates"}
{"query": "ايه سعر اليورو مقابل الجنيه عندكم؟", "language": "ar", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter", "forbidden_url_contains": "CertificatesRates"}
{"query": "محتاج جدول أسعار العملات الأجنبية", "language": "ar", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter"}
{"query": "عايز اعرف قيمة الجنيه قصاد الدولار", "language": "ar", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter"}
{"query": "قولي سعر شراء وبيع الدولار البنكي", "language": "ar", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter"}
{"query": "what is the USD to EGP buying price today", "language": "en", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter"}
{"query": "NBE forex board for EUR", "language": "en", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter"}
{"query": "show me today's banknote FX table", "language": "en", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter"}
{"query": "how many pounds for one dollar at NBE", "language": "en", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter"}
{"query": "current USD sell rate please", "language": "en", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter"}
{"query": "بكام الاسترليني النهاردة؟", "language": "ar", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter"}
{"query": "عايز محول من دولار لجنيه", "language": "ar", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter"}
{"query": "FX rates for tourists in EGP", "language": "en", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter"}
{"query": "الأهلي بيعالجنيه كام دولار", "language": "ar", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter"}
{"query": "need live currency board NBE", "language": "en", "intent": "exchange_rate", "expected_url_contains": "ExchangeRatesAndCurrencyConverter"}
{"query": "عايز أقساط تمويل شخصي بدون ضامن", "language": "ar", "intent": "personal_loan", "expected_url_contains": "Loans", "forbidden_url_contains": "NewsCat"}
{"query": "قد ايه تمويل فردي من البنك؟", "language": "ar", "intent": "personal_loan", "expected_url_contains": "Loans"}
{"query": "شروط أخذ قرض لأغراض شخصية", "language": "ar", "intent": "personal_loan", "expected_url_contains": "Loans"}
{"query": "محتاج تمويل للسفر أو الجواز", "language": "ar", "intent": "personal_loan", "expected_url_contains": "Loans"}
{"query": "نسبة الفائدة على التمويل الشخصي", "language": "ar", "intent": "personal_loan", "expected_url_contains": "Loans"}
```

## RETRY evidence and limitation

`decide()` emits `RETRY_RELATED` for an intent confidence >= 0.90 with a top-result family mismatch. `SearchService.retrieve()` sets `should_answer=False` for this decision. `SearchOrchestrator.search()` checks that flag before cache and answer construction and returns an abstention. There is no post-decision retrieval retry in this path. `HybridRetriever.retrieve()` independently performs actual related-type/broad retrieval when its initial filtered candidate counts are insufficient. The existing unit test `test_decision_retries_instead_of_force_canonical` verifies the label and absence of a pinned chunk, not execution of another retrieval.
