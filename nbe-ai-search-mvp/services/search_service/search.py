from dataclasses import dataclass, replace

from services.search_service.hybrid_retriever import HybridRetriever
from services.search_service.intent_boost import apply_intent_scoring
from services.search_service.product_detail import prioritize_product_chunks
from services.search_service.card_catalog import prioritize_credit_card_chunks
from services.search_service.account_rank import prioritize_account_chunks
from services.rag.hybrid_rank import apply_metadata_ranking
from services.rag.confidence import compute_confidence
from services.rag.decision_engine import RetrievalDecision, decide, pin_chunk_first
from services.rag.metadata_filter import canonical_markers_for_intent
from services.rag.query_understanding import QueryUnderstanding, understand_query
from services.search_service.intent_classifier import QueryIntent, should_apply_metadata_filter
from services.search_service.keyword_rank import extract_query_terms, keyword_overlap_score, rerank_chunks
from services.search_service.reranker import get_reranker
from shared.config import settings
from shared.url_canonical import canonical_url_key
from shared.document_quality import (
    is_junk_document,
    is_low_value_document,
    is_menu_heavy_text,
    is_official_product_category_url,
)
from shared.logging import get_logger
from ingestion.embedding.vector_store import RetrievedChunk
from ingestion.lexical.bm25_index import get_bm25_index

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


def _collect_canonical_candidates(
    intent: QueryIntent,
    language: str,
) -> list[RetrievedChunk]:
    markers = canonical_markers_for_intent(intent.intent)
    if not markers or not settings.force_canonical_inject:
        return []
    bm25 = get_bm25_index()
    if bm25.size <= 0:
        return []
    candidates: list[RetrievedChunk] = []
    seen: set[str] = set()
    for marker in markers:
        for indexed in bm25.match_urls(marker, language=language, limit=5):
            key = canonical_url_key(indexed.url or "")
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(bm25.to_retrieved_chunk(indexed, 0.99))
    return candidates


def _force_inject_canonical(
    intent: QueryIntent,
    language: str,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    if not settings.force_canonical_inject:
        return chunks
    markers = canonical_markers_for_intent(intent.intent)
    if not markers:
        return chunks

    def _matches(chunk: RetrievedChunk) -> bool:
        url = chunk.url or ""
        return any(marker in url for marker in markers)

    if any(_matches(chunk) for chunk in chunks[:2]):
        # Already near top — still pin the best match first.
        for idx, chunk in enumerate(chunks):
            if _matches(chunk):
                if idx == 0:
                    return chunks
                pinned = replace(chunk, score=max(chunk.score, 0.99))
                return pin_chunk_first(chunks, pinned)
        return chunks

    candidates = _collect_canonical_candidates(intent, language)
    if not candidates:
        return chunks

    best = replace(candidates[0], score=0.99)
    logger.info(
        "force_canonical_inject",
        intent=intent.intent,
        url=best.url,
    )
    return pin_chunk_first(chunks, best)


def _build_citation_pool(
    chunks: list[RetrievedChunk],
    query: str,
    language: str,
    intent: QueryIntent,
) -> list[RetrievedChunk]:
    pool = [chunk for chunk in chunks if _keep_chunk_for_citation_pool(chunk)]
    if not pool:
        return []

    pool = rerank_chunks(pool, query, language)
    pool = apply_intent_scoring(pool, intent, query, language)
    pool = apply_metadata_ranking(pool, intent)
    pool = _apply_family_prioritizers(query, language, pool, intent)
    pool = _force_inject_canonical(intent, language, pool)

    if intent.intent in {"credit_card", "card_types"}:
        card_pool = [
            chunk
            for chunk in pool
            if any(
                marker in (chunk.url or "")
                for marker in ("CreditCardsID", "DepitCardsID", "PrepaidCardsID", "#/AR/CreditCards", "#/EN/CreditCards")
            )
        ]
        if card_pool:
            pool = card_pool

    if intent.intent == "credit_card":
        bm25 = get_bm25_index()
        if bm25.size > 0:
            for indexed in bm25.match_urls(
                "CreditCardsID",
                "ProductDetails",
                language=language,
                limit=15,
            ):
                supplement = bm25.to_retrieved_chunk(indexed, 0.55)
                if _keep_chunk_for_citation_pool(supplement):
                    pool.append(supplement)

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
    ) -> RetrievalResult:
        import time

        t0 = time.perf_counter()
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
        apply_filter = settings.intent_filter_enabled and should_apply_metadata_filter(
            intent, settings.intent_filter_confidence
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
        if intent.intent in {"credit_card", "card_types"}:
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
                if not is_junk_document(chunk.document_id) and not is_low_value_document(chunk.document_id)
            ]

        citation_chunks = _build_citation_pool(citation_source, expanded_query, resolved_language, intent)

        chunks = rerank_chunks(chunks, expanded_query, resolved_language)
        chunks = apply_intent_scoring(chunks, intent, expanded_query, resolved_language)
        chunks = apply_metadata_ranking(chunks, intent)
        chunks = _apply_family_prioritizers(expanded_query, resolved_language, chunks, intent)
        chunks = _force_inject_canonical(intent, resolved_language, chunks)

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
                chunks = apply_intent_scoring(chunks, intent, expanded_query, resolved_language)
                chunks = apply_metadata_ranking(chunks, intent)
                chunks = _apply_family_prioritizers(expanded_query, resolved_language, chunks, intent)
                chunks = _force_inject_canonical(intent, resolved_language, chunks)
            except Exception as exc:  # noqa: BLE001
                logger.warning("reranker_failed_fallback_keyword", error=str(exc))
                chunks = chunks[: settings.rerank_keep_size]
        else:
            chunks = chunks[: settings.rerank_keep_size]
        rerank_ms = (time.perf_counter() - t_rerank) * 1000.0

        canonical_candidates = _collect_canonical_candidates(intent, resolved_language)
        breakdown = compute_confidence(chunks, llm_confidence=0.0, intent=intent)
        legacy = _compute_confidence(chunks, expanded_query, resolved_language)
        confidence = round(min(1.0, breakdown.final * 0.75 + legacy * 0.25), 3)
        confidence_reason = breakdown.reason

        decision = decide(
            intent,
            chunks,
            confidence=confidence,
            threshold=settings.confidence_threshold,
            canonical_candidates=canonical_candidates,
        )
        if decision.decision == RetrievalDecision.FORCE_CANONICAL and decision.pinned_chunk is not None:
            chunks = pin_chunk_first(chunks, replace(decision.pinned_chunk, score=0.99))
            chunks = chunks[: settings.rerank_keep_size]
            breakdown = compute_confidence(chunks, llm_confidence=0.0, intent=intent)
            confidence = breakdown.final
            confidence_reason = breakdown.reason
            decision = decide(
                intent,
                chunks,
                confidence=confidence,
                threshold=settings.confidence_threshold,
                canonical_candidates=canonical_candidates,
            )

        should_answer = (
            decision.decision == RetrievalDecision.ANSWER
            and confidence >= settings.confidence_threshold
            and len(chunks) > 0
        )
        abstention_reason = None
        if not chunks:
            abstention_reason = "no_relevant_chunks"
        elif decision.decision == RetrievalDecision.NO_ANSWER:
            abstention_reason = decision.reason
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
            decision=decision.decision.value,
            decision_reason=decision.reason,
            embedding_model=settings.embedding_model,
            reranker=settings.reranker_enabled,
            rewritten_len=len(expanded_query),
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
            decision=decision.decision.value,
            confidence_reason=confidence_reason,
            original_query=original_query,
            entity_payload=entity_payload,
            understand=qu,
            retrieve_ms=retrieve_ms,
            rerank_ms=rerank_ms,
        )
