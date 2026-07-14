from dataclasses import dataclass

from services.search_service.hybrid_retriever import HybridRetriever
from services.search_service.intent_boost import apply_intent_scoring
from services.search_service.product_detail import prioritize_product_chunks
from services.search_service.card_catalog import prioritize_credit_card_chunks
from services.search_service.intent_classifier import QueryIntent, classify_query, should_apply_metadata_filter
from services.search_service.keyword_rank import extract_query_terms, keyword_overlap_score, rerank_chunks
from services.search_service.language import detect_language, prepare_query
from services.search_service.query_expand import expand_query
from services.search_service.reranker import get_reranker
from services.search_service.synonyms import expand_with_synonyms
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
    pool = prioritize_product_chunks(query, language, pool)
    pool = prioritize_credit_card_chunks(query, language, pool)

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


def _expand_with_intent(query: str, language: str, intent: QueryIntent) -> str:
    base = expand_query(query, language)
    base = expand_with_synonyms(base, language)
    extra = intent.expand_ar if language == "ar" else intent.expand_en
    if extra and extra not in base:
        return f"{base} {extra}".strip()
    return base


class SearchService:
    def __init__(self, hybrid_retriever: HybridRetriever | None = None) -> None:
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()

    def retrieve(self, query: str, language: str = "auto") -> RetrievalResult:
        resolved_language = detect_language(query, language)
        normalized_query = prepare_query(query, resolved_language)
        intent = classify_query(normalized_query, resolved_language)
        expanded_query = _expand_with_intent(normalized_query, resolved_language, intent)
        embed_query = prepare_query(expanded_query, resolved_language)

        candidate_k = settings.retrieval_top_k * settings.retrieval_candidate_multiplier
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
        chunks = prioritize_product_chunks(expanded_query, resolved_language, chunks)
        chunks = prioritize_credit_card_chunks(expanded_query, resolved_language, chunks)

        if settings.reranker_enabled and chunks:
            try:
                chunks = get_reranker().rerank(
                    expanded_query,
                    chunks[: max(settings.retrieval_top_k * 2, 12)],
                    top_k=settings.retrieval_top_k,
                )
                chunks = apply_intent_scoring(chunks, intent, expanded_query, resolved_language)
                chunks = prioritize_product_chunks(expanded_query, resolved_language, chunks)
                chunks = prioritize_credit_card_chunks(expanded_query, resolved_language, chunks)
            except Exception as exc:  # noqa: BLE001
                logger.warning("reranker_failed_fallback_keyword", error=str(exc))
                chunks = chunks[: settings.retrieval_top_k]
        else:
            chunks = chunks[: settings.retrieval_top_k]

        confidence = _compute_confidence(chunks, expanded_query, resolved_language)
        if intent.confidence >= settings.intent_filter_confidence and chunks:
            top_doc_type = getattr(chunks[0], "doc_type", "general")
            if top_doc_type in intent.allowed_doc_types:
                confidence = min(1.0, confidence + 0.08)

        should_answer = confidence >= settings.confidence_threshold and len(chunks) > 0
        abstention_reason = None
        if not chunks:
            abstention_reason = "no_relevant_chunks"
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
            embedding_model=settings.embedding_model,
            reranker=settings.reranker_enabled,
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
        )
