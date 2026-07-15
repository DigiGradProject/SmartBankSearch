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
