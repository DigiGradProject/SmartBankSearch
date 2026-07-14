from dataclasses import dataclass

from services.search_service.hybrid_retriever import HybridRetriever
from services.search_service.intent_boost import apply_intent_scoring
from services.search_service.intent_classifier import QueryIntent, classify_query, should_apply_metadata_filter
from services.search_service.keyword_rank import extract_query_terms, keyword_overlap_score, rerank_chunks
from services.search_service.language import detect_language, prepare_query
from services.search_service.query_expand import expand_query
from services.search_service.reranker import get_reranker
from services.search_service.synonyms import expand_with_synonyms
from shared.config import settings
from shared.document_quality import is_junk_document, is_low_value_document, is_menu_heavy_text
from shared.logging import get_logger
from ingestion.embedding.vector_store import RetrievedChunk

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

        chunks = [
            chunk
            for chunk in raw_chunks
            if not is_junk_document(chunk.document_id)
            and not is_low_value_document(chunk.document_id)
            and not is_menu_heavy_text(chunk.text)
        ]
        if not chunks:
            chunks = [
                chunk
                for chunk in raw_chunks
                if not is_junk_document(chunk.document_id) and not is_low_value_document(chunk.document_id)
            ]

        chunks = rerank_chunks(chunks, expanded_query, resolved_language)
        chunks = apply_intent_scoring(chunks, intent, expanded_query)

        if settings.reranker_enabled and chunks:
            try:
                chunks = get_reranker().rerank(
                    expanded_query,
                    chunks[: max(settings.retrieval_top_k * 2, 12)],
                    top_k=settings.retrieval_top_k,
                )
                chunks = apply_intent_scoring(chunks, intent, expanded_query)
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
        )
