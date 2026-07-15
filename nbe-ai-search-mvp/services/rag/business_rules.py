"""Enterprise banking safety layer — soft ranking signals only.

Business rules MUST NEVER inject or force a document into the result set.
They only multiply candidate scores already produced by hybrid retrieval.

This module is an enterprise precision layer for high-value banking intents,
not an AI retrieval mechanism. Semantic retrieval remains independently
measurable via RetrievalMode.PURE_SEMANTIC.
"""

from __future__ import annotations

from dataclasses import replace

from ingestion.embedding.vector_store import RetrievedChunk
from services.rag.metadata_filter import (
    PREFERRED_CATEGORIES,
    preferred_categories_for_intent,
    soft_url_markers_for_intent,
)
from services.search_service.intent_boost import apply_intent_scoring
from services.search_service.intent_classifier import QueryIntent
from shared.arabic_normalize import normalize_arabic


# Multiplicative weights (applied as score *= weight). Tuned for soft influence.
CATEGORY_MATCH_WEIGHT = 1.22
METADATA_DOC_TYPE_WEIGHT = 1.18
URL_MARKER_WEIGHT = 1.28
PRODUCT_TYPE_WEIGHT = 1.12
MISMATCH_WEIGHT = 0.72
DEFAULT_WEIGHT = 1.0

# Cap so business rules cannot dominate the reranker.
MAX_BUSINESS_WEIGHT = 1.55
MIN_BUSINESS_WEIGHT = 0.55


def compute_business_weight(
    chunk: RetrievedChunk,
    intent: QueryIntent,
    *,
    query: str = "",
    language: str = "ar",
) -> float:
    """Return multiplicative business_weight for one retrieved candidate."""
    if intent.intent == "general_faq" or intent.confidence <= 0.0:
        return DEFAULT_WEIGHT

    weight = DEFAULT_WEIGHT
    conf_scale = max(0.35, min(1.0, intent.confidence))

    preferred = preferred_categories_for_intent(intent.intent)
    chunk_category = (getattr(chunk, "category", "") or "").lower()
    chunk_doc_type = (getattr(chunk, "doc_type", "general") or "general").lower()
    page_type = (getattr(chunk, "page_type", "") or "").lower()
    url = chunk.url or ""
    title = (chunk.title or "").lower()

    # Category / preferred-topic soft match
    if preferred and (
        any(cat in chunk_category for cat in preferred)
        or any(cat in chunk_doc_type for cat in preferred)
        or any(cat in page_type for cat in preferred)
        or any(cat in title for cat in preferred)
    ):
        weight *= 1.0 + (CATEGORY_MATCH_WEIGHT - 1.0) * conf_scale

    # Metadata doc_type alignment with intent allow-list
    allowed = {d.lower() for d in intent.allowed_doc_types}
    if allowed:
        if chunk_doc_type in allowed:
            weight *= 1.0 + (METADATA_DOC_TYPE_WEIGHT - 1.0) * conf_scale
        elif chunk_doc_type not in {"general", "faq", ""}:
            weight *= 1.0 - (1.0 - MISMATCH_WEIGHT) * conf_scale

    # Soft URL similarity (markers are preferences, never injection keys)
    markers = soft_url_markers_for_intent(intent.intent)
    if markers and any(marker in url for marker in markers):
        weight *= 1.0 + (URL_MARKER_WEIGHT - 1.0) * conf_scale

    # Product / page_type boost for product-family intents
    product_intents = {
        "credit_card",
        "debit_card",
        "card_types",
        "account_open",
        "personal_loan",
        "certificate_types",
        "certificate_buy",
        "certificate_rate",
    }
    if intent.intent in product_intents and page_type in {"product", "category"}:
        weight *= 1.0 + (PRODUCT_TYPE_WEIGHT - 1.0) * conf_scale

    # Exchange-rate: avoid certificate bleed without forcing FX URL
    if intent.intent == "exchange_rate":
        q = normalize_arabic(query) if language == "ar" else query.lower()
        if "شهاد" in title or "certificate" in title.lower():
            weight *= MISMATCH_WEIGHT
        if "certificate" in chunk_doc_type and "صرف" not in q and "exchange" not in q:
            weight *= MISMATCH_WEIGHT

    return max(MIN_BUSINESS_WEIGHT, min(MAX_BUSINESS_WEIGHT, weight))


def apply_business_rule_scoring(
    chunks: list[RetrievedChunk],
    intent: QueryIntent,
    query: str,
    language: str = "ar",
    *,
    enabled: bool = True,
) -> list[RetrievedChunk]:
    """Multiply candidate scores by enterprise business weights.

    Never adds chunks that were not already retrieved.
    """
    if not enabled or not chunks or intent.intent == "general_faq":
        return chunks

    # Preserve existing additive intent nuances, then apply multiplicative safety layer.
    scored = apply_intent_scoring(chunks, intent, query, language)
    weighted: list[RetrievedChunk] = []
    for chunk in scored:
        bw = compute_business_weight(chunk, intent, query=query, language=language)
        weighted.append(replace(chunk, score=max(0.0, min(1.0, chunk.score * bw))))
    weighted.sort(key=lambda item: item.score, reverse=True)
    return weighted


def preferred_category_summary(intent_name: str) -> tuple[str, ...]:
    """Public helper for docs / eval explainability."""
    return PREFERRED_CATEGORIES.get(intent_name, ())
