"""Hybrid ranking helpers (metadata-aware adjustments).

Uses preferred categories as soft weights — not exact URL substring filters.
"""

from __future__ import annotations

from dataclasses import replace

from ingestion.embedding.vector_store import RetrievedChunk
from services.rag.metadata_filter import preferred_categories_for_intent
from services.search_service.intent_classifier import QueryIntent


def apply_metadata_ranking(
    chunks: list[RetrievedChunk],
    intent: QueryIntent,
) -> list[RetrievedChunk]:
    """Boost chunks whose page_type/subcategory align with the query intent."""
    if not chunks or intent.intent == "general_faq":
        return chunks

    preferred_page_types = {
        "account_open": {"product", "category", "faq"},
        "credit_card": {"product", "category"},
        "card_types": {"category", "product"},
        "certificate_rate": {"product", "category"},
        "certificate_types": {"category", "product"},
        "personal_loan": {"product", "category"},
        "offers": {"offer"},
        "news": {"news"},
        "reports": {"report"},
    }.get(intent.intent, set())

    preferred_topics = preferred_categories_for_intent(intent.intent)
    penalized = {"navigation"}

    scored: list[RetrievedChunk] = []
    for chunk in chunks:
        score = chunk.score
        page_type = getattr(chunk, "page_type", "") or ""
        doc_type = (getattr(chunk, "doc_type", "") or "").lower()
        category = (getattr(chunk, "category", "") or "").lower()
        if preferred_page_types and page_type in preferred_page_types:
            score += 0.08
        if page_type in penalized and intent.intent not in {"news", "offers"}:
            score -= 0.12
        if intent.category and intent.category == getattr(chunk, "category", ""):
            score += 0.05
        # Soft preferred-category match (exchange/currency/forex …) instead of URL force.
        if preferred_topics and (
            any(token in category for token in preferred_topics)
            or any(token in doc_type for token in preferred_topics)
        ):
            score += 0.06
        scored.append(replace(chunk, score=max(0.0, min(1.0, score))))
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored
