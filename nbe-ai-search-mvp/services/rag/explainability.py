"""Debug-only explainability payload for AI Search."""

from __future__ import annotations

from typing import Any

from ingestion.embedding.vector_store import RetrievedChunk
from services.rag.query_understanding import QueryUnderstanding


def build_explain_payload(
    *,
    understanding: QueryUnderstanding,
    chunks: list[RetrievedChunk],
    confidence: float,
    confidence_reason: str,
    decision: str = "",
    filter_applied: bool = False,
    cache_hit: bool = False,
    plan_intents: list[str] | None = None,
) -> dict[str, Any]:
    categories = sorted(
        {
            (getattr(c, "category", None) or "general")
            for c in chunks
        }
    )
    docs = [
        {
            "title": c.title,
            "url": c.url,
            "category": getattr(c, "category", "general"),
            "doc_type": getattr(c, "doc_type", "general"),
            "score": round(c.score, 4),
            "reason": _selection_reason(c, understanding),
        }
        for c in chunks[:8]
    ]
    return {
        "detected_intent": {
            "intent": understanding.intent.intent,
            "category": understanding.intent.category,
            "confidence": understanding.intent.confidence,
        },
        "extracted_entities": understanding.entity_dicts(),
        "rewritten_query": understanding.search_query,
        "expanded_terms": understanding.expanded_terms,
        "confidence": confidence,
        "confidence_reason": confidence_reason,
        "retrieved_categories": categories,
        "retrieved_documents": docs,
        "decision": decision,
        "filter_applied": filter_applied,
        "cache_hit": cache_hit,
        "plan_intents": plan_intents or [understanding.intent.intent],
        "why_selected": (
            "Documents were selected via hybrid BM25+BGE-M3 retrieval, "
            "cross-encoder reranking, intent metadata filters, and canonical page injection when applicable."
        ),
    }


def _selection_reason(chunk: RetrievedChunk, understanding: QueryUnderstanding) -> str:
    parts: list[str] = []
    doc_type = getattr(chunk, "doc_type", "general") or "general"
    if doc_type in understanding.intent.allowed_doc_types:
        parts.append(f"matches intent doc_type={doc_type}")
    if chunk.score >= 0.8:
        parts.append("high reranker/hybrid score")
    category = getattr(chunk, "category", "") or ""
    if category and category == understanding.intent.category:
        parts.append(f"category={category}")
    return "; ".join(parts) or "retrieved in top-k hybrid pool"
