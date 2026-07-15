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
