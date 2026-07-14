"""Retrieval decision engine — ANSWER / RETRY / FORCE_CANONICAL / NO_ANSWER."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ingestion.embedding.vector_store import RetrievedChunk
from services.rag.metadata_filter import CANONICAL_URL_MARKERS
from services.search_service.intent_classifier import QueryIntent


class RetrievalDecision(str, Enum):
    ANSWER = "ANSWER"
    RETRY_RELATED = "RETRY_RELATED"
    FORCE_CANONICAL = "FORCE_CANONICAL"
    NO_ANSWER = "NO_ANSWER"


@dataclass(frozen=True)
class DecisionResult:
    decision: RetrievalDecision
    reason: str
    pinned_chunk: RetrievedChunk | None = None


def _top_matches_intent(chunk: RetrievedChunk | None, intent: QueryIntent) -> bool:
    if chunk is None:
        return False
    doc_type = getattr(chunk, "doc_type", "general") or "general"
    if doc_type in intent.allowed_doc_types:
        return True
    url = chunk.url or ""
    for marker in CANONICAL_URL_MARKERS.get(intent.intent, ()):
        if marker in url:
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
    top = chunks[0] if chunks else None

    # Prefer pinning a canonical page present in index but missing/low in ranking.
    if intentional_canonical := _best_canonical(intent, chunks, canonical_candidates or []):
        if top is None or not _top_matches_intent(top, intent):
            return DecisionResult(
                RetrievalDecision.FORCE_CANONICAL,
                "canonical_page_available_but_top_out_of_family",
                intentional_canonical,
            )
        # Canonical exists but ranked below Top-1 out-of-family — still force.
        if top and not _top_matches_intent(top, intent):
            return DecisionResult(
                RetrievalDecision.FORCE_CANONICAL,
                "top1_out_of_family",
                intentional_canonical,
            )

    if not chunks:
        return DecisionResult(RetrievalDecision.NO_ANSWER, "no_chunks")

    if intent.confidence >= 0.9 and not _top_matches_intent(top, intent):
        if canonical_candidates:
            return DecisionResult(
                RetrievalDecision.FORCE_CANONICAL,
                "high_intent_confidence_out_of_family_top1",
                canonical_candidates[0],
            )
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
    markers = CANONICAL_URL_MARKERS.get(intent.intent, ())
    if not markers:
        return None
    pool = list(candidates) + list(ranked)
    for chunk in pool:
        url = chunk.url or ""
        if any(marker in url for marker in markers):
            return chunk
    return None


def pin_chunk_first(chunks: list[RetrievedChunk], pinned: RetrievedChunk) -> list[RetrievedChunk]:
    rest = [chunk for chunk in chunks if chunk.chunk_id != pinned.chunk_id]
    return [pinned, *rest]
