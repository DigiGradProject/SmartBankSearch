"""Calibrated confidence scoring for enterprise RAG answers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ingestion.embedding.vector_store import RetrievedChunk
from services.search_service.intent_classifier import QueryIntent
from shared.url_canonical import canonical_url_key


@dataclass(frozen=True)
class ConfidenceBreakdown:
    embedding_similarity: float
    bm25_score: float
    metadata_match: float
    intent_confidence: float
    reranker_score: float
    rank_margin: float
    final: float
    # Calibration components
    top1_rerank: float = 0.0
    mean_top5: float = 0.0
    category_consistency: float = 0.0
    chunk_agreement: float = 0.0
    source_diversity: float = 0.0
    reason: str = ""
    # Legacy aliases for callers
    retriever_score: float = 0.0
    llm_confidence: float = 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _metadata_match(chunk: RetrievedChunk | None, intent: QueryIntent | None) -> float:
    if chunk is None or intent is None or intent.intent == "general_faq":
        return 0.5
    doc_type = getattr(chunk, "doc_type", "general") or "general"
    if doc_type in intent.allowed_doc_types:
        return 1.0
    url = chunk.url or ""
    soft_markers = {
        "exchange_rate": "ExchangeRates",
        "personal_loan": "Loan",
        "credit_card": "CreditCard",
        "account_open": "Account",
    }
    marker = soft_markers.get(intent.intent, "")
    if marker and marker in url:
        return 0.7
    return 0.0


def _category_consistency(chunks: list[RetrievedChunk], intent: QueryIntent | None) -> float:
    if not chunks:
        return 0.0
    top_n = chunks[:5]
    categories = [getattr(c, "category", "general") or "general" for c in top_n]
    if not categories:
        return 0.0
    most_common, count = Counter(categories).most_common(1)[0]
    consistency = count / len(categories)
    if intent and intent.category and most_common == intent.category:
        consistency = min(1.0, consistency + 0.15)
    return _clamp(consistency)


def _chunk_agreement(chunks: list[RetrievedChunk]) -> float:
    """Agreement via score dispersion (high = scores clustered near top)."""
    if not chunks:
        return 0.0
    top_n = chunks[:5]
    scores = [c.score for c in top_n]
    if len(scores) == 1:
        return _clamp(scores[0])
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    # Low variance + high mean → high agreement
    return _clamp(mean * (1.0 - min(1.0, variance * 4.0)))


def _source_diversity(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    top_n = chunks[:5]
    urls = {canonical_url_key(c.url or "") for c in top_n if c.url}
    urls.discard("")
    # Prefer moderate diversity (2–4 unique sources among top5)
    unique = len(urls) or 1
    if unique == 1:
        return 0.55
    if unique <= 4:
        return 0.85 + 0.05 * (unique - 2)
    return 0.70


def _build_reason(
    *,
    intent: QueryIntent | None,
    chunks: list[RetrievedChunk],
    category_consistency: float,
    chunk_agreement: float,
    top1: float,
    mean_top5: float,
) -> str:
    if not chunks:
        return "No retrieved documents; confidence is zero."
    cats = [getattr(c, "category", "general") or "general" for c in chunks[:5]]
    dominant, count = Counter(cats).most_common(1)[0]
    label = dominant.replace("_", " ").title()
    if category_consistency >= 0.7 and chunk_agreement >= 0.55:
        return (
            f"Top {min(5, len(chunks))} documents belong to {label} category "
            f"with high semantic agreement (top1={top1:.0%}, mean_top5={mean_top5:.0%})."
        )
    if intent and intent.confidence >= 0.9:
        return (
            f"Intent '{intent.intent}' at {intent.confidence:.0%} confidence; "
            f"{count}/{min(5, len(chunks))} top docs in {label}."
        )
    return (
        f"Mixed evidence across {label} "
        f"(consistency={category_consistency:.0%}, agreement={chunk_agreement:.0%})."
    )


def compute_confidence(
    chunks: list[RetrievedChunk],
    *,
    llm_confidence: float = 0.0,
    embedding_similarity: float | None = None,
    intent: QueryIntent | None = None,
    bm25_score: float | None = None,
) -> ConfidenceBreakdown:
    """
    Calibrated confidence:
      C = 0.15·intent + 0.25·top1_rerank + 0.15·mean_top5
        + 0.20·category_consistency + 0.15·chunk_agreement + 0.10·source_diversity
    Plus legacy metadata/bm25 signals blended lightly for continuity.
    """
    llm = _clamp(llm_confidence)
    if not chunks:
        return ConfidenceBreakdown(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            reason="No retrieved documents; confidence is zero.",
            llm_confidence=llm,
        )

    top = chunks[0]
    second = chunks[1] if len(chunks) > 1 else None
    top5 = chunks[:5]
    top1_rerank = _clamp(top.score)
    mean_top5 = _clamp(sum(c.score for c in top5) / len(top5))
    dense = _clamp(embedding_similarity if embedding_similarity is not None else top.score)
    bm25 = _clamp(bm25_score if bm25_score is not None else top.score)
    meta = _metadata_match(top, intent)
    intent_conf = _clamp(intent.confidence if intent else 0.0)
    margin = _clamp((top.score - second.score) if second else top.score)
    cat_cons = _category_consistency(chunks, intent)
    agreement = _chunk_agreement(chunks)
    diversity = _source_diversity(chunks)

    calibrated = (
        0.15 * intent_conf
        + 0.25 * top1_rerank
        + 0.15 * mean_top5
        + 0.20 * cat_cons
        + 0.15 * agreement
        + 0.10 * diversity
    )
    # Blend small share of legacy dense/bm25/meta for continuity with Phase A scoring.
    legacy = 0.10 * dense + 0.05 * bm25 + 0.05 * meta
    final = _clamp(0.85 * calibrated + 0.15 * (legacy / 0.20) * 0.20 + 0.08 * llm)

    if intent and intent.confidence >= 0.9 and meta == 0.0:
        final = _clamp(final * 0.55)

    reason = _build_reason(
        intent=intent,
        chunks=chunks,
        category_consistency=cat_cons,
        chunk_agreement=agreement,
        top1=top1_rerank,
        mean_top5=mean_top5,
    )

    return ConfidenceBreakdown(
        embedding_similarity=dense,
        bm25_score=bm25,
        metadata_match=meta,
        intent_confidence=intent_conf,
        reranker_score=top1_rerank,
        rank_margin=margin,
        final=round(final, 3),
        top1_rerank=top1_rerank,
        mean_top5=mean_top5,
        category_consistency=round(cat_cons, 3),
        chunk_agreement=round(agreement, 3),
        source_diversity=round(diversity, 3),
        reason=reason,
        retriever_score=round((dense + bm25) / 2, 3),
        llm_confidence=llm,
    )
