"""Retrieval mode control plane for semantic eval vs enterprise production."""

from __future__ import annotations

from enum import Enum


class RetrievalMode(str, Enum):
    """How retrieval applies enterprise banking safety rules.

    PURE_SEMANTIC
        Hybrid retrieval + reranker only. No business-rule score multipliers,
        no intent metadata filters, no confidence abstention for evaluation.
        Used to measure true semantic retrieval quality.

    ENTERPRISE
        Intent → business-rule soft boosts → hybrid → reranker → confidence
        gate → LLM. Business rules influence ranking only; they never inject
        or force a document into the result set.

    KEYWORD
        Traditional keyword search: BM25-only, no LLM, no reranker, no
        business rules. Used by the traditional-search experience
        (services/search_service/traditional.py); when it reaches
        SearchService.retrieve it behaves like PURE_SEMANTIC (no gate).
    """

    PURE_SEMANTIC = "PURE_SEMANTIC"
    ENTERPRISE = "ENTERPRISE"
    KEYWORD = "KEYWORD"


def parse_retrieval_mode(value: str | RetrievalMode | None) -> RetrievalMode:
    if value is None:
        return RetrievalMode.ENTERPRISE
    if isinstance(value, RetrievalMode):
        return value
    normalized = str(value).strip().upper().replace("-", "_")
    try:
        return RetrievalMode(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Unknown retrieval_mode={value!r}; expected PURE_SEMANTIC, ENTERPRISE or KEYWORD"
        ) from exc


def business_rules_active(
    mode: RetrievalMode,
    *,
    business_rules_enabled: bool = True,
) -> bool:
    """Enterprise safety layer applies only in ENTERPRISE mode when enabled."""
    return mode == RetrievalMode.ENTERPRISE and business_rules_enabled


def gate_active(mode: RetrievalMode) -> bool:
    """Confidence/decision gate applies only in ENTERPRISE mode."""
    return mode == RetrievalMode.ENTERPRISE
