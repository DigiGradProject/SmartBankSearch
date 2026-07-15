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
    """

    PURE_SEMANTIC = "PURE_SEMANTIC"
    ENTERPRISE = "ENTERPRISE"


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
            f"Unknown retrieval_mode={value!r}; expected PURE_SEMANTIC or ENTERPRISE"
        ) from exc


def business_rules_active(
    mode: RetrievalMode,
    *,
    business_rules_enabled: bool = True,
) -> bool:
    """Enterprise safety layer applies only in ENTERPRISE mode when enabled."""
    return mode == RetrievalMode.ENTERPRISE and business_rules_enabled
