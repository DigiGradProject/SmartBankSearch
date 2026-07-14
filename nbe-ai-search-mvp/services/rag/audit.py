"""Banking audit trail for AI Search answers (CBE-style explainability)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)


def _hash_query(query: str) -> str:
    return hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:16]


def write_audit_event(event: dict[str, Any]) -> None:
    if not settings.audit_log_enabled:
        return
    path: Path = settings.audit_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": datetime.now(UTC).isoformat(),
        **event,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    logger.info(
        "search_audit",
        query_hash=payload.get("query_hash"),
        intent=payload.get("intent"),
        answered=payload.get("answered"),
        confidence=payload.get("confidence"),
    )


def build_audit_event(
    *,
    query: str,
    language: str,
    intent: str,
    intent_confidence: float,
    chunk_ids: list[str],
    urls: list[str],
    confidence: float,
    answered: bool,
    abstention_reason: str | None,
    model_embedding: str,
    model_reranker: str,
    model_llm: str,
    rewritten_query: str = "",
    entities: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "query_hash": _hash_query(query),
        "language": language,
        "intent": intent,
        "intent_confidence": intent_confidence,
        "rewritten_query_hash": _hash_query(rewritten_query) if rewritten_query else "",
        "entities": entities or [],
        "chunk_ids": chunk_ids,
        "source_urls": urls,
        "confidence": confidence,
        "answered": answered,
        "abstention_reason": abstention_reason,
        "models": {
            "embedding": model_embedding,
            "reranker": model_reranker,
            "llm": model_llm,
        },
    }
