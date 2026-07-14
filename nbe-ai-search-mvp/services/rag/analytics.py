"""Retrieval analytics — JSONL events for Grafana / offline analysis."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnalyticsEvent:
    query: str
    intent: str
    rewritten_query: str
    entities: list[dict[str, Any]] = field(default_factory=list)
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0
    cache_hit: bool = False
    top_documents: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    confidence_reason: str = ""
    decision: str = ""
    faithfulness: str | None = None
    language: str = "ar"
    answered: bool = False
    ts: float = field(default_factory=time.time)


def write_analytics_event(event: AnalyticsEvent, path: Path | None = None) -> None:
    if not settings.analytics_log_enabled:
        return
    target = path or settings.analytics_log_path
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("analytics_write_failed", error=str(exc))
