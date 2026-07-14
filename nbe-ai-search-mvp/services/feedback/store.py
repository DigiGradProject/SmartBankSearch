"""Feedback JSONL store for helpful / not_helpful votes."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from shared.config import settings
from shared.logging import get_logger
from shared.schemas import FeedbackRequest

logger = get_logger(__name__)


def save_feedback(request: FeedbackRequest, path: Path | None = None) -> str:
    feedback_id = f"fb_{uuid.uuid4().hex[:16]}"
    target = path or settings.feedback_log_path
    target.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "feedback_id": feedback_id,
        "ts": time.time(),
        "query_hash": request.query_hash,
        "vote": request.vote,
        "reason": request.reason,
        "question": request.question,
        "answer": request.answer,
        "docs": request.docs,
        "confidence": request.confidence,
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("feedback_saved", feedback_id=feedback_id, vote=request.vote)
    return feedback_id
