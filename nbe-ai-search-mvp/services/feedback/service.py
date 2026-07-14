"""Feedback service façade."""

from __future__ import annotations

from pathlib import Path

from services.feedback.store import save_feedback
from shared.schemas import FeedbackRequest, FeedbackResponse


class FeedbackService:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    def submit(self, request: FeedbackRequest) -> FeedbackResponse:
        feedback_id = save_feedback(request, path=self._path)
        return FeedbackResponse(feedback_id=feedback_id)
