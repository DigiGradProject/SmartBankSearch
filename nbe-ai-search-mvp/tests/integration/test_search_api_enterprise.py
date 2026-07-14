"""Integration tests for additive search API fields and feedback."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from shared.schemas import SearchResponse
from services.search_service.search import RetrievalResult


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
    monkeypatch.setattr("shared.config.settings.semantic_cache_enabled", False)
    monkeypatch.setattr("shared.config.settings.self_eval_enabled", False)
    monkeypatch.setattr("shared.config.settings.analytics_log_path", tmp_path / "analytics.jsonl")
    monkeypatch.setattr("shared.config.settings.feedback_log_path", tmp_path / "feedback.jsonl")
    monkeypatch.setattr("shared.config.settings.audit_log_enabled", False)

    from services.api import main as api_main

    fake_retrieval = RetrievalResult(
        query="سعر الصرف",
        language="ar",
        chunks=[],
        confidence=0.2,
        should_answer=False,
        abstention_reason="low_retrieval_confidence",
        intent="exchange_rate",
        category="exchange_rates",
        intent_confidence=0.95,
        rewritten_query="سعر الصرف تحويل العملات",
        entities=["صرف"],
        entity_payload=[{"type": "product", "value": "صرف"}],
        confidence_reason="Low evidence",
        original_query="سعر الصرف",
        decision="NO_ANSWER",
    )

    async def fake_search(query: str, language: str = "auto", *, debug: bool = False) -> SearchResponse:
        resp = SearchResponse(
            answer=None,
            confidence=0.2,
            citations=[],
            answered=False,
            language="ar",
            abstention_reason="low_retrieval_confidence",
            confidence_reason="Low evidence",
            rewritten_query="سعر الصرف تحويل العملات",
            entities=[{"type": "product", "value": "صرف"}],
            cache_hit=False,
        )
        if debug:
            resp.explain = {"detected_intent": {"intent": "exchange_rate"}}
        return resp

    api_main.orchestrator.search = fake_search  # type: ignore[method-assign]
    return TestClient(api_main.app)


def test_search_additive_fields(client: TestClient):
    response = client.post("/v1/search", json={"query": "سعر الصرف", "language": "ar"})
    assert response.status_code == 200
    body = response.json()
    assert "confidence_reason" in body
    assert "rewritten_query" in body
    assert "entities" in body
    assert body.get("explain") is None


def test_search_debug_explains(client: TestClient):
    response = client.post(
        "/v1/search",
        json={"query": "سعر الصرف", "language": "ar", "debug": True},
    )
    assert response.status_code == 200
    assert response.json()["explain"]["detected_intent"]["intent"] == "exchange_rate"


def test_feedback_endpoint(client: TestClient, tmp_path, monkeypatch):
    monkeypatch.setattr("shared.config.settings.feedback_log_path", tmp_path / "feedback.jsonl")
    response = client.post(
        "/v1/feedback",
        json={
            "query_hash": "abcd1234",
            "vote": "helpful",
            "question": "سعر الصرف",
            "answer": "…",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["feedback_id"].startswith("fb_")
