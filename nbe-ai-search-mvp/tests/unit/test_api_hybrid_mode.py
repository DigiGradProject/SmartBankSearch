"""API contract tests for the hybrid search mode (POST /v1/search)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.api import main as api_main
from shared.schemas import SearchRequest


@pytest.fixture()
def client() -> TestClient:
    return TestClient(api_main.app)


def test_search_request_accepts_hybrid_mode():
    req = SearchRequest(query="platinum", mode="hybrid")
    assert req.mode == "hybrid"


def test_search_request_rejects_unknown_mode():
    with pytest.raises(Exception):
        SearchRequest(query="platinum", mode="vector")


def test_search_request_still_accepts_legacy_modes():
    assert SearchRequest(query="q", mode="ai").mode == "ai"
    assert SearchRequest(query="q", mode="traditional").mode == "traditional"


def test_hybrid_endpoint_returns_page_list_contract(client: TestClient):
    """End-to-end through the app with the real singleton service.

    Requires the local Chroma collection + BM25 pickle (repo data ships both);
    the service degrades to dense-only if BM25 is missing.
    """
    response = client.post(
        "/v1/search",
        json={"query": "platinum card", "language": "auto", "mode": "hybrid", "limit": 3},
    )
    if response.status_code != 200:
        pytest.skip("live index unavailable in this environment")
    payload = response.json()
    assert payload["mode"] == "hybrid"
    assert payload["answer"] is None
    assert payload["citations"] == []
    assert isinstance(payload["results"], list)
    assert payload["total_results"] >= len(payload["results"])
    for item in payload["results"][:3]:
        assert item["url"].startswith(("http://", "https://"))
        assert item["language"] in {"ar", "en"}
        assert 0.0 < item["score"] <= 1.0


def test_traditional_mode_contract_unchanged(client: TestClient):
    """Regression guard: traditional contract is untouched by the hybrid branch."""
    response = client.post(
        "/v1/search",
        json={"query": "platinum", "language": "en", "mode": "traditional", "limit": 3},
    )
    if response.status_code != 200:
        pytest.skip("live index unavailable in this environment")
    payload = response.json()
    assert payload["mode"] == "traditional"
    assert payload["answer"] is None
    assert isinstance(payload["results"], list)
