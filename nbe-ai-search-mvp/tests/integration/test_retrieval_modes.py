"""Integration tests for retrieval modes and soft-boost safety invariants."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ingestion.embedding.vector_store import RetrievedChunk
from services.search_service.intent_classifier import QueryIntent
from services.search_service.search import SearchService
from shared.retrieval_mode import RetrievalMode


def _chunk(chunk_id: str, *, url: str, doc_type: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        title=chunk_id,
        url=url,
        language="ar",
        text="نص بنكي للاختبار",
        score=score,
        doc_type=doc_type,
    )


@pytest.fixture()
def service(monkeypatch):
    store = MagicMock()
    account = _chunk(
        "acc",
        url="https://nbe/#/AR/ProductDetails?CurrentAccountsID",
        doc_type="account",
        score=0.92,
    )
    fx = _chunk(
        "fx",
        url="https://nbe/#/AR/ExchangeRatesAndCurrencyConverter",
        doc_type="exchange_rate",
        score=0.50,
    )
    store.query.return_value = [account, fx]
    svc = SearchService(vector_store=store)
    monkeypatch.setattr("services.search_service.search.settings.reranker_enabled", False)
    monkeypatch.setattr("services.search_service.search.settings.bm25_enabled", False)
    monkeypatch.setattr(
        "services.search_service.search.understand_query",
        lambda q, lang: MagicMock(
            language="ar",
            intent=QueryIntent(
                intent="exchange_rate",
                category="exchange_rates",
                confidence=0.95,
                allowed_doc_types=("exchange_rate", "currency_converter"),
            ),
            original_query=q,
            normalized_query=q,
            search_query=q,
            entities=[],
            entity_dicts=lambda: [],
        ),
    )
    return svc


def test_enterprise_mode_soft_boosts_without_inject(service: SearchService):
    result = service.retrieve(
        "كام سعر الدولار",
        "ar",
        retrieval_mode=RetrievalMode.ENTERPRISE,
    )
    assert result.business_rules_applied is True
    ids = {c.chunk_id for c in result.chunks}
    assert ids <= {"acc", "fx"}
    # Soft boost should lift FX when it was already retrieved.
    assert result.chunks[0].chunk_id == "fx"


def test_pure_semantic_preserves_raw_ranking_signal(service: SearchService):
    result = service.retrieve(
        "كام سعر الدولار",
        "ar",
        retrieval_mode=RetrievalMode.PURE_SEMANTIC,
    )
    assert result.retrieval_mode == "PURE_SEMANTIC"
    assert result.business_rules_applied is False
    assert result.filter_applied is False
    # Without business rules, the higher hybrid score (account) stays first.
    assert result.chunks[0].chunk_id == "acc"


def test_api_remains_unchanged_for_search_shape(client_optional=None):
    # Guard: SearchRequest schema still accepts only query/language/debug.
    from shared.schemas import SearchRequest

    req = SearchRequest(query="test", language="ar")
    assert req.debug is False
    dumped = req.model_dump()
    assert set(dumped.keys()) == {"query", "language", "debug"}
