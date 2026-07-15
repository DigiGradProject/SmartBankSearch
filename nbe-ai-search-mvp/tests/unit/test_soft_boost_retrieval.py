"""Unit tests for soft business-rule scoring and retrieval modes."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

from ingestion.embedding.vector_store import RetrievedChunk
from services.rag.business_rules import apply_business_rule_scoring, compute_business_weight
from services.rag.decision_engine import RetrievalDecision, decide
from services.rag.metadata_filter import preferred_categories_for_intent
from services.search_service.intent_classifier import QueryIntent
from services.search_service.search import SearchService
from shared.retrieval_mode import RetrievalMode, business_rules_active


def _chunk(
    chunk_id: str,
    *,
    url: str,
    doc_type: str,
    score: float = 0.5,
    title: str = "t",
    category: str = "",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        title=title,
        url=url,
        language="ar",
        text="نص تجريبي للمنتج البنكي",
        score=score,
        doc_type=doc_type,
        category=category,
    )


def test_preferred_categories_for_exchange_rate():
    preferred = preferred_categories_for_intent("exchange_rate")
    assert "exchange" in preferred
    assert "currency" in preferred
    assert "forex" in preferred


def test_business_weight_boosts_matching_metadata_not_inject():
    intent = QueryIntent(
        intent="exchange_rate",
        category="exchange_rates",
        confidence=0.95,
        allowed_doc_types=("exchange_rate",),
    )
    fx = _chunk(
        "fx",
        url="https://nbe/#/AR/ExchangeRatesAndCurrencyConverter",
        doc_type="exchange_rate",
        score=0.60,
        category="exchange",
    )
    account = _chunk(
        "acc",
        url="https://nbe/#/AR/ProductDetails?CurrentAccountsID",
        doc_type="account",
        score=0.90,
        category="account",
    )
    w_fx = compute_business_weight(fx, intent, query="سعر الدولار", language="ar")
    w_acc = compute_business_weight(account, intent, query="سعر الدولار", language="ar")
    assert w_fx > 1.0
    assert w_acc < 1.0


def test_soft_boost_never_adds_new_documents():
    intent = QueryIntent(
        intent="exchange_rate",
        category="exchange_rates",
        confidence=0.95,
        allowed_doc_types=("exchange_rate",),
    )
    only = [
        _chunk(
            "acc",
            url="https://nbe/accounts",
            doc_type="account",
            score=0.9,
        )
    ]
    ranked = apply_business_rule_scoring(only, intent, "كام سعر الدولار", "ar")
    assert len(ranked) == 1
    assert ranked[0].chunk_id == "acc"


def test_soft_boost_promotes_preferred_url_already_in_pool():
    intent = QueryIntent(
        intent="exchange_rate",
        category="exchange_rates",
        confidence=0.95,
        allowed_doc_types=("exchange_rate",),
    )
    chunks = [
        _chunk(
            "acc",
            url="https://nbe/#/AR/ProductDetails?CurrentAccountsID",
            doc_type="account",
            score=0.85,
        ),
        _chunk(
            "fx",
            url="https://nbe/#/AR/ExchangeRatesAndCurrencyConverter",
            doc_type="exchange_rate",
            score=0.55,
        ),
    ]
    ranked = apply_business_rule_scoring(chunks, intent, "دولار بكام", "ar")
    assert ranked[0].chunk_id == "fx"


def test_decision_retries_instead_of_force_canonical():
    intent = QueryIntent(
        intent="exchange_rate",
        category="exchange_rates",
        confidence=0.95,
        allowed_doc_types=("exchange_rate",),
    )
    account = _chunk("acc", url="https://nbe/accounts", doc_type="account", score=0.9)
    fx = _chunk(
        "fx",
        url="https://nbe/#/AR/ExchangeRatesAndCurrencyConverter",
        doc_type="exchange_rate",
        score=0.5,
    )
    result = decide(
        intent,
        [account],
        confidence=0.8,
        threshold=0.42,
        canonical_candidates=[fx],
    )
    assert result.decision == RetrievalDecision.RETRY_RELATED
    assert result.pinned_chunk is None


def test_business_rules_active_matrix():
    assert business_rules_active(RetrievalMode.ENTERPRISE, business_rules_enabled=True)
    assert not business_rules_active(RetrievalMode.ENTERPRISE, business_rules_enabled=False)
    assert not business_rules_active(RetrievalMode.PURE_SEMANTIC, business_rules_enabled=True)


def test_pure_semantic_skips_business_rules(monkeypatch):
    fx = _chunk(
        "fx",
        url="https://nbe/#/AR/ExchangeRatesAndCurrencyConverter",
        doc_type="exchange_rate",
        score=0.4,
    )
    account = _chunk(
        "acc",
        url="https://nbe/accounts",
        doc_type="account",
        score=0.9,
    )

    store = MagicMock()
    store.query.return_value = [account, fx]
    service = SearchService(vector_store=store)

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
                allowed_doc_types=("exchange_rate",),
            ),
            original_query=q,
            normalized_query=q,
            search_query=q,
            entities=[],
            entity_dicts=lambda: [],
        ),
    )

    pure = service.retrieve("دولار", "ar", retrieval_mode=RetrievalMode.PURE_SEMANTIC)
    assert pure.retrieval_mode == "PURE_SEMANTIC"
    assert pure.business_rules_applied is False
    assert pure.filter_applied is False

    enterprise = service.retrieve("دولار", "ar", retrieval_mode=RetrievalMode.ENTERPRISE)
    assert enterprise.business_rules_applied is True
