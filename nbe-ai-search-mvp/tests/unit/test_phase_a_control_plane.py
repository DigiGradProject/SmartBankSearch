"""Phase A control-plane hotfix tests (analysis.md)."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

from ingestion.embedding.vector_store import RetrievedChunk
from ingestion.classification.doc_classifier import classify_document
from services.rag.confidence import compute_confidence
from services.rag.decision_engine import RetrievalDecision, decide, pin_chunk_first
from services.rag.metadata_filter import should_broaden
from services.search_service.account_rank import prioritize_account_chunks
from services.search_service.hybrid_retriever import HybridRetriever
from services.search_service.intent_boost import apply_intent_scoring
from services.search_service.intent_classifier import QueryIntent


def _chunk(
    chunk_id: str,
    *,
    url: str,
    doc_type: str,
    score: float = 0.5,
    title: str = "t",
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
    )


def test_should_not_broaden_when_high_conf_has_one_hit():
    intent = QueryIntent(
        intent="exchange_rate",
        category="exchange_rates",
        confidence=0.95,
        allowed_doc_types=("exchange_rate",),
    )
    assert should_broaden(intent, dense_count=1, bm25_count=0) is False
    assert should_broaden(intent, dense_count=0, bm25_count=0) is True


def test_account_prioritizer_gated_by_intent():
    account = _chunk(
        "a",
        url="https://nbe/#/AR/ProductDetails?inParams={\"CategoryID\":\"CurrentAccountsID\"}",
        doc_type="account",
        score=0.4,
    )
    fx = _chunk(
        "fx",
        url="https://nbe/#/AR/ExchangeRatesAndCurrencyConverter",
        doc_type="exchange_rate",
        score=0.9,
    )
    ranked = prioritize_account_chunks(
        "سعر الصرف",
        "ar",
        [fx, account],
        intent="exchange_rate",
    )
    assert ranked[0].chunk_id == "fx"

    account_ranked = prioritize_account_chunks(
        "فتح حساب جاري",
        "ar",
        [fx, account],
        intent="account_open",
    )
    assert account_ranked[0].chunk_id == "a"


def test_intent_boost_penalizes_accounts_on_fx_and_news_on_loans():
    fx_intent = QueryIntent(
        intent="exchange_rate",
        category="exchange_rates",
        confidence=0.95,
        allowed_doc_types=("exchange_rate",),
    )
    chunks = [
        _chunk(
            "acc",
            url="https://nbe/#/AR/ProductDetails?inParams={\"CategoryID\":\"CurrentAccountsID\"}",
            doc_type="account",
            score=0.95,
            title="الحساب الجاري",
        ),
        _chunk(
            "fx",
            url="https://nbe/#/AR/ExchangeRatesAndCurrencyConverter",
            doc_type="exchange_rate",
            score=0.70,
            title="أسعار العملات",
        ),
    ]
    ranked = apply_intent_scoring(chunks, fx_intent, "سعر الصرف", "ar")
    assert ranked[0].chunk_id == "fx"

    loan_intent = QueryIntent(
        intent="personal_loan",
        category="loans",
        confidence=0.95,
        allowed_doc_types=("loan",),
    )
    loan_chunks = [
        _chunk(
            "news",
            url="https://nbe/#/AR/ProductDetails?inParams={\"CategoryID\":\"NewsCatProduct\"}",
            doc_type="news",
            score=0.95,
            title="خبر",
        ),
        _chunk(
            "loan",
            url="https://nbe/#/AR/ProductCategory?inParams={\"CategoryID\":\"PersonalLoansCatID\"}",
            doc_type="loan",
            score=0.60,
            title="قرض شخصي",
        ),
    ]
    ranked_loans = apply_intent_scoring(loan_chunks, loan_intent, "قرض شخصي", "ar")
    assert ranked_loans[0].chunk_id == "loan"


def test_loan_category_id_classified_as_loan_and_newscat_as_news():
    loan = classify_document(
        url='https://nbe/#/AR/ProductCategory?inParams={"CategoryID":"PersonalLoansCatID"}',
        title="قروض",
        content="تفاصيل القرض",
        language="ar",
    )
    assert loan.doc_type == "loan"

    news = classify_document(
        url='https://nbe/#/AR/ProductDetails?inParams={"CategoryID":"NewsCatXYZ"}',
        title="خبر",
        content="نص الخبر",
        language="ar",
    )
    assert news.doc_type == "news"


def test_decision_force_canonical_when_top_out_of_family():
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
    result = decide(intent, [account], confidence=0.8, threshold=0.42, canonical_candidates=[fx])
    assert result.decision == RetrievalDecision.FORCE_CANONICAL
    pinned = pin_chunk_first([account], fx)
    assert pinned[0].chunk_id == "fx"


def test_confidence_penalizes_metadata_mismatch_at_high_intent():
    intent = QueryIntent(
        intent="exchange_rate",
        category="exchange_rates",
        confidence=0.95,
        allowed_doc_types=("exchange_rate",),
    )
    mismatch = compute_confidence(
        [_chunk("acc", url="u", doc_type="account", score=0.9)],
        intent=intent,
    )
    match = compute_confidence(
        [
            _chunk(
                "fx",
                url="https://nbe/#/AR/ExchangeRatesAndCurrencyConverter",
                doc_type="exchange_rate",
                score=0.9,
            )
        ],
        intent=intent,
    )
    assert match.final > mismatch.final


def test_hybrid_keeps_filter_with_one_dense_hit():
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
        score=0.8,
    )
    store = MagicMock()
    store.query.side_effect = lambda *args, **kwargs: (
        [fx] if kwargs.get("doc_types") else [_chunk("acc", url="accounts", doc_type="account", score=0.9)]
    )
    retriever = HybridRetriever(vector_store=store, bm25_index=None)
    chunks, filter_applied = retriever.retrieve(
        "سعر الصرف",
        "ar",
        candidate_k=5,
        intent=intent,
        apply_filter=True,
        doc_types=["exchange_rate"],
    )
    assert filter_applied is True
    assert chunks[0].chunk_id == "fx"
    # Must not have fallen back to broad (second store.query without doc_types).
    assert all(call.kwargs.get("doc_types") for call in store.query.call_args_list)
