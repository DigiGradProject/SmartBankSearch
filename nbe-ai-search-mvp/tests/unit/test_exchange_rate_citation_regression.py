"""Regression tests for the exchange-rate citation poisoning bug.

Root cause: title-like brand queries ("What is National Bank of Egypt - Al Ahly
X?") classified as exchange_rate (~0.52 cosine, floor was 0.42), whose intent
expansion "exchange rate currency converter banknote transfer rate" matched the
ExchangeRatesAndCurrencyConverter page title almost verbatim (rerank 1.0). The
score-only citation selection then cited only that page. See
docs/root-cause-exchange-rate-citations.md.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ingestion.embedding.vector_store import RetrievedChunk
from services.context_builder.builder import ContextBuilder
from shared.config import settings

EXCHANGE_URL = "https://www.nbe.com.eg/NBE/E/#/EN/ExchangeRatesAndCurrencyConverter"
POINTS_URL = (
    "https://www.nbe.com.eg/NBE/E/#/EN/ProductDetails"
    '?inParams={"CategoryID":"AlAhlyPoints"}'
)


def _chunks() -> list[RetrievedChunk]:
    """Reproduce the observed post-rerank scores from the production logs."""
    return [
        RetrievedChunk(
            "c1", "EN_ExchangeRates", "National Bank of Egypt - Exchange Rates And Currency Converter",
            EXCHANGE_URL, "en",
            "Check the daily banknote and transfer exchange rates published by the bank. "
            "The currency converter tool shows buying and selling prices for major foreign "
            "currencies including USD, EUR and GBP against the Egyptian pound.",
            1.0,
            doc_type="exchange_rate", category="exchange_rates",
        ),
        RetrievedChunk(
            "c2", "EN_ProductDetails_AlAhlyPoints", "National Bank of Egypt - Al Ahly Points",
            POINTS_URL, "en",
            "Al Ahly Points is a loyalty program that rewards customers whenever they use "
            "banking services. Earned points can be redeemed against fees, purchases and "
            "exclusive offers across participating merchants.",
            0.142,
            doc_type="product", category="products",
        ),
        RetrievedChunk(
            "c3", "EN_Cards_World", "National Bank of Egypt - World",
            "https://www.nbe.com.eg/NBE/E/#/EN/ProductDetails?inParams={\"CategoryID\":\"World\"}",
            "en",
            "The World credit card offers premium travel benefits, airport lounge access "
            "and comprehensive purchase protection for cardholders.",
            0.119,
            doc_type="credit_card", category="cards",
        ),
    ]


def test_offtopic_head_with_dominant_pool_is_dropped() -> None:
    """A saturated off-topic page must not crowd out the dominant category."""
    # Runs with compression enabled (the default) so the citation guard is
    # exercised behind the real compressor path.
    built = ContextBuilder().build("Al Ahly Points", _chunks())
    urls = [c.url for c in built.citations]
    assert EXCHANGE_URL not in urls
    assert POINTS_URL in urls


def test_offtopic_head_gap_threshold_is_configurable() -> None:
    """Below the configured gap the head is kept (no over-triggering)."""
    chunks = _chunks()
    # True source scored 0.95: gap 0.05 < citation_category_gap → keep head.
    chunks[1] = replace(chunks[1], score=0.95)
    built = ContextBuilder().build("Al Ahly Points", chunks)
    urls = [c.url for c in built.citations]
    assert EXCHANGE_URL in urls


def test_no_dominant_pool_keeps_head() -> None:
    """If the head category is the only category, nothing is dropped."""
    chunks = [replace(_chunks()[0], title="Only page")]
    built = ContextBuilder().build("exchange rates", chunks)
    assert [c.url for c in built.citations] == [EXCHANGE_URL]


def test_low_citation_floor_would_admit_true_source_when_reanchored() -> None:
    """Re-anchored floor admits the true source even below the global floor."""
    chunks = _chunks()
    assert chunks[1].score < settings.citation_min_score
    built = ContextBuilder().build("Al Ahly Points", chunks)
    assert POINTS_URL in [c.url for c in built.citations]


def test_semantic_intent_gates_apply_to_config() -> None:
    """Guard the two knobs the fix depends on (drift protection)."""
    assert settings.semantic_intent_min_score >= 0.55
    assert settings.semantic_intent_margin >= 0.05
    assert settings.citation_category_gap >= 0.25


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is National Bank of Egypt - Al Ahly Points?", "alahly points loyalty points"),
        ("What is National Bank of Egypt - Al Ahly Business?", "alahly business corporate banking business banking"),
    ],
)
def test_rewrite_still_appends_synonyms_but_not_intent_expansion(query: str, expected: str) -> None:
    """Mid-band brand queries keep their synonym additions but gain no
    intent expansion tokens (no 'exchange rate currency converter ...')."""
    from services.rag.query_rewrite import rewrite_query

    result = rewrite_query(query, "en")
    assert "exchange rate" not in result.rewritten.lower()
    assert expected in result.rewritten.lower()
