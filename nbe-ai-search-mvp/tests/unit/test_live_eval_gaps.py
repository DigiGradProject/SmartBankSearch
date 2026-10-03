"""Regression tests for the live-evaluation failure fixes (2026-10-03 batch).

A 47-query live /v1/search evaluation surfaced 8 failures; these tests pin the
fixes:
  - routing gaps: car loan (EN/AR), AR debit cards ("بطاقات الخصم"), savings
    account interest, CBE mortgage initiatives → intent_prototypes.py +
    banking_synonyms.ar.json
  - citation order when the reranker saturates at 1.0 (Belady certificates,
    Al Ahly Mobile retail vs corporate) → builder.py title tie-break
  - thin-page title match ("الأسئلة الشائعة") abstaining at confidence 0.405
    (threshold 0.42) → confidence.py title-match floor
"""

from __future__ import annotations

from ingestion.embedding.vector_store import RetrievedChunk
from services.context_builder.builder import ContextBuilder
from services.rag.confidence import compute_confidence
from services.rag.semantic_intent import classify_semantic
from shared.config import settings


# --------------------------------------------------------------------------
# Routing (margin gate must now let these win with the new prototypes)
# --------------------------------------------------------------------------


def _route(query: str, language: str) -> str:
    """classify_semantic enforces the floor + margin gate (reject → general_faq),
    so asserting the intent also proves the gate passed with the new prototypes."""
    return classify_semantic(query, language).intent


def test_car_loan_en_routes_to_personal_loan() -> None:
    assert _route("car loan", "en") == "personal_loan"


def test_car_loan_ar_routes_to_personal_loan() -> None:
    assert _route("قرض سيارات", "ar") == "personal_loan"


def test_arabic_debit_cards_route_to_debit_card() -> None:
    assert _route("بطاقات الخصم", "ar") == "debit_card"


def test_savings_interest_routes_to_account_open() -> None:
    assert _route("savings account interest", "en") == "account_open"


def test_mortgage_initiatives_route_to_personal_loan() -> None:
    assert _route("مبادرات التمويل العقاري", "ar") == "personal_loan"


# --------------------------------------------------------------------------
# Citation tie-breaks (reranker saturates at 1.0)
# --------------------------------------------------------------------------


def _chunk(cid: str, url: str, title: str, text: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=f"d-{cid}",
        title=title,
        url=url,
        language="ar",
        text=text,
        score=score,
        doc_type="certificate",
        category="certificates",
    )


def test_belady_title_phrase_wins_citation_tie() -> None:
    """Query 'شهادات بلادي' with everything reranking at 1.0 must cite the
    Belady page (title contains the query phrase) before the generic one."""
    generic = _chunk(
        "c1",
        "https://www.nbe.com.eg/NBE/E/#/AR/ProductCategory?inParams=%7B%22CategoryID%22%3A%22CertificatesID%22%7D",
        "البنك الأهلى المصرى - شهادات الادخار",
        "شهادات ادخار بالجنيه المصري بشروط ومواعيد متعددة.",
    )
    belady = _chunk(
        "c2",
        "https://www.nbe.com.eg/NBE/E/#/AR/ProductCategory?inParams=%7B%22CategoryID%22%3A%22BeladyCertificateID%22%7D",
        "البنك الأهلى المصرى - شهادات بلادي",
        "شهادات بلادي تصدر للمصريين المقيمين بالخارج بالجنيه الاسترليني واليورو.",
    )
    # Generic first in retrieval order — the tie-break must override it.
    citations = ContextBuilder()._select_citations("شهادات بلادي", [generic, belady])
    assert citations, "expected citations"
    assert "BeladyCertificateID" in citations[0].url


def test_retail_mobile_wins_citation_tie_by_ratio() -> None:
    """'NBE mobile app': the retail Al Ahly Mobile title covers more of its
    (template-stripped) tokens than '... Mobile Corporate' at equal score."""
    retail = RetrievedChunk(
        "r1", "d-r1", "National Bank of Egypt - Al Ahly Mobile",
        "https://www.nbe.com.eg/NBE/E/#/EN/ProductDetails?inParams=%7B%22CategoryID%22%3A%2250%22%7D",
        "en",
        "Al Ahly Mobile is the NBE mobile banking application for retail customers.",
        1.0,
        doc_type="product", category="products",
    )
    corporate = RetrievedChunk(
        "r2", "d-r2", "National Bank of Egypt - Al Ahly Mobile Corporate",
        "https://www.nbe.com.eg/NBE/E/#/EN/ProductDetails?inParams=%7B%22CategoryID%22%3A%22ElectronicServicesCorporate%22%7D",
        "en",
        "Al Ahly Mobile Corporate serves corporate customers mobile banking needs.",
        1.0,
        doc_type="product", category="products",
    )
    cars = RetrievedChunk(
        "r3", "d-r3", "National Bank of Egypt - Cars And Services",
        "https://www.nbe.com.eg/NBE/E/#/EN/ProductCategory?inParams=%7B%22CategoryID%22%3A%22carsandservices%22%7D",
        "en",
        "Cars and services offers and benefits for vehicle owners.",
        1.0,
        doc_type="product", category="products",
    )
    citations = ContextBuilder()._select_citations("NBE mobile app", [cars, corporate, retail])
    assert citations
    assert "%2250%22" in citations[0].url, f"retail mobile should be first, got {citations[0].url}"


def test_belady_product_pages_are_citable():
    """Belady Euro/Sterling/EGP pages have real content and must be citable
    (only the empty 'Belady USD' variants stay blacklisted)."""
    from services.context_builder.builder import _is_citable

    euro = RetrievedChunk(
        "b1", "d-b1", "البنك الأهلى المصرى - بلادى باليورو",
        "https://www.nbe.com.eg/NBE/E/#/AR/ProductDetails?inParams="
        "%7B%22CategoryID%22%3A%22beladythreeyears%22%2C%22ProductID%22%3A%2216610%22%7D",
        "ar",
        "شهادة بلادي باليورو تصدر للمصريين المقيمين بالخارج بأجل ثلاث سنوات.",
        1.0,
        doc_type="certificate", category="certificates",
    )
    usd = RetrievedChunk(
        "b2", "d-b2", "National Bank of Egypt - Belady USD",
        "https://www.nbe.com.eg/NBE/E/#/EN/ProductDetails?inParams="
        "%7B%22CategoryID%22%3A%22beladyoneyear%22%2C%22ProductID%22%3A%22Belady%20USD_16615%22%7D",
        "en",
        "Belady USD one year certificate.",
        1.0,
        doc_type="certificate", category="certificates",
    )
    assert _is_citable(euro)
    assert not _is_citable(usd)


# --------------------------------------------------------------------------
# Title-match confidence floor (thin FAQ page abstained at 0.405 < 0.42)
# --------------------------------------------------------------------------


def _thin_faq_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        "f1", "d-f1", "الأسئلة الشائعة",
        "https://www.nbe.com.eg/NBE/E/#/AR/FAQs",
        "ar",
        "قائمة الأسئلة الشائعة وإجاباتها عن خدمات البنك.",
        0.183,
        doc_type="faq", category="general",
    )


def test_title_match_floors_confidence() -> None:
    chunks = [_thin_faq_chunk()]
    boosted = compute_confidence(chunks, intent=None, query="الأسئلة الشائعة")
    assert boosted.final >= settings.title_match_confidence_floor


def test_no_title_match_keeps_low_confidence() -> None:
    chunks = [_thin_faq_chunk()]
    unchanged = compute_confidence(chunks, intent=None, query="سعر الدولار اليوم")
    assert unchanged.final < settings.title_match_confidence_floor


def test_title_match_floor_knob_exists() -> None:
    assert 0.45 <= settings.title_match_confidence_floor <= 0.70


def test_phrase_match_chunk_cited_below_global_floor():
    """The FAQ page reranks at 0.183 (< citation_min_score 0.45) — without the
    phrase-match re-anchor the answer came back with ZERO citations."""
    faq = RetrievedChunk(
        "q1", "d-q1", "الأسئلة الشائعة",
        "https://www.nbe.com.eg/NBE/E/#/AR/FAQs",
        "ar",
        "قائمة الأسئلة الشائعة وإجاباتها عن خدمات البنك.",
        0.183,
        doc_type="faq", category="general",
    )
    citations = ContextBuilder()._select_citations("الأسئلة الشائعة", [faq])
    assert citations, "FAQ page must be cited despite score below global floor"
    assert "FAQs" in citations[0].url
