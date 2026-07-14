from ingestion.embedding.vector_store import RetrievedChunk
from services.search_service.card_catalog import (
    build_card_citations,
    build_card_types_answer,
    build_credit_cards_answer,
    is_card_types_query,
    is_credit_cards_overview_query,
)
from services.search_service.intent_classifier import classify_query


def test_card_types_intent():
    intent = classify_query("انواع البطاقات البنكيه", "ar")
    assert intent.intent == "card_types"


def test_is_card_types_query():
    assert is_card_types_query("انواع البطاقات البنكيه", "ar")
    assert not is_card_types_query("بطاقات ائتمان", "ar")


def test_build_card_types_answer_lists_three_families():
    answer = build_card_types_answer([], "ar")
    assert answer is not None
    assert "بطاقات الائتمان" in answer
    assert "بطاقات الخصم المباشر" in answer
    assert "المدفوعة مقدما" in answer
    assert "فيزا كلاسيك" in answer
    assert "ميزة" in answer


def test_credit_cards_overview_query():
    assert is_credit_cards_overview_query("بطاقات الائتمان", "ar")
    assert not is_credit_cards_overview_query("انواع البطاقات البنكيه", "ar")
    assert not is_credit_cards_overview_query("بطاقات الخصم المباشر", "ar")


def test_build_credit_cards_answer_is_rich():
    answer = build_credit_cards_answer([], "ar")
    assert answer is not None
    assert "فيزا كلاسيك" in answer
    assert "فيزا انفينيت" in answer
    assert "الأهلي بوينتس" in answer
    assert "بناءً على محتوى" not in answer
    assert len(answer) > 400


def test_build_card_citations_prefers_product_pages_over_generic_nav():
    chunks = [
        RetrievedChunk(
            "stub",
            "nav",
            "بطاقات الائتمان",
            "https://www.nbe.com.eg/NBE/E/#/AR/CreditCards",
            "ar",
            "بطاقات الائتمان",
            1.0,
        ),
        RetrievedChunk(
            "cat",
            "credit_cat",
            "البنك الأهلى المصرى - بطاقات الإئتمان",
            'https://www.nbe.com.eg/NBE/E/#/AR/ProductCategory?inParams={"CategoryID":"CreditCardsID"}',
            "ar",
            "فيزا كلاسيك ماستركارد استاندرد",
            0.7,
        ),
        RetrievedChunk(
            "visa",
            "visa_classic",
            "البنك الأهلى المصرى - فيزا كلاسيك",
            'https://www.nbe.com.eg/NBE/E/#/AR/ProductDetails?inParams={"CategoryID":"CreditCardsID","ProductID":"visa"}',
            "ar",
            "فيزا كلاسيك مميزات البطاقة",
            0.75,
        ),
        RetrievedChunk(
            "mc",
            "mc_air",
            "البنك الأهلى المصرى - ماستركارد مصر للطيران",
            'https://www.nbe.com.eg/NBE/E/#/AR/ProductDetails?inParams={"CategoryID":"CreditCardsID","ProductID":"air"}',
            "ar",
            "ماستركارد مصر للطيران",
            0.72,
        ),
    ]
    citations = build_card_citations(chunks, max_items=5, include_families=("credit",))
    urls = [c.url for c in citations]
    assert "https://www.nbe.com.eg/NBE/E/#/AR/CreditCards" not in urls
    assert any("CreditCardsID" in url and "ProductCategory" in url for url in urls)
    assert sum("ProductDetails" in url for url in urls) >= 1
    assert len(citations) >= 2


