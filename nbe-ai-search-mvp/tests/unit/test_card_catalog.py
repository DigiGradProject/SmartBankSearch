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


def _card_chunk(cid: str, title: str, category_id: str, doc_type: str):
    return RetrievedChunk(
        cid,
        cid,
        title,
        f'https://www.nbe.com.eg/#/AR/ProductDetails?inParams={{"CategoryID":"{category_id}"}}',
        "ar",
        f"صفحة رسمية عن {title}",
        0.9,
        doc_type=doc_type,
        category="cards",
    )


def test_build_card_types_answer_uses_only_retrieved_families():
    chunks = [
        _card_chunk("credit", "فيزا كلاسيك", "CreditCardsID", "credit_card"),
        _card_chunk("debit", "ميزة", "DepitCardsID", "debit_card"),
        _card_chunk("prepaid", "بطاقة ميزة المدفوعة مقدما", "PrepaidCardsID", "debit_card"),
    ]
    answer = build_card_types_answer(chunks, "ar")
    assert answer is not None
    assert "بطاقات الائتمان" in answer
    assert "بطاقات الخصم المباشر" in answer
    assert "المدفوعة مقدما" in answer
    assert "فيزا كلاسيك" in answer
    assert "ميزة" in answer
    assert build_card_types_answer([], "ar") is None


def test_credit_cards_overview_query():
    assert is_credit_cards_overview_query("بطاقات الائتمان", "ar")
    assert not is_credit_cards_overview_query("انواع البطاقات البنكيه", "ar")
    assert not is_credit_cards_overview_query("بطاقات الخصم المباشر", "ar")


def test_build_credit_cards_answer_does_not_invent_features():
    chunks = [
        _card_chunk("credit", "فيزا كلاسيك", "CreditCardsID", "credit_card"),
    ]
    answer = build_credit_cards_answer(chunks, "ar")
    assert answer is not None
    assert "فيزا كلاسيك" in answer
    assert "فيزا انفينيت" not in answer
    assert "الأهلي بوينتس" not in answer
    assert build_credit_cards_answer([], "ar") is None


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


