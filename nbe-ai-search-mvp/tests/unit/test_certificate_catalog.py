from ingestion.embedding.vector_store import RetrievedChunk
from services.search_service.certificate_catalog import (
    build_certificate_buy_answer,
    build_certificate_types_answer,
    is_certificate_types_query,
)
from services.search_service.intent_classifier import classify_query


def test_certificate_types_intent():
    q = "انواع الشهادات البنكيه"
    assert classify_query(q, "ar").intent == "certificate_types"
    assert is_certificate_types_query(q, "ar") is True


def test_build_certificate_types_answer_from_stubs():
    chunks = [
        RetrievedChunk(
            "c1",
            "d1",
            "شهادات بلادي",
            "https://nbe/CertificatesID",
            "ar",
            "شهادات بلادي وشهادات الادخار بالعملة المحلية وشهادات الاستثمار",
            0.9,
            doc_type="certificate",
        )
    ]
    answer = build_certificate_types_answer(chunks, "ar")
    assert answer
    assert "شهادات بلادي" in answer
    assert "المسترجعة" in answer
    assert build_certificate_types_answer([], "ar") is None


def test_certificate_types_retrieval_optional():
    assert is_certificate_types_query("انواع الشهادات البنكيه", "ar") is True


def test_certificate_buy_answer_is_explanatory_and_hides_numeric_product_id():
    chunks = [
        RetrievedChunk(
            "c1", "d1", "National Bank of Egypt - Belady Euro",
            "https://nbe/ProductID=16616", "en", "Belady Euro certificate details", 1.0,
            doc_type="certificate", category="certificates", product_name="16616",
        )
    ]
    types_answer = build_certificate_types_answer(chunks, "en")
    assert types_answer
    assert "Belady Euro" in types_answer
    assert "16616" not in types_answer

    buy_answer = build_certificate_buy_answer(chunks, "en")
    assert buy_answer
    assert "Currency and term" in buy_answer
    assert "Minimum purchase amount" in buy_answer
    assert "Top 5" not in buy_answer
