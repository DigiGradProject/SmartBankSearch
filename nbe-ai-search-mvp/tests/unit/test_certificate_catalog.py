from ingestion.embedding.vector_store import RetrievedChunk
from services.search_service.certificate_catalog import (
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
