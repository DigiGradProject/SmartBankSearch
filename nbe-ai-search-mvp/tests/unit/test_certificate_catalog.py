from services.search_service.certificate_catalog import (
    build_certificate_types_answer,
    is_certificate_types_query,
)
from services.search_service.intent_classifier import classify_query
from services.search_service.search import SearchService


def test_certificate_types_intent():
    q = "انواع الشهادات البنكيه"
    assert classify_query(q, "ar").intent == "certificate_types"
    assert is_certificate_types_query(q, "ar") is True


def test_build_certificate_types_answer_from_stubs():
    retrieval = SearchService().retrieve("انواع الشهادات البنكيه", "ar")
    answer = build_certificate_types_answer(retrieval.chunks, "ar")
    assert answer
    assert "شهادات" in answer
    assert "بلادي" in answer or "محلية" in answer or "ادخار" in answer
