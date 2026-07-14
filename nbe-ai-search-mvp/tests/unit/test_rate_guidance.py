from services.search_service.rate_guidance import (
    context_has_applicable_rate,
    is_local_egp_rate_query,
    is_rate_query,
    rate_answer,
    rate_citations,
    rate_guidance,
    rate_suggestions,
)
from services.search_service.suggestions import build_suggestions


def test_detects_egp_certificate_rate_query():
    q = "ممكن تقولي كم فايده شهادة لمده سنه واحده ب الجنيه المصري"
    assert is_rate_query(q, "ar") is True
    assert is_local_egp_rate_query(q, "ar") is True
    guidance = rate_guidance(q, "ar")
    assert guidance is not None
    assert "غير متوفرة" in guidance or "لا تتوفر" in guidance
    assert "%" not in guidance and "٪" not in guidance


def test_detects_expanded_local_certificate_query():
    q = "شهادات الادخار بالعملة المحلية عائد شهادة سنة جنيه"
    assert is_rate_query(q, "ar") is True
    assert is_local_egp_rate_query(q, "ar") is True


def test_foreign_percent_not_applicable_for_egp_query():
    q = "كم فايده شهادة سنة بالجنيه"
    foreign_context = "سعر العائد 4.75% دولار - 3 سنوات"
    assert context_has_applicable_rate(q, "ar", foreign_context) is False


def test_local_context_with_percent_is_applicable():
    q = "كم فايده شهادة سنة بالجنيه"
    local_context = "شهادات الادخار بالعملة المحلية عائد 14% لمدة سنة"
    assert context_has_applicable_rate(q, "ar", local_context) is True


def test_rate_answer_includes_actionable_guidance():
    q = "شهادات الادخار بالعملة المحلية عائد شهادة سنة جنيه"
    answer = rate_answer(q, "ar")
    assert answer
    assert "المحتوى المفهرس" in answer
    assert "فرع" in answer or "موقع" in answer


def test_rate_citations_include_local_certificates_page():
    q = "كم عائد شهادة سنة بالجنيه"
    citations = rate_citations(q, "ar")
    assert any("CertificatesID" in item.url for item in citations)


def test_rate_suggestions_include_local_certificates_page():
    q = "كم عائد شهادة سنة بالجنيه"
    suggestions = rate_suggestions(q, "ar")
    assert any("CertificatesID" in (item.url or "") for item in suggestions)
    assert any(item.reason == "official_page" for item in suggestions)


def test_build_suggestions_prioritizes_official_pages():
    suggestions = build_suggestions("كم فايده شهادة سنة جنيه", "ar", [])
    assert suggestions
    assert suggestions[0].reason == "official_page"
    assert suggestions[0].url
