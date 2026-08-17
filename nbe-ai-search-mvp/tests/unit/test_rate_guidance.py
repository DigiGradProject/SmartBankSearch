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
    assert "لن أذكر رقماً غير موثق" in guidance
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


def test_rate_answer_reports_only_missing_retrieved_evidence():
    q = "شهادات الادخار بالعملة المحلية عائد شهادة سنة جنيه"
    answer = rate_answer(q, "ar")
    assert answer
    assert "المستندات المسترجعة" in answer
    assert "ديناميك" not in answer


def test_rate_guidance_does_not_inject_citations():
    q = "كم عائد شهادة سنة بالجنيه"
    citations = rate_citations(q, "ar")
    assert citations == []


def test_rate_guidance_does_not_inject_suggestions():
    q = "كم عائد شهادة سنة بالجنيه"
    suggestions = rate_suggestions(q, "ar")
    assert suggestions == []


def test_build_suggestions_uses_queries_not_hard_coded_urls():
    suggestions = build_suggestions("كم فايده شهادة سنة جنيه", "ar", [])
    assert suggestions
    assert all(item.reason != "official_page" for item in suggestions)
    assert all(item.url is None for item in suggestions)
