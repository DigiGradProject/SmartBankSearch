from services.search_service.autocomplete import build_autocomplete


def test_autocomplete_popular_when_empty():
    suggestions = build_autocomplete("", "ar", limit=5)
    assert len(suggestions) == 5
    assert all(item.reason == "popular" for item in suggestions)


def test_autocomplete_prefix_match_arabic():
    suggestions = build_autocomplete("شهاد", "ar", limit=5)
    assert len(suggestions) >= 1
    assert any("شهاد" in item.label for item in suggestions)


def test_autocomplete_colloquial_arabic():
    suggestions = build_autocomplete("عايز", "ar", limit=5)
    assert len(suggestions) >= 1
    assert any("شهاد" in item.query or "حساب" in item.query for item in suggestions)
