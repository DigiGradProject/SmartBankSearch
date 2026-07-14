from services.search_service.keyword_rank import extract_query_terms, keyword_overlap_score


def test_arabic_term_extraction():
    terms = extract_query_terms("شهادات بلادي سنة بالدولار", "ar")
    assert "بلادى" in terms
    assert "سنه" in terms
    assert "بالدولار" in terms


def test_keyword_overlap():
    score = keyword_overlap_score("شهادات بلادي سنه بلادى بالدولار امريكي", ["بلادى", "دولار", "سنه"], "ar")
    assert score >= 0.66
