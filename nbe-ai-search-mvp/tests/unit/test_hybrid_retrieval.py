from services.search_service.hybrid_retriever import reciprocal_rank_fusion
from services.search_service.synonyms import expand_with_synonyms
from ingestion.lexical.bm25_index import tokenize_text


def test_rrf_favors_chunks_in_both_rankings():
    dense = ["a", "b", "c"]
    bm25 = ["b", "a", "d"]
    scores = reciprocal_rank_fusion([dense, bm25], k=60)
    assert scores["b"] > scores["c"]
    assert scores["b"] > scores["d"]
    assert scores["a"] > 0
    assert scores["b"] > 0


def test_arabic_tokenization():
    tokens = tokenize_text("أسعار العملات بما يعادل الجنيه المصري", "ar")
    assert "اسعار" in tokens or "العملات" in tokens or "الجنيه" in tokens


def test_synonym_expansion_ar():
    expanded = expand_with_synonyms("أسعار العملات", "ar")
    assert "سعر الصرف" in expanded or "تحويل العملات" in expanded


def test_synonym_expansion_en():
    expanded = expand_with_synonyms("exchange rate", "en")
    assert "currency converter" in expanded or "banknote rate" in expanded
