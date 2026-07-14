from services.rag.entity_extractor import extract_entities
from services.rag.language import detect_language
from services.rag.query_rewrite import rewrite_query


def test_language_detects_arabic():
    assert detect_language("عايز افتح حساب", "auto") == "ar"


def test_language_detects_english():
    assert detect_language("open a current account", "auto") == "en"


def test_language_honors_explicit():
    assert detect_language("hello", "ar") == "ar"


def test_entity_extraction_currency_and_tenor():
    result = extract_entities("شهادة سنة بالدولار", "ar")
    assert result.currencies
    assert result.tenors or result.product_mentions


def test_query_rewrite_is_deterministic():
    first = rewrite_query("عايز اعرف فايده شهادة", "ar")
    second = rewrite_query("عايز اعرف فايده شهادة", "ar")
    assert first.rewritten == second.rewritten
    assert first.used_llm is False
    assert "عائد" in first.rewritten or "شهادة" in first.rewritten
