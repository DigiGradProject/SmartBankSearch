from shared.arabic_normalize import normalize_arabic, normalize_text


def test_normalize_arabic_variants():
    assert normalize_arabic("أهلاً بك") == "اهلا بك"
    assert normalize_arabic("مدرسة") == "مدرسه"


def test_normalize_english_passthrough():
    assert normalize_text("  Hello   World  ", "en") == "Hello World"
