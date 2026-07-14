import re

from shared.arabic_normalize import normalize_text

ARABIC_CHARS = re.compile(r"[\u0600-\u06FF]")


def detect_language(text: str, requested: str = "auto") -> str:
    if requested in {"ar", "en"}:
        return requested
    arabic_count = len(ARABIC_CHARS.findall(text))
    return "ar" if arabic_count >= max(3, len(text) // 8) else "en"


def prepare_query(query: str, language: str) -> str:
    return normalize_text(query.strip(), language)
