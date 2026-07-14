"""Banking domain synonym expansion for Arabic and English queries."""

from __future__ import annotations

import json
from functools import lru_cache

from shared.arabic_normalize import normalize_arabic
from shared.config import settings


@lru_cache(maxsize=2)
def _load_synonyms(language: str) -> dict[str, list[str]]:
    filename = "banking_synonyms.ar.json" if language == "ar" else "banking_synonyms.en.json"
    path = settings.project_root / "data" / "vocabulary" / filename
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if language == "ar":
        return {normalize_arabic(key): [normalize_arabic(v) for v in values] for key, values in payload.items()}
    return {key.lower(): [v.lower() for v in values] for key, values in payload.items()}


def expand_with_synonyms(query: str, language: str) -> str:
    synonyms = _load_synonyms(language)
    if not synonyms:
        return query

    normalized = normalize_arabic(query) if language == "ar" else query.lower()
    extra: list[str] = []

    # Longest-key-first so "قرض شخصي" matches before "قرض".
    for key in sorted(synonyms, key=len, reverse=True):
        if key in normalized:
            for value in synonyms[key]:
                if value not in extra and value not in normalized:
                    extra.append(value)

    if not extra:
        return query

    return f"{query} {' '.join(extra)}"
