"""Enterprise language detection — Lingua primary, heuristic fallback.

Decision: lingua-language-detector is the best short-query AR/EN detector for
on-prem banking search. See docs/enterprise-model-decisions.md.
"""

from __future__ import annotations

import re
from functools import lru_cache

from shared.arabic_normalize import normalize_text
from shared.logging import get_logger

logger = get_logger(__name__)

ARABIC_CHARS = re.compile(r"[\u0600-\u06FF]")

_LINGUA_AVAILABLE = False
try:
    from lingua import Language, LanguageDetectorBuilder

    _LINGUA_AVAILABLE = True
except Exception:  # noqa: BLE001
    Language = None  # type: ignore[misc, assignment]
    LanguageDetectorBuilder = None  # type: ignore[misc, assignment]


@lru_cache(maxsize=1)
def _detector():
    if not _LINGUA_AVAILABLE:
        return None
    return (
        LanguageDetectorBuilder.from_languages(Language.ARABIC, Language.ENGLISH)
        .with_preloaded_language_models()
        .build()
    )


def _heuristic_language(text: str) -> str:
    arabic_count = len(ARABIC_CHARS.findall(text))
    return "ar" if arabic_count >= max(3, len(text) // 8) else "en"


def detect_language(text: str, requested: str = "auto") -> str:
    """Return 'ar' or 'en'. Honors explicit UI language when provided."""
    if requested in {"ar", "en"}:
        return requested

    detector = _detector()
    if detector is not None and text.strip():
        try:
            language = detector.detect_language_of(text)
            if language is not None and language.name == "ARABIC":
                return "ar"
            if language is not None and language.name == "ENGLISH":
                return "en"
        except Exception as exc:  # noqa: BLE001
            logger.warning("lingua_detect_failed", error=str(exc))

    return _heuristic_language(text)


def prepare_query(query: str, language: str) -> str:
    return normalize_text(query.strip(), language)
