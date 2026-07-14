"""Extractive answers for certificate catalog / types questions."""

from __future__ import annotations

import re

from ingestion.embedding.vector_store import RetrievedChunk
from services.search_service.intent_classifier import classify_query
from shared.arabic_normalize import normalize_arabic

CERTIFICATE_TYPE_MARKERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"بلادي", re.I), "شهادات بلادي (سنة، 3 سنوات، 5 سنوات) بالعملات الأجنبية"),
    (re.compile(r"العملة المحلية|بالعمله المحلية|بالجنيه", re.I), "شهادات الادخار بالعملة المحلية (جنيه مصري)"),
    (re.compile(r"العملة الأجنبية|بالعمله الاجنبية|بالعملات الأجنبية", re.I), "شهادات بالعملة الأجنبية"),
    (re.compile(r"شهادات الاستثمار|شهادة استثمار", re.I), "شهادات الاستثمار"),
    (re.compile(r"شهادات ادخار|شهادات الادخار", re.I), "شهادات ادخار عامة"),
]

TYPES_QUERY_AR = re.compile(
    r"انواع?\s*(ال)?شهاد|ما\s*هي\s*(ال)?شهاد|شهادات\s*ادخار|انواع?\s*الشهادات\s*البنكيه|قائمة\s*الشهادات",
    re.I,
)
TYPES_QUERY_EN = re.compile(
    r"types?\s+of\s+(saving\s+)?certificates?|certificate\s+types?|saving\s+certificates?",
    re.I,
)


def is_certificate_types_query(query: str, language: str) -> bool:
    intent = classify_query(query, language)
    if intent.intent == "certificate_types":
        return True
    if language == "ar":
        return bool(TYPES_QUERY_AR.search(normalize_arabic(query)))
    return bool(TYPES_QUERY_EN.search(query.lower()))


def _collect_types_from_text(text: str) -> list[str]:
    found: list[str] = []
    for pattern, label in CERTIFICATE_TYPE_MARKERS:
        if pattern.search(text) and label not in found:
            found.append(label)
    return found


def build_certificate_types_answer(
    chunks: list[RetrievedChunk],
    language: str,
) -> str | None:
    if not chunks:
        return None

    certificate_chunks = [c for c in chunks if getattr(c, "doc_type", "") == "certificate" or "شهاد" in c.text]
    source_chunks = certificate_chunks or chunks[:6]
    combined = "\n".join(chunk.text for chunk in source_chunks[:6])
    types = _collect_types_from_text(combined)

    if not types:
        best = source_chunks[0].text.strip()
        if len(best) < 40:
            return None
        if language == "ar":
            return f"وفقاً لمحتوى موقع البنك الأهلي المصري: {best[:650]}"
        return f"According to NBE website content: {best[:650]}"

    if language == "ar":
        bullet = "\n• ".join(types)
        return (
            "وفقاً لمحتوى موقع البنك الأهلي المصري، أنواع شهادات الادخار المتاحة تشمل:\n"
            f"• {bullet}\n"
            "للتفاصيل والاشتراك راجع صفحة الشهادات على موقع البنك أو أقرب فرع."
        )

    bullet = "\n• ".join(types)
    return (
        "According to NBE website content, available saving certificate types include:\n"
        f"• {bullet}\n"
        "See the certificates section on nbe.com.eg or visit a branch for details."
    )
