"""Extractive answers for certificate catalog / types questions."""

from __future__ import annotations

import re

from ingestion.embedding.vector_store import RetrievedChunk
from services.search_service.intent_classifier import classify_query
from shared.arabic_normalize import normalize_arabic

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


def _clean_source_title(title: str) -> str:
    return re.sub(
        r"^(?:البنك\s+الأهلى\s+(?:المصرى|المصري)|National Bank of Egypt)\s*-\s*",
        "",
        title.strip(),
        flags=re.I,
    ).strip()


def _certificate_names(chunks: list[RetrievedChunk], language: str) -> list[str]:
    names: list[str] = []
    for chunk in chunks:
        if getattr(chunk, "category", "") != "certificates" and not re.search(
            r"شهاد|certificate|belady", f"{chunk.title} {chunk.text}", re.I
        ):
            continue
        candidate = _clean_source_title(chunk.product_name or chunk.title)
        if not candidate or candidate.lower() in {"شهادات", "certificates"}:
            continue
        if language == "ar" and not re.search(r"[\u0600-\u06FF]", candidate):
            continue
        if language == "en" and re.search(r"[\u0600-\u06FF]", candidate):
            continue
        if candidate not in names:
            names.append(candidate)
    return names[:10]


def build_certificate_types_answer(
    chunks: list[RetrievedChunk],
    language: str,
) -> str | None:
    if not chunks:
        return None

    names = _certificate_names(chunks, language)
    if not names:
        return None

    bullet = "\n• ".join(names)
    if language == "ar":
        return (
            "وفقاً للمستندات المسترجعة، صفحات شهادات الادخار المطابقة تشمل:\n"
            f"• {bullet}\n"
            "التفاصيل الواردة في الإجابة مقتصرة على هذه المصادر."
        )

    return (
        "According to the retrieved documents, matching certificate pages include:\n"
        f"• {bullet}\n"
        "The answer is limited to these retrieved sources."
    )


def build_certificate_buy_answer(
    chunks: list[RetrievedChunk],
    language: str,
) -> str | None:
    """Offer grounded certificate choices without making financial advice."""
    if not chunks:
        return None

    names = _certificate_names(chunks, language)
    if not names:
        return None

    bullet = "\n• ".join(names[:5])
    if language == "ar":
        return (
            "لو هدفك استثمار المبلغ في شهادة من البنك الأهلي المصري، "
            "فالنتائج المتاحة تشمل:\n"
            f"• {bullet}\n"
            "راجع صفحة كل شهادة وقارن مدة الشهادة والعائد ودورية صرفه، "
            "ثم افتح صفحة المنتج المناسبة لمعرفة التفاصيل وخطوات الشراء."
        )

    return (
        "If you want to invest the amount in an NBE certificate, the available results include:\n"
        f"• {bullet}\n"
        "Review each certificate page and compare its term, yield, and payout frequency, "
        "then open the matching product page for details and purchase steps."
    )
