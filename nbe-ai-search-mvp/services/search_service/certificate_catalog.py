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
        # Some scraped pages carry the numeric ProductID in product_name.
        # Prefer the human-readable page title when that happens.
        raw_name = (chunk.product_name or "").strip()
        if not raw_name or re.fullmatch(r"\d+", raw_name):
            raw_name = chunk.title
        candidate = _clean_source_title(raw_name)
        if not candidate or candidate.lower() in {"شهادات", "certificates"}:
            continue
        if re.fullmatch(r"\d+", candidate):
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

    has_certificate_source = any(
        getattr(chunk, "category", "") == "certificates"
        or re.search(r"شهاد|certificate|belady", f"{chunk.title} {chunk.text}", re.I)
        for chunk in chunks
    )
    if not has_certificate_source:
        return None

    if language == "ar":
        return (
            "لشراء شهادة استثمار أو ادخار من البنك الأهلي المصري، افتح صفحة "
            "شهادات الادخار الرسمية أولاً، ثم اختر الشهادة المناسبة بعد مراجعة:\n"
            "• العملة ومدة الشهادة.\n"
            "• قيمة العائد ودورية صرفه.\n"
            "• الحد الأدنى للشراء وشروط الاسترداد والاقتراض بضمان الشهادة.\n"
            "بعد اختيار الشهادة، افتح صفحة المنتج لمعرفة وسائل وخطوات الشراء المتاحة. "
            "إذا حددت العملة والمبلغ والمدة المطلوبة، يمكنني مساعدتك في المقارنة."
        )

    return (
        "To buy an NBE investment or savings certificate, start with the official "
        "Saving Certificates page, then choose a suitable certificate after reviewing:\n"
        "• Currency and term.\n"
        "• Yield and payout frequency.\n"
        "• Minimum purchase amount, redemption rules, and borrowing terms.\n"
        "After choosing a certificate, open its product page for the available purchase methods "
        "and steps. If you share your preferred currency, amount, and term, I can help compare "
        "the relevant options."
    )
