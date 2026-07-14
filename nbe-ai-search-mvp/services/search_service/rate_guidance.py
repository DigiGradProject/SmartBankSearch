"""Detect rate/yield questions and provide honest guidance without inventing numbers."""

from __future__ import annotations

import re

from shared.arabic_normalize import normalize_arabic
from shared.schemas import Citation, SearchSuggestion

LOCAL_CERTIFICATES_URL = (
    'https://www.nbe.com.eg/NBE/E/#/AR/ProductCategory?inParams={"CategoryID":"CertificatesID"}'
)
LOCAL_DEPOSIT_RATES_URL = "https://www.nbe.com.eg/NBE/E/#/AR/Depositrateslocalcurrency"
FOREIGN_RATES_URL = "https://www.nbe.com.eg/NBE/E/#/AR/CertificatesRatesForeignCurrency"

RATE_INTENT_AR = re.compile(
    r"(عائد|فايده|فائدة|فائده|نسبه|نسبة|كام|كم).{0,40}(شهاد|شهادة)"
    r"|(شهاد|شهادة).{0,40}(عائد|فايده|فائدة|فائده|نسبه|نسبة)"
    r"|(شهاد|شهادة).{0,30}(محلي|محلية|جنيه).{0,30}(سنه|سنة)"
    r"|(محلي|محلية).{0,20}(شهاد|عائد|فايد)",
    re.IGNORECASE,
)
EGP_HINT = re.compile(r"(جنيه|مصري|محلي|محلية|سنه|سنة|واحد)", re.IGNORECASE)
LOCAL_RATE_CONTEXT = re.compile(
    r"(جنيه\s*مصري|العملة\s*المحلية|الودائع\s*بالجنيه|شهاد.*محل|محلي.*شهاد|deposit.*local|local\s*currency)",
    re.IGNORECASE,
)
RATE_IN_CONTEXT = re.compile(r"\d+(?:[.,]\d+)?\s*[٪%]")
RATE_INTENT_EN = re.compile(
    r"(interest|yield|rate).{0,40}(certificate|deposit)|(certificate|deposit).{0,40}(interest|yield|rate)",
    re.IGNORECASE,
)


def is_rate_query(query: str, language: str) -> bool:
    if language == "ar":
        return bool(RATE_INTENT_AR.search(normalize_arabic(query)))
    return bool(RATE_INTENT_EN.search(query))


def is_local_egp_rate_query(query: str, language: str) -> bool:
    return language == "ar" and is_rate_query(query, language) and bool(EGP_HINT.search(normalize_arabic(query)))


def context_has_applicable_rate(query: str, language: str, context_text: str) -> bool:
    """True only when indexed context contains a rate that matches the query scope."""
    if not RATE_IN_CONTEXT.search(context_text):
        return False
    if is_local_egp_rate_query(query, language):
        return bool(LOCAL_RATE_CONTEXT.search(normalize_arabic(context_text[:4000])))
    return True


def rate_answer(query: str, language: str) -> str | None:
    """User-facing answer when numeric yield is not in the index."""
    if not is_rate_query(query, language):
        return None
    if language == "ar":
        if is_local_egp_rate_query(query, language):
            return (
                "لا تتوفر نسبة العائد الرقمية لشهادة الادخار بالجنيه المصري في المحتوى المفهرس حالياً، "
                "لأن صفحة الأسعار على موقع البنك تُحمَّل ديناميكياً ولم تُسجَّل في عملية الأرشفة. "
                "للاطلاع على العائد المعلن حالياً، افتح صفحة شهادات الادخار بالعملة المحلية على موقع "
                "البنك الأهلي المصري أو تواصل مع أقرب فرع."
            )
        return (
            "نسبة العائد الرقمية غير متوفرة في المحتوى المفهرس حالياً. "
            "راجع صفحات أسعار الشهادات على موقع البنك الأهلي المصري أو أقرب فرع."
        )
    return (
        "The exact yield figure is not available in the indexed content because NBE rate pages "
        "are dynamically rendered. Please open the local-currency certificates page on nbe.com.eg "
        "or visit a branch for the latest announced rate."
    )


def rate_guidance(query: str, language: str) -> str | None:
    """Backward-compatible detailed guidance (used when answered=False)."""
    return rate_answer(query, language)


def rate_citations(query: str, language: str) -> list[Citation]:
    if not is_rate_query(query, language):
        return []
    citations: list[Citation] = []
    for item in rate_suggestions(query, language):
        if item.url:
            citations.append(Citation(title=item.label, url=item.url))
    return citations


def rate_suggestions(query: str, language: str) -> list[SearchSuggestion]:
    if not is_rate_query(query, language):
        return []
    if language == "ar":
        items = [
            SearchSuggestion(
                query="شهادات الادخار بالعملة المحلية",
                label="شهادات الادخار بالعملة المحلية",
                url=LOCAL_CERTIFICATES_URL,
                reason="official_page",
                score=1.0,
            ),
        ]
        if is_local_egp_rate_query(query, language):
            items.append(
                SearchSuggestion(
                    query="أسعار العائد على الودائع بالجنيه المصري",
                    label="أسعار العائد على الودائع بالجنيه المصري",
                    url=LOCAL_DEPOSIT_RATES_URL,
                    reason="official_page",
                    score=0.95,
                )
            )
        items.extend(
            [
                SearchSuggestion(
                    query="أسعار الشهادات بالعملة الأجنبية",
                    label="أسعار الشهادات بالعملة الأجنبية",
                    url=FOREIGN_RATES_URL,
                    reason="official_page",
                    score=0.9,
                ),
                SearchSuggestion(
                    query="شراء شهادة ادخار",
                    label="شراء شهادة ادخار",
                    reason="topic_match",
                    score=0.8,
                ),
            ]
        )
        return items
    return [
        SearchSuggestion(
            query="local currency saving certificates NBE",
            label="Local currency saving certificates",
            url='https://www.nbe.com.eg/NBE/E/#/EN/ProductCategory?inParams={"CategoryID":"CertificatesID"}',
            reason="official_page",
            score=1.0,
        ),
        SearchSuggestion(
            query="certificates foreign currency rates",
            label="Foreign currency certificate rates",
            url="https://www.nbe.com.eg/NBE/E/#/EN/CertificatesRatesForeignCurrency",
            reason="official_page",
            score=0.9,
        ),
    ]
