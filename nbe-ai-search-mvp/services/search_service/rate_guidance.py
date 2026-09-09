"""Detect rate/yield questions and provide honest guidance without inventing numbers."""

from __future__ import annotations

import re

from shared.arabic_normalize import normalize_arabic
from shared.schemas import Citation, SearchSuggestion


RATE_INTENT_AR = re.compile(
    r"(عائد|فايده|فائدة|فائده|نسبه|نسبة|كام|كم).{0,40}(شهاد|شهادة)"
    r"|(شهاد|شهادة).{0,40}(عائد|فايده|فائدة|فائده|نسبه|نسبة)"
    r"|(شهاد|شهادة).{0,30}(محلي|محلية|جنيه).{0,30}(سنه|سنة)"
    r"|(محلي|محلية).{0,20}(شهاد|عائد|فايد)",
    re.IGNORECASE,
)
EGP_HINT = re.compile(r"(جنيه|مصري|محلي|محلية|سنه|سنة|واحد)", re.IGNORECASE)
LOCAL_RATE_CONTEXT = re.compile(
    r"(جنيه\s*مصري|العملة\s*المحلية|الودائع\s*بالجنيه|شهاد.*محل|محلي.*شهاد|"
    r"localcertificatesid|الشهادة\s*البلاتينية|الشهادة\s*الخماسية|شهادة\s*امان|deposit.*local|local\s*currency)",
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
    """Evidence-based guidance when no applicable numeric rate was retrieved."""
    if not is_rate_query(query, language):
        return None
    if language == "ar":
        return (
            "لم أجد نسبة عائد رقمية قابلة للتحقق ومطابقة للسؤال في المستندات "
            "المسترجعة، لذلك لن أذكر رقماً غير موثق."
        )
    return (
        "No verifiable numeric yield matching the question was found in the "
        "retrieved documents, so no unsupported figure will be provided."
    )


def rate_guidance(query: str, language: str) -> str | None:
    """Backward-compatible detailed guidance (used when answered=False)."""
    return rate_answer(query, language)


def rate_citations(query: str, language: str) -> list[Citation]:
    return []


def rate_suggestions(query: str, language: str) -> list[SearchSuggestion]:
    return []
