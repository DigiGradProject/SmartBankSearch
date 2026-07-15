"""Critical banking rules — always evaluated before semantic/regex intent.

These are deterministic safety routes (contact numbers, password reset, etc.)
that must never be left to probabilistic classifiers.
"""

from __future__ import annotations

import re

from services.search_service.intent_classifier import INTENT_DOC_TYPES, QueryIntent
from shared.arabic_normalize import normalize_arabic

_CRITICAL_RULES: list[tuple[str, str, str, tuple[re.Pattern[str], ...], float]] = [
    # (intent, category, expand_ar, patterns, confidence)
    (
        "customer_service",
        "contact",
        "خدمة العملاء 19623 اتصل بنا",
        (
            re.compile(r"19623", re.I),
            re.compile(r"خدم[ةه]\s*العملاء", re.I),
            re.compile(r"رقم\s*الخدمة", re.I),
            re.compile(r"اتصل\s*بنا", re.I),
            re.compile(r"call\s*center", re.I),
            re.compile(r"customer\s*service", re.I),
            re.compile(r"hotline", re.I),
        ),
        0.99,
    ),
    (
        "password_reset",
        "digital_banking",
        "نسيت كلمة المرور استعادة كلمة السر",
        (
            re.compile(r"نسيت\s*(ال)?باسورد", re.I),
            re.compile(r"نسيت\s*كلمة\s*المرور", re.I),
            re.compile(r"استعاد[ةه]\s*كلمة\s*السر", re.I),
            re.compile(r"forgot\s*password", re.I),
            re.compile(r"reset\s*password", re.I),
            re.compile(r"password\s*recovery", re.I),
        ),
        0.99,
    ),
]


def classify_critical(query: str, language: str) -> QueryIntent | None:
    """Return a critical intent if matched; otherwise None."""
    normalized = normalize_arabic(query) if language == "ar" else query.lower().strip()
    for intent, category, expand_ar, patterns, confidence in _CRITICAL_RULES:
        if any(p.search(normalized) for p in patterns):
            return QueryIntent(
                intent=intent,
                category=category,
                confidence=confidence,
                allowed_doc_types=("faq", "general", "digital_banking"),
                expand_ar=expand_ar,
                expand_en="customer service contact password reset",
                source="critical",
            )
    return None
