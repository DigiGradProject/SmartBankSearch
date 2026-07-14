"""Rule-based banking intent classifier for query-time metadata filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass

from shared.arabic_normalize import normalize_arabic

# doc_types allowed per intent (Chroma $in filter)
INTENT_DOC_TYPES: dict[str, tuple[str, ...]] = {
    "exchange_rate": ("exchange_rate", "currency_converter"),
    "certificate_rate": ("certificate", "certificate_rate"),
    "certificate_types": ("certificate",),
    "certificate_buy": ("certificate",),
    "personal_loan": ("loan",),
    "credit_card": ("credit_card",),
    "debit_card": ("debit_card",),
    "card_types": ("credit_card", "debit_card"),
    "branch_locator": ("branch",),
    "atm_locator": ("branch", "atm"),
    "account_open": ("account",),
    "wallet": ("wallet",),
    "digital_banking": ("digital_banking", "wallet"),
    "offers": ("offer", "product", "general"),
    "news": ("news", "general"),
    "reports": ("report", "general"),
    "corporate": ("corporate", "product", "general"),
    "sme": ("sme", "product", "general"),
    "faq": ("faq", "general", "product"),
    "general_faq": ("faq", "general", "product"),
}


@dataclass(frozen=True)
class QueryIntent:
    intent: str
    category: str
    confidence: float
    allowed_doc_types: tuple[str, ...]
    expand_ar: str = ""
    expand_en: str = ""
    negative_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntentRule:
    intent: str
    category: str
    patterns_ar: tuple[re.Pattern[str], ...]
    patterns_en: tuple[re.Pattern[str], ...]
    negative_terms: tuple[str, ...] = ()
    expand_ar: str = ""
    expand_en: str = ""
    confidence: float = 0.95


def _ar(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _en(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


INTENT_RULES: list[IntentRule] = [
    IntentRule(
        intent="exchange_rate",
        category="exchange_rates",
        patterns_ar=(
            _ar(r"اسعار?\s*(ال)?عملات"),
            _ar(r"سعر\s*الصرف"),
            _ar(r"تحويل\s*العملات"),
            _ar(r"محول\s*العملات"),
            _ar(r"دولار\s*بكام"),
            _ar(r"بكام\s*الدولار"),
            _ar(r"اسعار?\s*الصرف"),
            _ar(r"بما\s*يعادل\s*الجنيه"),
        ),
        patterns_en=(
            _en(r"exchange\s*rate"),
            _en(r"currency\s*convert"),
            _en(r"usd.*egp"),
            _en(r"banknote\s*rate"),
        ),
        negative_terms=("شهاد", "عائد", "فايد", "فائد", "certificate", "yield", "belady", "بلادي"),
        expand_ar="سعر الصرف تحويل العملات محول العملات ExchangeRatesAndCurrencyConverter",
        expand_en="exchange rate currency converter banknote transfer rate",
    ),
    IntentRule(
        intent="certificate_rate",
        category="certificates",
        patterns_ar=(
            _ar(r"(عائد|فايد|فائد|نسبه|نسبة).{0,30}شهاد"),
            _ar(r"شهاد.{0,30}(عائد|فايد|فائد|نسبه|نسبة)"),
            _ar(r"اسعار?\s*الشهاد"),
        ),
        patterns_en=(
            _en(r"(interest|yield|rate).{0,30}certificate"),
            _en(r"certificate.{0,30}(interest|yield|rate)"),
        ),
        negative_terms=("سعر الصرف", "تحويل العملات", "exchange rate", "currency convert"),
        expand_ar="شهادات الادخار عائد شهادة CertificatesRates",
        expand_en="certificate yield interest rate",
        confidence=0.92,
    ),
    IntentRule(
        intent="certificate_types",
        category="certificates",
        patterns_ar=(
            _ar(r"انواع?\s*(ال)?شهاد"),
            _ar(r"ما\s*هي\s*(ال)?شهاد"),
            _ar(r"انواع?\s*الشهادات\s*البنكيه"),
            _ar(r"قائمة\s*الشهادات"),
            _ar(r"شهادات\s*ادخار"),
        ),
        patterns_en=(
            _en(r"types?\s+of\s+(saving\s+)?certificates?"),
            _en(r"certificate\s+types?"),
            _en(r"saving\s+certificates?"),
        ),
        expand_ar="شهادات الادخار بلادي محلية اجنبية استثمار CertificatesID",
        expand_en="saving certificates belady local foreign investment",
        confidence=0.93,
    ),
    IntentRule(
        intent="certificate_buy",
        category="certificates",
        patterns_ar=(
            _ar(r"(شراء|اشتري|اشترى).{0,20}شهاد"),
            _ar(r"عا[يو]ز.{0,30}(شراء|اشتري).{0,20}شهاد"),
        ),
        patterns_en=(_en(r"buy.{0,20}certificate"),),
        expand_ar="شراء شهادة شهادات ادخار CertificatesID",
        expand_en="buy certificate savings",
    ),
    IntentRule(
        intent="personal_loan",
        category="loans",
        patterns_ar=(_ar(r"قرض\s*شخصي"), _ar(r"قروض\s*شخصية"), _ar(r"تمويل\s*شخصي")),
        patterns_en=(_en(r"personal\s*loan"),),
        expand_ar="قرض شخصي تمويل Loans",
        expand_en="personal loan financing",
    ),
    IntentRule(
        intent="card_types",
        category="cards",
        patterns_ar=(
            _ar(r"انواع?\s*(ال)?بطاق"),
            _ar(r"انواع?\s*البطاقات\s*البنكيه"),
            _ar(r"ما\s*هي\s*(ال)?بطاق"),
            _ar(r"قائمة\s*البطاقات"),
            _ar(r"انواع?\s*كروت"),
        ),
        patterns_en=(
            _en(r"types?\s+of\s+(bank\s+)?cards?"),
            _en(r"card\s+types?"),
        ),
        expand_ar="بطاقات ائتمان خصم مباشر مدفوعة مقدما CreditCards DepitCards PrepaidCards",
        expand_en="credit debit prepaid cards NBE",
        confidence=0.93,
    ),
    IntentRule(
        intent="credit_card",
        category="cards",
        patterns_ar=(_ar(r"بطاق.{0,6}ائتمان"), _ar(r"كريدت\s*كارد")),
        patterns_en=(_en(r"credit\s*card"),),
        expand_ar="بطاقات ائتمان CreditCards",
        expand_en="credit card NBE",
    ),
    IntentRule(
        intent="branch_locator",
        category="branches",
        patterns_ar=(_ar(r"الفروع"), _ar(r"اقرب\s*فرع"), _ar(r"موقع\s*الفرع")),
        patterns_en=(_en(r"branch\s*locator"), _en(r"find\s*branch")),
        expand_ar="الفروع ATMBranch",
        expand_en="branch locator ATMBranch",
    ),
    IntentRule(
        intent="atm_locator",
        category="branches",
        patterns_ar=(_ar(r"الصراف\s*الالي"), _ar(r"ماكينات\s*الصرف"), _ar(r"\batm\b")),
        patterns_en=(_en(r"\batm\b"), _en(r"cash\s*machine")),
        expand_ar="ماكينات الصرف الآلي ATMBranch",
        expand_en="ATM locator",
    ),
    IntentRule(
        intent="account_open",
        category="accounts",
        patterns_ar=(_ar(r"فتح\s*حساب"), _ar(r"افتح\s*حساب")),
        patterns_en=(_en(r"open\s*(a\s*)?account"),),
        expand_ar="فتح حساب بنكي CurrentAccountsID الأوراق المطلوبة",
        expand_en="open bank account current account required documents",
    ),
    IntentRule(
        intent="wallet",
        category="wallet",
        patterns_ar=(_ar(r"فون\s*كاش"),),
        patterns_en=(_en(r"phone\s*cash"),),
        expand_ar="خدمة فون كاش",
        expand_en="phone cash wallet",
    ),
    IntentRule(
        intent="digital_banking",
        category="digital_banking",
        patterns_ar=(_ar(r"اهلي\s*نت|الأهلي\s*نت|موبايل\s*بانك|انترنت\s*بنكي|خدمه\s*رقمي"),),
        patterns_en=(_en(r"internet\s*banking|mobile\s*banking|digital\s*banking"),),
        expand_ar="الأهلي نت الموبايل بانكنج خدمات رقمية",
        expand_en="internet banking mobile banking digital services",
        confidence=0.9,
    ),
    IntentRule(
        intent="offers",
        category="offers",
        patterns_ar=(_ar(r"عروض|عرض\s*خاص|خصومات"),),
        patterns_en=(_en(r"\boffers?\b|promotions?|discounts?"),),
        expand_ar="عروض البنك الأهلي Offers",
        expand_en="NBE offers promotions",
        confidence=0.9,
    ),
    IntentRule(
        intent="news",
        category="news",
        patterns_ar=(_ar(r"اخبار|أخبار|بيان\s*صحفي"),),
        patterns_en=(_en(r"\bnews\b|press\s*release"),),
        expand_ar="أخبار البنك الأهلي",
        expand_en="NBE news",
        confidence=0.88,
    ),
    IntentRule(
        intent="reports",
        category="reports",
        patterns_ar=(_ar(r"تقارير|تقرير\s*سنوي|القوائم\s*المالية"),),
        patterns_en=(_en(r"annual\s*report|financial\s*statements?|reports?"),),
        expand_ar="تقارير البنك القوائم المالية",
        expand_en="NBE annual report financial statements",
        confidence=0.88,
    ),
    IntentRule(
        intent="corporate",
        category="corporate",
        patterns_ar=(_ar(r"خدمات\s*الشركات|قطاع\s*الشركات|corporate"),),
        patterns_en=(_en(r"corporate\s*banking|business\s*banking"),),
        expand_ar="خدمات الشركات Corporate Banking",
        expand_en="corporate banking NBE",
        confidence=0.9,
    ),
    IntentRule(
        intent="sme",
        category="sme",
        patterns_ar=(_ar(r"مشروعات?\s*(ال)?صغير|المتوسط[ةه]|SME|منشآت"),),
        patterns_en=(_en(r"\bsme\b|small\s*(and|&)\s*medium"),),
        expand_ar="المشروعات الصغيرة والمتوسطة SME",
        expand_en="SME small medium enterprises",
        confidence=0.9,
    ),
    IntentRule(
        intent="faq",
        category="general",
        patterns_ar=(_ar(r"اسئل[ةه]\s*شائع|الأسئلة\s*الشائعة|الاسئله\s*الشائعه|FAQ"),),
        patterns_en=(_en(r"\bfaq\b|frequently\s*asked"),),
        expand_ar="أسئلة شائعة FAQs",
        expand_en="frequently asked questions FAQ",
        confidence=0.9,
    ),
]


def _has_negative(query: str, terms: tuple[str, ...]) -> bool:
    if not terms:
        return False
    normalized = normalize_arabic(query) if any(ord(c) > 127 for c in query) else query.lower()
    for term in terms:
        term_norm = normalize_arabic(term) if any(ord(c) > 127 for c in term) else term.lower()
        if term_norm in normalized:
            return True
    return False


def classify_query(query: str, language: str) -> QueryIntent:
    normalized = normalize_arabic(query) if language == "ar" else query.lower().strip()

    for rule in INTENT_RULES:
        patterns = rule.patterns_ar if language == "ar" else rule.patterns_en
        if not patterns and language == "en":
            patterns = rule.patterns_ar  # fallback for mixed queries
        matched = any(p.search(normalized) for p in patterns)
        if not matched:
            continue
        if _has_negative(query, rule.negative_terms):
            continue
        allowed = INTENT_DOC_TYPES.get(rule.intent, ("general",))
        return QueryIntent(
            intent=rule.intent,
            category=rule.category,
            confidence=rule.confidence,
            allowed_doc_types=allowed,
            expand_ar=rule.expand_ar,
            expand_en=rule.expand_en,
            negative_terms=rule.negative_terms,
        )

    return QueryIntent(
        intent="general_faq",
        category="general",
        confidence=0.0,
        allowed_doc_types=INTENT_DOC_TYPES["general_faq"],
    )


def should_apply_metadata_filter(intent: QueryIntent, threshold: float) -> bool:
    return intent.confidence >= threshold and intent.intent != "general_faq"
