import re

from shared.arabic_normalize import normalize_arabic

# Egyptian colloquial -> formal banking terms used on NBE site.
WORD_MAP = {
    "عايز": "اريد",
    "عاوز": "اريد",
    "محتاج": "اريد",
    "اشتري": "شراء",
    "أشتري": "شراء",
    "اشترى": "شراء",
    "شهاده": "شهادة",
    "شهادات": "شهادات",
    "حساب": "حساب",
    "افتح": "فتح",
    "فتح": "فتح حساب",
    "قرض": "قرض",
    "بطاقه": "بطاقة",
    "كارت": "بطاقة",
    "فلوس": "نقود",
    "فايده": "عائد",
    "فائدة": "عائد",
    "انواع": "شهادات",
    "البنكيه": "بنكي",
    "بنكية": "بنكي",
}

INTENT_EXPANSIONS = [
    (
        re.compile(r"(?i)انواع?\s*(ال)?شهاد|ما\s*هي\s*(ال)?شهاد|قائمة\s*الشهادات"),
        "شهادات الادخار بلادي محلية اجنبية استثمار شهادات الاستثمار",
    ),
    (
        re.compile(r"(?i)اسعار?\s*(ال)?عملات|سعر\s*الصرف|تحويل\s*العملات|محول\s*العملات"),
        "سعر الصرف تحويل العملات محول العملات ExchangeRatesAndCurrencyConverter",
    ),
    (re.compile(r"(?i)عا[يو]ز.*(اشتري|شراء).*(شهاد|شهادة)"), "شراء شهادة شهادات ادخار شهادات بلادي"),
    (re.compile(r"(?i)عا[يو]ز.*(اشتري|شراء)"), "شراء منتج خدمة بنكية"),
    (re.compile(r"(?i)عا[يو]ز.*فتح.*حساب"), "فتح حساب بنكي افتح حسابك"),
    (re.compile(r"(?i)عا[يو]ز.*قرض"), "قرض شخصي تمويل"),
    (re.compile(r"(?i)عا[يو]ز.*بطاق"), "بطاقة ائتمان بطاقات"),
    (
        re.compile(r"(?i)(كم|كام|نسبه|نسبة|عائد|فايده|فائدة).{0,30}(شهاد|شهادة).{0,30}(سنه|سنة|جنيه|مصري)"),
        "شهادات الادخار بالعملة المحلية عائد شهادة سنة جنيه",
    ),
    (
        re.compile(r"(?i)(شهاد|شهادة).{0,30}(سنه|سنة).{0,30}(جنيه|مصري|عائد|فايده)"),
        "شهادات الادخار بالعملة المحلية",
    ),
]


def expand_query_intent(query: str, language: str) -> str | None:
    if language != "ar":
        return None
    normalized = normalize_arabic(query)
    for pattern, expansion in INTENT_EXPANSIONS:
        if pattern.search(normalized):
            return f"{query} {expansion}"
    return None


def expand_query(query: str, language: str) -> str:
    if language != "ar":
        return query

    normalized = normalize_arabic(query)
    extra_terms: list[str] = []

    for word in normalized.split():
        mapped = WORD_MAP.get(word)
        if mapped and mapped not in extra_terms:
            extra_terms.append(mapped)

    for pattern, expansion in INTENT_EXPANSIONS:
        if pattern.search(normalized):
            extra_terms.append(expansion)
            break

    if not extra_terms:
        return query

    return f"{query} {' '.join(extra_terms)}"
