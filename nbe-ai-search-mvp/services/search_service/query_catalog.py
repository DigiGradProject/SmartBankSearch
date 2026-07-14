"""Curated banking queries used for autocomplete and suggestions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogEntry:
    query: str
    label: str
    language: str
    keywords: tuple[str, ...] = ()


POPULAR_QUERIES_AR: list[CatalogEntry] = [
    CatalogEntry("أسعار العملات", "أسعار العملات", "ar", ("سعر", "عملات", "صرف", "دولار")),
    CatalogEntry("سعر الصرف وتحويل العملات", "سعر الصرف وتحويل العملات", "ar", ("سعر", "صرف", "تحويل", "عملات")),
    CatalogEntry("انواع الشهادات البنكية", "انواع الشهادات البنكية", "ar", ("انواع", "شهاد", "بنك")),
    CatalogEntry("شهادات الادخار", "شهادات الادخار", "ar", ("شهاد", "ادخار", "انواع")),
    CatalogEntry("عايز اشتري شهادة", "عايز اشتري شهادة", "ar", ("عايز", "اشتري", "شهاد")),
    CatalogEntry("شهادات الادخار بالعملة المحلية", "شهادات الادخار بالعملة المحلية", "ar", ("شهاد", "محلي", "جنيه", "عائد", "فايده")),
    CatalogEntry("فتح حساب بنكي", "فتح حساب بنكي", "ar", ("حساب", "فتح")),
    CatalogEntry("شهادات بلادي سنة بالدولار", "شهادات بلادي سنة بالدولار", "ar", ("بلاد", "دولار", "شهاد")),
    CatalogEntry("أسعار الشهادات بالعملة الأجنبية", "أسعار الشهادات بالعملة الأجنبية", "ar", ("سعر", "شهاد", "عملة")),
    CatalogEntry("خدمة فون كاش", "خدمة فون كاش", "ar", ("فون", "كاش")),
    CatalogEntry("الأهلي نت", "الأهلي نت", "ar", ("اهلي", "نت")),
    CatalogEntry("بطاقات الائتمان", "بطاقات الائتمان", "ar", ("بطاق", "ائتمان")),
    CatalogEntry("القروض الشخصية", "القروض الشخصية", "ar", ("قرض", "تمويل")),
    CatalogEntry("الفروع وماكينات الصرف الآلي", "الفروع وماكينات الصرف الآلي", "ar", ("فرع", "atm", "صراف")),
]

POPULAR_QUERIES_EN: list[CatalogEntry] = [
    CatalogEntry("exchange rates NBE", "Exchange rates", "en", ("exchange", "rate", "currency")),
    CatalogEntry("currency converter NBE", "Currency converter", "en", ("currency", "convert", "usd")),
    CatalogEntry("phone cash service", "Phone Cash service", "en", ("phone", "cash")),
    CatalogEntry("open bank account NBE", "Open bank account", "en", ("account", "open")),
    CatalogEntry("buy investment certificate", "Buy investment certificate", "en", ("certificate", "buy")),
    CatalogEntry("credit cards NBE", "Credit cards", "en", ("card", "credit")),
    CatalogEntry("personal loan NBE", "Personal loan", "en", ("loan", "personal")),
    CatalogEntry("ATM branch locator", "ATM and branch locator", "en", ("branch", "atm")),
    CatalogEntry("certificates foreign currency rates", "Foreign currency certificate rates", "en", ("rate", "certificate")),
]

TOPIC_QUERIES_AR = {
    "شهاد": "شهادات الادخار بالعملة المحلية",
    "انواع": "انواع الشهادات البنكية",
    "فايده": "شهادات الادخار بالعملة المحلية",
    "فائدة": "شهادات الادخار بالعملة المحلية",
    "عائد": "شهادات الادخار بالعملة المحلية",
    "حساب": "فتح حساب بنكي",
    "بطاق": "بطاقات الائتمان",
    "قرض": "القروض الشخصية",
    "فون كاش": "خدمة فون كاش",
    "الاهلي نت": "الأهلي نت",
    "بلاد": "شهادات بلادي سنة بالدولار",
    "سعر": "أسعار العملات",
    "عملات": "أسعار العملات",
    "صرف": "سعر الصرف وتحويل العملات",
    "فرع": "الفروع وماكينات الصرف الآلي",
}

TOPIC_QUERIES_EN = {
    "certificate": "buy investment certificate",
    "account": "open bank account NBE",
    "card": "credit cards NBE",
    "loan": "personal loan NBE",
    "branch": "ATM branch locator",
    "rate": "exchange rates NBE",
    "exchange": "exchange rates NBE",
    "currency": "currency converter NBE",
}


def catalog_for_language(language: str) -> list[CatalogEntry]:
    return POPULAR_QUERIES_AR if language == "ar" else POPULAR_QUERIES_EN
