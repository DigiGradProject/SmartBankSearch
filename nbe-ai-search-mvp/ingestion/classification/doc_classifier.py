"""Classify NBE documents into banking categories for metadata filtering."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import unquote

# URL slug -> (doc_type, category)
URL_SLUG_MAP: dict[str, tuple[str, str]] = {
    "ExchangeRatesAndCurrencyConverter": ("exchange_rate", "exchange_rates"),
    "CertificatesRatesForeignCurrency": ("certificate_rate", "certificates"),
    "Depositrateslocalcurrency": ("certificate_rate", "certificates"),
    "Depositcertificate": ("certificate", "certificates"),
    "CertificatesID": ("certificate", "certificates"),
    "BeladyCertificateID": ("certificate", "certificates"),
    "CreditCards": ("credit_card", "cards"),
    "DebitCards": ("debit_card", "cards"),
    "Loans": ("loan", "loans"),
    "ATMBranch": ("branch", "branches"),
    "Accounts": ("account", "accounts"),
    "AccountsFAQs": ("account", "accounts"),
    "FAQs": ("faq", "general"),
    "PhoneCash": ("wallet", "wallet"),
    "CustomerLogin": ("digital_banking", "digital_banking"),
    "Offers": ("offer", "offers"),
    "ProductDetails": ("product", "products"),
    "ProductCategory": ("product", "products"),
    "CategorySubCategory": ("general", "general"),
}

# ProductCategory CategoryID hints
CATEGORY_ID_MAP: dict[str, tuple[str, str]] = {
    "certificatesid": ("certificate", "certificates"),
    "beladycertificateid": ("certificate", "certificates"),
    "beladyoneyear": ("certificate", "certificates"),
    "beladythreeyears": ("certificate", "certificates"),
    "beladyfiveyears": ("certificate", "certificates"),
    "creditcards": ("credit_card", "cards"),
    "debitcards": ("debit_card", "cards"),
    "loans": ("loan", "loans"),
    "personalloan": ("loan", "loans"),
    "accounts": ("account", "accounts"),
    "phonecash": ("wallet", "wallet"),
}

TITLE_KEYWORDS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"سعر\s*الصرف|تحويل\s*العملات|محول\s*العملات|exchange\s*rate|currency\s*convert", re.I), "exchange_rate", "exchange_rates"),
    (re.compile(r"شهاد.{0,6}(عائد|فايد|سعر|اسعار|أسعار)|certificate.{0,10}rate|yield", re.I), "certificate_rate", "certificates"),
    (re.compile(r"شهاد|certificate|belady|بلادي", re.I), "certificate", "certificates"),
    (re.compile(r"بطاق.{0,6}ائتمان|credit\s*card", re.I), "credit_card", "cards"),
    (re.compile(r"بطاق.{0,6}(مدين|خصم)|debit\s*card", re.I), "debit_card", "cards"),
    (re.compile(r"قرض|قروض|loan|تمويل", re.I), "loan", "loans"),
    (re.compile(r"فرع|فروع|branch|atm|صراف", re.I), "branch", "branches"),
    (re.compile(r"حساب|account", re.I), "account", "accounts"),
    (re.compile(r"فون\s*كاش|phone\s*cash", re.I), "wallet", "wallet"),
]

CONTENT_HINTS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"سعر\s*البنكنوت|banknote\s*rate|transfer\s*rate", re.I), "exchange_rate", "exchange_rates"),
    (re.compile(r"شراء\s*\d+\.\d+\s*بيع", re.I), "exchange_rate", "exchange_rates"),
]


@dataclass(frozen=True)
class DocumentClassification:
    doc_type: str
    category: str
    canonical_url_slug: str
    source_type: str = "static_scrape"
    is_stub: bool = False
    quality_score: float = 0.7


def _extract_url_slug(url: str) -> str:
    if not url:
        return ""
    if "#/AR/" in url or "#/EN/" in url:
        fragment = url.split("#/", 1)[-1]
        slug = fragment.split("?", 1)[0]
        if slug.startswith("AR/") or slug.startswith("EN/"):
            slug = slug.split("/", 1)[-1]
        return slug
    return ""


def _extract_category_id(url: str) -> str:
    if "CategoryID" not in url:
        return ""
    try:
        if "inParams=" in url:
            raw = url.split("inParams=", 1)[1]
            raw = unquote(raw)
            payload = json.loads(raw)
            return str(payload.get("CategoryID", "")).lower()
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""
    return ""


def classify_document(
    *,
    url: str,
    title: str,
    content: str,
    language: str,
    metadata: dict | None = None,
) -> DocumentClassification:
    meta = metadata or {}
    slug = _extract_url_slug(url)
    category_id = _extract_category_id(url)

    if meta.get("doc_type") and meta.get("category"):
        return DocumentClassification(
            doc_type=str(meta["doc_type"]),
            category=str(meta["category"]),
            canonical_url_slug=slug or str(meta.get("canonical_url_slug", "")),
            source_type=str(meta.get("source", meta.get("source_type", "static_scrape"))),
            is_stub=meta.get("source") == "product_stub" or bool(meta.get("is_stub")),
            quality_score=float(meta.get("quality_score", 0.85 if meta.get("source") == "product_stub" else 0.7)),
        )

    doc_type, category = "general", "general"
    if slug in URL_SLUG_MAP:
        doc_type, category = URL_SLUG_MAP[slug]
    elif category_id and category_id in CATEGORY_ID_MAP:
        doc_type, category = CATEGORY_ID_MAP[category_id]

    title_text = f"{title} {slug.replace('_', ' ')}"
    for pattern, dt, cat in TITLE_KEYWORDS:
        if pattern.search(title_text):
            doc_type, category = dt, cat
            break

    if doc_type == "general":
        sample = content[:2500]
        for pattern, dt, cat in CONTENT_HINTS:
            if pattern.search(sample):
                doc_type, category = dt, cat
                break

    # Generic SPA shells can embed FX widgets; keep them out of the exchange_rate filter.
    if slug in {"CategorySubCategory", "ProductCategory", "ProductDetails"} and doc_type == "exchange_rate":
        doc_type, category = "general", "general"

    quality = 0.5
    if len(content) >= 500:
        quality += 0.15
    if meta.get("has_tables"):
        quality += 0.1
    if doc_type != "general":
        quality += 0.1
    if meta.get("source") == "product_stub":
        quality = max(quality, 0.85)

    return DocumentClassification(
        doc_type=doc_type,
        category=category,
        canonical_url_slug=slug,
        source_type=str(meta.get("source", "static_scrape")),
        is_stub=meta.get("source") == "product_stub",
        quality_score=min(1.0, quality),
    )
