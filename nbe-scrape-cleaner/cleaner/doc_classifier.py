"""Classify NBE documents into banking categories (scrape-cleaner)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import unquote

URL_SLUG_MAP: dict[str, tuple[str, str]] = {
    "ExchangeRatesAndCurrencyConverter": ("exchange_rate", "exchange_rates"),
    "CertificatesRatesForeignCurrency": ("certificate_rate", "certificates"),
    "Depositrateslocalcurrency": ("certificate_rate", "certificates"),
    "Depositcertificate": ("certificate", "certificates"),
    "CreditCards": ("credit_card", "cards"),
    "DebitCards": ("debit_card", "cards"),
    "Loans": ("loan", "loans"),
    "ATMBranch": ("branch", "branches"),
    "Accounts": ("account", "accounts"),
    "AccountsFAQs": ("account", "accounts"),
}

CATEGORY_ID_MAP: dict[str, tuple[str, str]] = {
    "certificatesid": ("certificate", "certificates"),
    "beladycertificateid": ("certificate", "certificates"),
    "beladyoneyear": ("certificate", "certificates"),
    "beladythreeyears": ("certificate", "certificates"),
    "beladyfiveyears": ("certificate", "certificates"),
    "creditcards": ("credit_card", "cards"),
    "loans": ("loan", "loans"),
    "accounts": ("account", "accounts"),
}

TITLE_KEYWORDS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"سعر\s*الصرف|تحويل\s*العملات|exchange\s*rate|currency\s*convert", re.I), "exchange_rate", "exchange_rates"),
    (re.compile(r"شهاد|certificate|belady", re.I), "certificate", "certificates"),
    (re.compile(r"بطاق.{0,6}ائتمان|credit\s*card", re.I), "credit_card", "cards"),
    (re.compile(r"قرض|loan", re.I), "loan", "loans"),
    (re.compile(r"فرع|branch|atm|صراف", re.I), "branch", "branches"),
]


@dataclass(frozen=True)
class DocumentClassification:
    doc_type: str
    category: str
    canonical_url_slug: str


def _extract_url_slug(url: str) -> str:
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
            raw = unquote(url.split("inParams=", 1)[1])
            payload = json.loads(raw)
            return str(payload.get("CategoryID", "")).lower()
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""
    return ""


def classify_document(url: str, title: str, content: str) -> DocumentClassification:
    slug = _extract_url_slug(url)
    category_id = _extract_category_id(url)
    doc_type, category = "general", "general"

    if slug in URL_SLUG_MAP:
        doc_type, category = URL_SLUG_MAP[slug]
    elif category_id and category_id in CATEGORY_ID_MAP:
        doc_type, category = CATEGORY_ID_MAP[category_id]

    title_text = f"{title} {slug}"
    for pattern, dt, cat in TITLE_KEYWORDS:
        if pattern.search(title_text):
            doc_type, category = dt, cat
            break

    if doc_type == "general" and re.search(r"سعر\s*البنكنوت|banknote\s*rate", content[:2000], re.I):
        doc_type, category = "exchange_rate", "exchange_rates"

    return DocumentClassification(doc_type=doc_type, category=category, canonical_url_slug=slug)
