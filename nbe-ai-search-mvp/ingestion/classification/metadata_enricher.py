"""Automatic metadata enrichment for NBE documents and chunks."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from urllib.parse import unquote

from ingestion.classification.doc_classifier import classify_document
from ingestion.cleaning.cleaner import cleaning_quality_penalty
from shared.document_quality import is_menu_heavy_text

PRODUCT_SECTION = re.compile(
    r"(الأوراق المطلوبة|الاوراق المطلوبة|المستندات|مميزات|الشروط|الرسوم|العائد|"
    r"eligibility|required documents|features|fees|interest)",
    re.I,
)

BANKING_KEYWORDS = (
    "حساب",
    "شهادة",
    "بطاقة",
    "قرض",
    "عائد",
    "فرع",
    "صراف",
    "تحويل",
    "account",
    "certificate",
    "card",
    "loan",
    "branch",
    "atm",
    "rate",
)

PAGE_TYPE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"OpenYourBankAccountInEgypt", re.I), "initiative"),
    (re.compile(r"ProductDetails", re.I), "product"),
    (re.compile(r"ProductCategory", re.I), "category"),
    (re.compile(r"FAQs?|أسئلة", re.I), "faq"),
    (re.compile(r"Offers?|عروض", re.I), "offer"),
    (re.compile(r"News|أخبار", re.I), "news"),
    (re.compile(r"Report|تقرير", re.I), "report"),
    (re.compile(r"ATMBranch|Branch", re.I), "branch"),
    (re.compile(r"#/AR/|#/EN/", re.I), "web_page"),
]

SUBCATEGORY_MAP = {
    "currentaccountsid": "current_accounts",
    "savinglocalaccountsid": "savings_local",
    "savingforeignaccountsid": "savings_foreign",
    "openyourbankaccountinegypt": "diaspora_initiative",
    "accountsid": "accounts_hub",
    "creditcardsid": "credit_cards",
    "depitcardsid": "debit_cards",
    "prepaidcardsid": "prepaid_cards",
    "localcertificatesid": "certificates_local",
    "forigencertificatesid": "certificates_foreign",
    "certificatesid": "certificates_hub",
}


@dataclass
class EnrichedMetadata:
    title: str
    page_type: str
    category: str
    subcategory: str
    language: str
    product_name: str
    service_name: str
    document_type: str
    intent: str
    keywords: str
    last_updated: str
    doc_type: str
    canonical_url_slug: str
    quality_score: float
    is_stub: bool
    source_type: str


def _category_id(url: str) -> str:
    if "CategoryID" not in url:
        return ""
    try:
        raw = unquote(url.split("inParams=", 1)[1])
        return str(json.loads(raw).get("CategoryID", "")).lower()
    except Exception:  # noqa: BLE001
        return ""


def _product_id(url: str) -> str:
    if "ProductID" not in url:
        return ""
    try:
        raw = unquote(url.split("inParams=", 1)[1])
        value = str(json.loads(raw).get("ProductID", ""))
        return re.sub(r"_\d+$", "", value).replace("_", " ").strip()
    except Exception:  # noqa: BLE001
        return ""


def _page_type(url: str, title: str, content: str) -> str:
    blob = f"{url} {title}"
    for pattern, page_type in PAGE_TYPE_RULES:
        if pattern.search(blob):
            return page_type
    if is_menu_heavy_text(content):
        return "navigation"
    return "web_page"


def _intent_from_category(category: str, doc_type: str) -> str:
    mapping = {
        "exchange_rates": "Exchange Rates",
        "certificates": "Certificates",
        "loans": "Loans",
        "accounts": "Accounts",
        "cards": "Cards",
        "branches": "Branches",
        "offers": "Offers",
        "wallet": "Digital Banking",
        "digital_banking": "Digital Banking",
        "general": "FAQ",
    }
    if doc_type == "offer":
        return "Offers"
    return mapping.get(category, "Other")


def _keywords(title: str, content: str, language: str) -> str:
    text = f"{title}\n{content[:2000]}".lower()
    hits = [kw for kw in BANKING_KEYWORDS if kw.lower() in text]
    if PRODUCT_SECTION.search(content):
        hits.append("required_documents" if language == "ar" or "اوراق" in content else "documents")
    # de-dupe preserve order
    seen: set[str] = set()
    ordered: list[str] = []
    for hit in hits:
        if hit not in seen:
            seen.add(hit)
            ordered.append(hit)
    return ",".join(ordered[:12])


def enrich_document_metadata(
    *,
    url: str,
    title: str,
    content: str,
    language: str,
    metadata: dict | None = None,
) -> EnrichedMetadata:
    meta = metadata or {}
    classification = classify_document(
        url=url,
        title=title,
        content=content,
        language=language,
        metadata=meta,
    )
    cat_id = _category_id(url)
    product_name = _product_id(url) or (title.split(" - ")[-1].strip() if " - " in title else title)
    page_type = _page_type(url, title, content)
    subcategory = SUBCATEGORY_MAP.get(cat_id, classification.canonical_url_slug or "general")
    quality = float(classification.quality_score) - cleaning_quality_penalty(content)
    document_type = "stub" if classification.is_stub else str(meta.get("document_type") or "web_page")
    if meta.get("source") == "pdf":
        document_type = "pdf"

    service_name = ""
    if "PhoneCash" in url or "فون كاش" in title:
        service_name = "Phone Cash"
    elif "InternetBanking" in url or "الأهلي نت" in title:
        service_name = "Internet Banking"

    return EnrichedMetadata(
        title=title,
        page_type=page_type,
        category=classification.category,
        subcategory=subcategory,
        language=language,
        product_name=product_name[:180],
        service_name=service_name,
        document_type=document_type,
        intent=_intent_from_category(classification.category, classification.doc_type),
        keywords=_keywords(title, content, language),
        last_updated=str(meta.get("extracted_at") or meta.get("last_updated") or ""),
        doc_type=classification.doc_type,
        canonical_url_slug=classification.canonical_url_slug,
        quality_score=max(0.1, min(1.0, quality)),
        is_stub=classification.is_stub,
        source_type=classification.source_type,
    )


def enriched_as_extra(enriched: EnrichedMetadata) -> dict[str, object]:
    return asdict(enriched)
