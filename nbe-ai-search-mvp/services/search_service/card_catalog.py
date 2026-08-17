"""Extractive answers for bank card catalog / types questions."""

from __future__ import annotations

import re

from ingestion.embedding.vector_store import RetrievedChunk
from services.search_service.intent_classifier import classify_query
from shared.arabic_normalize import normalize_arabic
from shared.document_quality import is_broken_citation_url, is_junk_document, is_low_value_document
from shared.schemas import Citation
from shared.url_canonical import canonical_url_key

TYPES_QUERY_AR = re.compile(
    r"انواع?\s*(ال)?بطاق|ما\s*هي\s*(ال)?بطاق|انواع?\s*البطاقات\s*البنكيه|"
    r"قائمة\s*البطاقات|انواع?\s*كروت|انواع?\s*الكروت",
    re.I,
)
TYPES_QUERY_EN = re.compile(
    r"types?\s+of\s+(bank\s+)?cards?|card\s+types?|debit\s+and\s+credit\s+cards?",
    re.I,
)
CREDIT_OVERVIEW_AR = re.compile(r"بطاق.{0,8}ائتمان|كريدت\s*كارد", re.I)
CREDIT_OVERVIEW_EN = re.compile(r"credit\s*cards?", re.I)
DEBIT_OR_PREPAID = re.compile(r"خصم|مدفوع|مدين|debit|prepaid", re.I)

GENERIC_CARD_NAV = re.compile(r"#/AR/CreditCards$|#/EN/CreditCards$", re.I)
CARD_FAMILY_LABELS = {
    "ar": {
        "credit": "بطاقات الائتمان",
        "debit": "بطاقات الخصم المباشر",
        "prepaid": "البطاقات المدفوعة مقدماً",
    },
    "en": {
        "credit": "Credit cards",
        "debit": "Debit cards",
        "prepaid": "Prepaid cards",
    },
}


def _citation_priority(chunk: RetrievedChunk) -> int:
    url = chunk.url or ""
    title = (chunk.title or "").strip()
    if "CreditCardsID" in url and "ProductCategory" in url:
        return 100
    if "DepitCardsID" in url and "ProductCategory" in url:
        return 95
    if "PrepaidCardsID" in url and "ProductCategory" in url:
        return 90
    if "CreditCardsID" in url and "ProductDetails" in url:
        return 80
    if "DepitCardsID" in url and "ProductDetails" in url:
        return 75
    if "PrepaidCardsID" in url and "ProductDetails" in url:
        return 70
    if GENERIC_CARD_NAV.search(url) or title in {"بطاقات الائتمان", "بطاقات الائتمان"}:
        return 15
    if "CreditCards" in url or "DebitCards" in url:
        return 25
    if "بطاق" in title:
        return 40
    return 0


def _card_family(chunk: RetrievedChunk) -> str:
    url = (chunk.url or "").lower()
    doc_type = getattr(chunk, "doc_type", "").lower()
    title = (chunk.title or "").lower()
    if "prepaidcardsid" in url or "prepaid" in doc_type or "مدفوع" in title:
        return "prepaid"
    if "depitcardsid" in url or "debit" in doc_type or "خصم" in title:
        return "debit"
    if (
        "creditcardsid" in url
        or "credit_card" in doc_type
        or "credit card" in title
        or "ائتمان" in title
    ):
        return "credit"
    return ""

def _is_citable_card_chunk(chunk: RetrievedChunk) -> bool:
    if not chunk.url or is_broken_citation_url(chunk.url):
        return False
    if is_junk_document(chunk.document_id) or is_low_value_document(chunk.document_id):
        return False
    title = (chunk.title or "").strip()
    return len(title) >= 4


def build_card_citations(
    chunks: list[RetrievedChunk],
    *,
    max_items: int = 5,
    include_families: tuple[str, ...] = ("credit", "debit", "prepaid"),
    language: str = "ar",
) -> list[Citation]:
    """Return only citations that were present in the retrieved chunks."""
    has_rich_credit_source = any("CreditCardsID" in (chunk.url or "") for chunk in chunks)

    ranked: list[tuple[int, float, RetrievedChunk]] = []
    for chunk in chunks:
        family = _card_family(chunk)
        if family and family not in include_families:
            continue
        if not _is_citable_card_chunk(chunk):
            continue
        if language == "ar" and "/EN/" in (chunk.url or ""):
            continue
        priority = _citation_priority(chunk)
        if GENERIC_CARD_NAV.search(chunk.url or "") and has_rich_credit_source:
            continue
        ranked.append((priority, chunk.score, chunk))

    ranked.sort(key=lambda item: (-item[0], -item[1]))

    citations: list[Citation] = []
    seen_urls: set[str] = set()

    for _, _, chunk in ranked:
        url_key = canonical_url_key(chunk.url)
        if url_key in seen_urls:
            continue
        seen_urls.add(url_key)
        citations.append(
            Citation(
                title=chunk.title or chunk.url,
                url=chunk.url,
                category=getattr(chunk, "category", None) or "cards",
                relevance_score=round(chunk.score, 3),
                reranker_score=round(chunk.score, 3),
            )
        )
        if len(citations) >= max_items:
            break

    citations = [
        citation
        for citation in citations
        if not (GENERIC_CARD_NAV.search(citation.url) and has_rich_credit_source)
    ]

    return citations[:max_items]



def is_card_types_query(query: str, language: str) -> bool:
    intent = classify_query(query, language)
    if intent.intent == "card_types":
        return True
    if language == "ar":
        return bool(TYPES_QUERY_AR.search(normalize_arabic(query)))
    return bool(TYPES_QUERY_EN.search(query.lower()))


def is_credit_cards_overview_query(query: str, language: str) -> bool:
    if is_card_types_query(query, language):
        return False
    intent = classify_query(query, language)
    if intent.intent == "credit_card":
        if language == "ar" and DEBIT_OR_PREPAID.search(normalize_arabic(query)):
            return False
        return True
    if language == "ar":
        text = normalize_arabic(query)
        return bool(CREDIT_OVERVIEW_AR.search(text)) and not DEBIT_OR_PREPAID.search(text)
    return bool(CREDIT_OVERVIEW_EN.search(query.lower()))


def _credit_card_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [
        chunk
        for chunk in chunks
        if "CreditCardsID" in (chunk.url or "")
        or "CreditCards" in (chunk.url or "")
        or re.search(r"بطاقات\s*الائتمان|بطاقات\s*الإئتمان", chunk.title or "", re.I)
    ]


def prioritize_credit_card_chunks(
    query: str,
    language: str,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    if not is_credit_cards_overview_query(query, language):
        return chunks

    def sort_key(chunk: RetrievedChunk) -> tuple[int, float]:
        url = chunk.url or ""
        text_len = len((chunk.text or "").strip())
        priority = 0
        if "CreditCardsID" in url and "ProductCategory" in url:
            priority = 4
        elif "CreditCardsID" in url and "ProductDetails" in url:
            priority = 3
        elif "CreditCardsID" in url:
            priority = 2
        elif "CreditCards" in url:
            priority = 1
        if text_len < 80:
            priority -= 2
        return (-priority, -chunk.score)

    return sorted(chunks, key=sort_key)


def _clean_source_title(title: str) -> str:
    return re.sub(
        r"^(?:البنك\s+الأهلى\s+(?:المصرى|المصري)|National Bank of Egypt)\s*-\s*",
        "",
        title.strip(),
        flags=re.I,
    ).strip()


def _card_names(
    chunks: list[RetrievedChunk],
    language: str,
    *,
    family: str | None = None,
    limit: int = 12,
) -> list[str]:
    names: list[str] = []
    for chunk in chunks:
        chunk_family = _card_family(chunk)
        if family and chunk_family != family:
            continue
        if not chunk_family and getattr(chunk, "category", "") != "cards":
            continue
        candidate = _clean_source_title(chunk.product_name or chunk.title)
        if not candidate:
            continue
        if language == "ar" and not re.search(r"[\u0600-\u06FF]", candidate):
            continue
        if language == "en" and re.search(r"[\u0600-\u06FF]", candidate):
            continue
        if candidate not in names:
            names.append(candidate)
    return names[:limit]


def build_credit_cards_answer(chunks: list[RetrievedChunk], language: str) -> str | None:
    names = _card_names(_credit_card_chunks(chunks), language, family="credit")
    if not names:
        return None
    bullet = "\n• ".join(names)
    if language == "ar":
        return (
            "وفقاً للمستندات المسترجعة، بطاقات الائتمان المطابقة تشمل:\n"
            f"• {bullet}\n"
            "لم تُضف أي مزايا أو حدود غير موجودة في هذه المصادر."
        )
    return (
        "According to the retrieved documents, matching credit card pages include:\n"
        f"• {bullet}\n"
        "No benefits or limits were added beyond those sources."
    )


def build_card_types_answer(chunks: list[RetrievedChunk], language: str) -> str | None:
    sections: list[str] = []
    labels = CARD_FAMILY_LABELS[language]
    for family in ("credit", "debit", "prepaid"):
        names = _card_names(chunks, language, family=family)
        if names:
            bullet = "\n  • ".join(names)
            sections.append(f"• {labels[family]}:\n  • {bullet}")

    if not sections:
        return None
    if language == "ar":
        return (
            "وفقاً للمستندات المسترجعة، صفحات البطاقات المطابقة تشمل:\n"
            + "\n".join(sections)
            + "\nالتفاصيل مقتصرة على المنتجات التي ظهرت في نتائج البحث."
        )
    return (
        "According to the retrieved documents, matching card pages include:\n"
        + "\n".join(sections)
        + "\nDetails are limited to products present in the retrieval results."
    )
