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

CARD_CATEGORY_MARKERS: list[tuple[re.Pattern[str], str, list[str]]] = [
    (
        re.compile(r"CreditCardsID|بطاقات\s*الائتمان|بطاقات\s*الإئتمان", re.I),
        "بطاقات الائتمان",
        [
            "ماستركارد استاندرد",
            "فيزا كلاسيك",
            "ماستركارد مصر للطيران",
            "فيزا جولد",
            "ماستركارد تيتانيوم",
            "ماستركارد UEFA Champions League",
            "فيزا بلاتينم",
            "فيزا بلاتينم الدولارية",
            "ماستركارد بلاتينم",
            "ماستركارد وورلد",
            "فيزا سيغنتشر",
            "ماستركارد وورلد إيليت",
            "فيزا انفينيت",
        ],
    ),
    (
        re.compile(r"DepitCardsID|بطاقات\s*الخصم\s*المباشر", re.I),
        "بطاقات الخصم المباشر",
        [
            "ميزة",
            "كلاسيك",
            "جولد",
            "تيتانيوم",
            "بلاتينم",
            "وورلد",
            "وورلد إيليت",
            "فيزا بلاتينم بالدولار الأمريكي",
        ],
    ),
    (
        re.compile(r"PrepaidCardsID|البطاقات\s*المدفوعة\s*مقدما|المدفوعة\s*مقدم", re.I),
        "البطاقات المدفوعة مقدماً",
        [
            "بطاقة ميزة المدفوعة مقدما",
            "البطاقات المدفوعة مقدما من البنك الأهلى المصري",
            "بطاقة ميزة المدفوعة مقدمًا الموحدة للجامعات",
        ],
    ),
]

CREDIT_CARD_TIERS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "بطاقات الدخول",
        [
            (
                "فيزا كلاسيك / ماستركارد استاندرد",
                "مناسبة للاستخدام المحلي والدولي، مع برنامج نقاط الأهلي وفترة سماح على المشتريات.",
            ),
            (
                "ماستركارد مصر للطيران",
                "بطاقة مرتبطة ببرنامج مصر للطيران مع مزايا سفر ونقاط مكافآت.",
            ),
        ],
    ),
    (
        "بطاقات متوسطة",
        [
            (
                "فيزا جولد / ماستركارد تيتانيوم",
                "حد ائتماني أعلى ومزايا إضافية للتسوق محلياً ودولياً.",
            ),
            (
                "ماستركارد UEFA Champions League",
                "بطاقة مرتبطة بدوري أبطال أوروبا مع مزايا ترويجية.",
            ),
        ],
    ),
    (
        "بطاقات مميزة (بريميوم)",
        [
            ("فيزا بلاتينم / ماستركارد بلاتينم", "مزايا رفاهية أعلى وحدود استخدام أكبر."),
            ("فيزا بلاتينم الدولارية", "بطاقة بالدولار الأمريكي للمعاملات الدولية."),
            ("ماستركارد وورلد / فيزا سيغنتشر", "مزايا سفر وخدمات مميزة لعملاء الملاءة المتوسطة والعالية."),
            ("ماستركارد وورلد إيليت / فيزا انفينيت", "أعلى فئات البطاقات مع حدود ائتمانية كبيرة وخدمات حصرية."),
        ],
    ),
]

SHARED_CREDIT_FEATURES = [
    "برنامج الأهلي بوينتس لاستبدال النقاط أو استرداد نقدي.",
    "تقسيط المشتريات حتى 36 شهراً عبر خدمة التقسيط بالهاتف.",
    "تقسيط بدون فوائد لمدة تصل إلى 12 شهراً لدى تجار معتمدين.",
    "التسوق الآمن عبر الإنترنت بخدمة الكود الأمن OTP.",
    "السداد عبر ماكينات الصراف الآلي، فوري، الأهلي نت، وإنستاباي.",
]

PRODUCT_LINE = re.compile(
    r"^(?:ماستركارد|فيزا|بطاقة|البطاقات)\s+.+$|^(?:ميزة|كلاسيك|جولد|تيتانيوم|بلاتينم|وورلد)$",
    re.I,
)
EGP_LIMIT = re.compile(r"(\d{1,3}(?:[,\.]\d{3})*)\s*جم")
GRACE_DAYS = re.compile(r"حتي\s*(\d+)\s*يوم|حتى\s*(\d+)\s*يوم", re.I)
GENERIC_CARD_NAV = re.compile(r"#/AR/CreditCards$|#/EN/CreditCards$", re.I)

FALLBACK_CARD_CITATIONS: dict[str, Citation] = {
    "credit_category": Citation(
        title="بطاقات الإئتمان - البنك الأهلي المصري",
        url='https://www.nbe.com.eg/NBE/E/#/AR/ProductCategory?inParams={"CategoryID":"CreditCardsID"}',
    ),
    "debit_category": Citation(
        title="بطاقات الخصم المباشر - البنك الأهلي المصري",
        url='https://www.nbe.com.eg/NBE/E/#/AR/ProductCategory?inParams={"CategoryID":"DepitCardsID"}',
    ),
    "prepaid_category": Citation(
        title="البطاقات المدفوعة مقدماً - البنك الأهلي المصري",
        url='https://www.nbe.com.eg/NBE/E/#/AR/ProductCategory?inParams={"CategoryID":"PrepaidCardsID"}',
    ),
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
    """Pick diverse official card pages instead of one generic navigation link."""
    has_rich_credit_source = any("CreditCardsID" in (chunk.url or "") for chunk in chunks)

    ranked: list[tuple[int, float, RetrievedChunk]] = []
    for chunk in chunks:
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
    seen_families: set[str] = set()

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
        if "CreditCardsID" in chunk.url:
            seen_families.add("credit")
        if "DepitCardsID" in chunk.url:
            seen_families.add("debit")
        if "PrepaidCardsID" in chunk.url:
            seen_families.add("prepaid")
        if len(citations) >= max_items:
            break

    if "credit" in include_families:
        has_credit_category = any(
            "CreditCardsID" in citation.url and "ProductCategory" in citation.url
            for citation in citations
        )
        if not has_credit_category:
            fallback = FALLBACK_CARD_CITATIONS["credit_category"]
            if canonical_url_key(fallback.url) not in seen_urls:
                citations.insert(0, fallback)
                seen_urls.add(canonical_url_key(fallback.url))
                seen_families.add("credit")

    for family in include_families:
        if family not in seen_families and len(citations) < max_items:
            fallback_key = f"{family}_category"
            fallback = FALLBACK_CARD_CITATIONS.get(fallback_key)
            if fallback and canonical_url_key(fallback.url) not in seen_urls:
                citations.append(fallback)
                seen_urls.add(canonical_url_key(fallback.url))

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


def _enrich_product_hint(name: str, text: str, default_hint: str) -> str:
    aliases = [part.strip() for part in re.split(r"/|،", name) if part.strip()]
    window = ""
    for alias in aliases:
        if alias in text:
            idx = text.find(alias)
            window = text[max(0, idx - 40) : idx + 500]
            break
    if not window:
        return default_hint
    extras: list[str] = []
    limit = EGP_LIMIT.search(window)
    if limit:
        extras.append(f"حد ائتماني يبدأ من {limit.group(1)} جنيه")
    grace = GRACE_DAYS.search(window)
    if grace:
        days = grace.group(1) or grace.group(2)
        extras.append(f"فترة سماح حتى {days} يوم على المشتريات")
    if extras:
        return f"{default_hint} ({'، '.join(extras)})"
    return default_hint


def build_credit_cards_answer(chunks: list[RetrievedChunk], language: str) -> str | None:
    if language != "ar":
        return _build_credit_cards_answer_en(chunks)

    source = _credit_card_chunks(chunks) or chunks[:8]
    combined = "\n".join(chunk.text for chunk in source[:8])

    lines = [
        "يوفّر البنك الأهلي المصري مجموعة واسعة من بطاقات الائتمان ضمن فئات مختلفة حسب احتياجات العميل وملاءته المالية:",
        "",
    ]
    for tier_name, products in CREDIT_CARD_TIERS:
        lines.append(f"{tier_name}:")
        for product_name, hint in products:
            detail = _enrich_product_hint(product_name, combined, hint)
            lines.append(f"• {product_name}: {detail}")
        lines.append("")

    lines.append("مزايا مشتركة لبطاقات الائتمان:")
    for feature in SHARED_CREDIT_FEATURES:
        lines.append(f"• {feature}")

    lines.append(
        "\nللمقارنة بين البطاقات أو طلب إصدار بطاقة، راجع صفحة بطاقات الائتمان على موقع البنك الأهلي أو تواصل مع أقرب فرع / خدمة الأهلي فون 19623."
    )
    return "\n".join(lines)


def _build_credit_cards_answer_en(chunks: list[RetrievedChunk]) -> str | None:
    return (
        "NBE offers multiple credit card tiers: entry cards (Visa Classic, Mastercard Standard), "
        "mid-tier (Visa Gold, Titanium), and premium cards (Platinum, World, Signature, Infinite). "
        "Benefits include Al Ahly Points, installment plans, OTP secure online shopping, and multiple repayment channels. "
        "See nbe.com.eg cards section or visit a branch for details."
    )


def _collect_products_from_text(text: str, defaults: list[str]) -> list[str]:
    found: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or len(line) < 3:
            continue
        if PRODUCT_LINE.match(line) and line not in found:
            found.append(line)
    return found or defaults


def build_card_types_answer(chunks: list[RetrievedChunk], language: str) -> str | None:
    if language != "ar":
        return _build_card_types_answer_en(chunks)

    sections: list[str] = []
    card_chunks = [
        chunk
        for chunk in chunks
        if "بطاق" in (chunk.title or "") or "card" in (chunk.url or "").lower()
    ]
    source = card_chunks or chunks[:8]
    combined = "\n".join(f"{chunk.title}\n{chunk.url}\n{chunk.text}" for chunk in source[:8])

    for pattern, category, defaults in CARD_CATEGORY_MARKERS:
        if pattern.search(combined):
            category_text = "\n".join(
                chunk.text
                for chunk in source
                if pattern.search(f"{chunk.title} {chunk.url} {chunk.text}")
            )
            products = _collect_products_from_text(category_text, defaults)
        else:
            products = defaults
        bullet = "\n  • ".join(products)
        sections.append(f"• {category}:\n  • {bullet}")

    return (
        "وفقاً لمحتوى موقع البنك الأهلي المصري، أنواع البطاقات البنكية المتاحة تشمل:\n"
        + "\n".join(sections)
        + "\n\nلتفاصيل كل بطاقة أو طلب الإصدار، راجع قسم البطاقات على موقع البنك أو أقرب فرع."
    )


def _build_card_types_answer_en(chunks: list[RetrievedChunk]) -> str | None:
    sections = [
        "• Credit cards: Visa Classic, Visa Gold, Visa Platinum, Mastercard Standard, and more",
        "• Debit cards: Meeza, Classic, Gold, Titanium, Platinum, World, World Elite",
        "• Prepaid cards: Meeza prepaid and university prepaid cards",
    ]
    return (
        "According to NBE website content, available bank card types include:\n"
        + "\n".join(sections)
        + "\nSee the cards section on nbe.com.eg or visit a branch for details."
    )
