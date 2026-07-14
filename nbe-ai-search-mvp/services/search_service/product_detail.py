"""Extractive answers for named certificate/product queries."""

from __future__ import annotations

import re

from ingestion.embedding.vector_store import RetrievedChunk
from shared.arabic_normalize import normalize_arabic

INVESTMENT_CERT_QUERY = re.compile(
    r"شهاد[ةه]?\s*استثمار\s*['\"'']?\s*([ابج])\s*['\"'']?",
    re.IGNORECASE,
)

LETTER_DISPLAY = {"ا": "أ", "ب": "ب", "ج": "ج"}

NAMED_PRODUCT_QUERY = re.compile(
    r"(ما\s*هي|ما\s*هو|تفاصيل|معلومات|اشرح|شرح|about|what\s+is|details?)",
    re.IGNORECASE,
)

FIELD_LABELS = (
    "فئات الشهادة",
    "المدة",
    "دورية صرف العائد",
    "سعر العائد",
    "من يحق لهم الشراء",
    "الاقتراض",
    "ألاسترداد",
    "مميزات أخري",
)

SKIP_LINES = re.compile(
    r"^(?:#|شهادات الادخار|شهادات الاستثمار|منتجات اخرى|هذه الصفحه غير متاحة|للمقارنة)",
    re.IGNORECASE,
)


def extract_target_product_label(query: str, language: str) -> str | None:
    if language != "ar":
        return None
    text = normalize_arabic(query)
    match = INVESTMENT_CERT_QUERY.search(text)
    if match:
        letter = LETTER_DISPLAY.get(match.group(1).strip(), match.group(1).strip())
        return f"شهادة استثمار '{letter}'"
    return None


def is_named_product_query(query: str, language: str) -> bool:
    if extract_target_product_label(query, language):
        return True
    if language == "ar":
        return bool(NAMED_PRODUCT_QUERY.search(normalize_arabic(query)) and "شهاد" in normalize_arabic(query))
    return bool(NAMED_PRODUCT_QUERY.search(query.lower()) and "certificate" in query.lower())


def _normalized_product_key(label: str) -> str:
    text = normalize_arabic(label)
    text = re.sub(r"['\"'']", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def find_product_chunk(chunks: list[RetrievedChunk], product_label: str) -> RetrievedChunk | None:
    target = _normalized_product_key(product_label)
    best: RetrievedChunk | None = None
    best_score = -1.0

    for chunk in chunks:
        title_key = _normalized_product_key(chunk.title or "")
        text_head = _normalized_product_key(chunk.text[:240])
        title_match = target in title_key
        text_match = target in text_head and "productdetails" in (chunk.url or "").lower()
        if not title_match and not text_match:
            continue
        if chunk.score > best_score:
            best = chunk
            best_score = chunk.score

    return best


def prioritize_product_chunks(
    query: str,
    language: str,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    label = extract_target_product_label(query, language)
    if not label:
        return chunks
    match = find_product_chunk(chunks, label)
    if not match:
        return chunks
    rest = [chunk for chunk in chunks if chunk.url != match.url]
    return [match, *rest]


def _parse_fields(content: str) -> dict[str, str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    fields: dict[str, str] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line not in FIELD_LABELS:
            idx += 1
            continue
        values: list[str] = []
        idx += 1
        while idx < len(lines):
            candidate = lines[idx]
            if candidate in FIELD_LABELS or candidate.startswith("#") or SKIP_LINES.match(candidate):
                break
            values.append(candidate)
            idx += 1
        if values:
            fields[line] = " ".join(values[:8])
    return fields


def _extract_highlights(content: str) -> list[str]:
    highlights: list[str] = []
    capture = False
    for line in content.splitlines():
        line = line.strip()
        if line == "مميزات أخري":
            capture = True
            continue
        if not capture:
            continue
        if line.startswith("#") or SKIP_LINES.match(line):
            break
        if len(line) > 8:
            highlights.append(line)
    return highlights[:4]


def build_product_detail_answer(
    query: str,
    language: str,
    chunks: list[RetrievedChunk],
) -> tuple[str, RetrievedChunk] | None:
    label = extract_target_product_label(query, language)
    if not label:
        return None

    chunk = find_product_chunk(chunks, label)
    if not chunk:
        return None

    fields = _parse_fields(chunk.text)
    if "سعر العائد" not in fields and "فئات الشهادة" not in fields:
        return None

    if language == "ar":
        lines = [f"{label} من البنك الأهلي المصري:"]
        for key in FIELD_LABELS:
            if key in fields:
                lines.append(f"• {key}: {fields[key]}")
        for item in _extract_highlights(chunk.text):
            if item not in fields.get("مميزات أخري", ""):
                lines.append(f"• {item}")
        lines.append("للاشتراك أو المزيد من التفاصيل راجع صفحة المنتج على موقع البنك أو أقرب فرع.")
        return "\n".join(lines), chunk

    lines = [f"{label} — National Bank of Egypt:"]
    for key in FIELD_LABELS:
        if key in fields:
            lines.append(f"- {key}: {fields[key]}")
    return "\n".join(lines), chunk


def try_product_detail_answer(
    query: str,
    language: str,
    chunks: list[RetrievedChunk],
) -> tuple[str, RetrievedChunk] | None:
    if not is_named_product_query(query, language):
        return None
    return build_product_detail_answer(query, language, chunks)
