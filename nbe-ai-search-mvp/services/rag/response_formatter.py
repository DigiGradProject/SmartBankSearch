"""Structured response formatting helpers."""

from __future__ import annotations

import re

from shared.schemas import Citation

SECTION_HEADERS = (
    "Answer",
    "Key Information",
    "Source",
    "Product",
    "Summary",
    "Features",
    "Eligibility",
    "Required Documents",
    "Fees / Interest",
    "Fees",
    "Interest",
    "Currency",
    "Buying Rate",
    "Selling Rate",
    "Last Updated",
    "Branch",
    "Address",
    "Working Hours",
    "Services",
)


def format_structured_sections(answer: str) -> dict[str, str]:
    """Parse markdown ### sections into a dict; empty if unstructured."""
    if not answer or answer.strip() == "NO_ANSWER":
        return {}

    parts = re.split(r"\n###\s+", answer.strip())
    sections: dict[str, str] = {}
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("### "):
            part = part[4:]
        lines = part.splitlines()
        header = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if header in SECTION_HEADERS or any(header.startswith(h) for h in SECTION_HEADERS):
            sections[header] = body
        elif not sections and header:
            sections["Answer"] = part
    return sections


def citations_as_source_lines(citations: list[Citation]) -> list[str]:
    lines: list[str] = []
    for citation in citations:
        title = (citation.title or "").strip() or citation.url
        lines.append(f"- {title}")
    return lines


def wrap_plain_answer(
    answer: str,
    *,
    language: str,
    citations: list[Citation],
    intent: str = "",
) -> str:
    """
    If the LLM already returned ### sections, keep them.
    Otherwise wrap into the enterprise structured template.
    """
    cleaned = (answer or "").strip()
    if not cleaned or cleaned == "NO_ANSWER":
        return cleaned
    if "### " in cleaned or cleaned.startswith("###"):
        return cleaned

    source_block = "\n".join(citations_as_source_lines(citations)) or "- NBE website"

    # Prefer product template for account/card/loan/certificate intents.
    productish = intent in {
        "account_open",
        "credit_card",
        "debit_card",
        "card_types",
        "personal_loan",
        "certificate_buy",
        "certificate_types",
        "certificate_rate",
    }
    if productish:
        bullets = [
            line.strip("•- ").strip()
            for line in cleaned.splitlines()
            if line.strip() and len(line.strip()) > 8
        ][:8]
        key_info = "\n".join(f"- {item}" for item in bullets) or f"- {cleaned[:280]}"
        product_label = "المنتج" if language == "ar" else "Product"
        summary_label = "الملخص" if language == "ar" else "Summary"
        # Keep English ### headers so prompt contract stays stable across languages.
        return (
            f"### Product\n\n{product_label}\n\n"
            f"### Summary\n\n{cleaned[:600].strip()}\n\n"
            f"### Key Information\n\n{key_info}\n\n"
            f"### Source\n\n{source_block}"
        )

    return (
        f"### Answer\n\n{cleaned}\n\n"
        f"### Key Information\n\n- {cleaned[:240].strip()}\n\n"
        f"### Source\n\n{source_block}"
    )


def ensure_bullet_key_information(answer: str) -> str:
    if "### " in answer:
        return answer
    return answer
