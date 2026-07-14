"""Account-aware ranking for open-account / required-documents queries."""

from __future__ import annotations

import re

from ingestion.embedding.vector_store import RetrievedChunk
from shared.arabic_normalize import normalize_arabic

DIASPORA_HINT = re.compile(
    r"مبادر[ةه]|خارج\s*مصر|سفار[ةه]|قنصلي[ةه]|مغترب|الجالي[ةه]|diaspora|abroad|embassy|initiative",
    re.I,
)
DOCS_HINT = re.compile(
    r"اوراق|أوراق|مستندات|مطلوب|documents?|required|papers?",
    re.I,
)
CURRENT_HINT = re.compile(r"جاري|جارى|current\s*account", re.I)
SAVINGS_HINT = re.compile(r"توفير|ادخار|savings?", re.I)

RETAIL_ACCOUNT_MARKERS = (
    "CurrentAccountsID",
    "SavingLocalAccountsID",
    "SavingForeignAccountsID",
    "AccountsID",
    "AccountsFAQs",
)
DIASPORA_MARKERS = ("OpenYourBankAccountInEgypt",)


def _wants_diaspora(query: str, language: str) -> bool:
    text = normalize_arabic(query) if language == "ar" else query.lower()
    return bool(DIASPORA_HINT.search(text))


def prioritize_account_chunks(
    query: str,
    language: str,
    chunks: list[RetrievedChunk],
    *,
    intent: str | None = None,
) -> list[RetrievedChunk]:
    """Prefer retail account product pages over diaspora initiative pages.

    Only runs for account_open (or when intent is omitted for backward-compatible tests).
    """
    if not chunks:
        return chunks
    if intent is not None and intent != "account_open":
        return chunks

    diaspora = _wants_diaspora(query, language)
    text = normalize_arabic(query) if language == "ar" else query.lower()
    wants_docs = bool(DOCS_HINT.search(text))
    wants_current = bool(CURRENT_HINT.search(text))
    wants_savings = bool(SAVINGS_HINT.search(text))

    def sort_key(chunk: RetrievedChunk) -> tuple[int, float]:
        url = chunk.url or ""
        title = chunk.title or ""
        body = chunk.text or ""
        priority = 0

        if any(marker in url for marker in DIASPORA_MARKERS):
            priority = 8 if diaspora else -3
        elif "CurrentAccountsID" in url:
            priority = 5 if (wants_current or not wants_savings) else 3
        elif "SavingLocalAccountsID" in url or "SavingForeignAccountsID" in url:
            priority = 5 if wants_savings else 3
        elif any(marker in url for marker in RETAIL_ACCOUNT_MARKERS):
            priority = 2
        elif "ProductDetails" in url and "account" in (getattr(chunk, "doc_type", "") or ""):
            priority = 2

        if wants_docs and re.search(r"اوراق|أوراق|مستندات|مطلوب|documents?", f"{title}\n{body}", re.I):
            if not any(marker in url for marker in DIASPORA_MARKERS) or diaspora:
                priority += 2

        # Prefer concrete product pages over thin shells — but not over diaspora when intended.
        if "ProductDetails" in url and not diaspora:
            priority += 1
        if len(body.strip()) < 80:
            priority -= 2

        return (-priority, -chunk.score)

    return sorted(chunks, key=sort_key)
