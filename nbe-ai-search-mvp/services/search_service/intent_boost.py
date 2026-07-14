"""Intent-aware score boosting and penalties after retrieval."""

from __future__ import annotations

from ingestion.embedding.vector_store import RetrievedChunk
from services.search_service.intent_classifier import QueryIntent

INTENT_DOC_TYPE_BOOST: dict[str, dict[str, float]] = {
    "exchange_rate": {"exchange_rate": 0.25, "currency_converter": 0.25},
    "certificate_rate": {"certificate": 0.15, "certificate_rate": 0.25},
    "certificate_types": {"certificate": 0.25},
    "certificate_buy": {"certificate": 0.20},
    "personal_loan": {"loan": 0.25},
    "credit_card": {"credit_card": 0.25},
    "debit_card": {"debit_card": 0.25},
    "branch_locator": {"branch": 0.25},
    "atm_locator": {"branch": 0.20, "atm": 0.25},
    "account_open": {"account": 0.25},
    "wallet": {"wallet": 0.25},
}

INTENT_DOC_TYPE_PENALTY: dict[str, dict[str, float]] = {
    "exchange_rate": {"certificate": 0.35, "certificate_rate": 0.40},
    "certificate_rate": {"exchange_rate": 0.30},
    "certificate_buy": {"exchange_rate": 0.20, "loan": 0.15},
    "personal_loan": {"certificate": 0.20, "exchange_rate": 0.15},
    "credit_card": {"loan": 0.15, "certificate": 0.15},
}

CANONICAL_SLUG_BOOST: dict[str, str] = {
    "exchange_rate": "ExchangeRatesAndCurrencyConverter",
    "certificate_rate": "CertificatesRatesForeignCurrency",
    "certificate_types": "CertificatesID",
    "certificate_buy": "CertificatesID",
    "personal_loan": "Loans",
    "credit_card": "CreditCards",
    "branch_locator": "ATMBranch",
    "atm_locator": "ATMBranch",
    "account_open": "Accounts",
}

# SPA shells that mirror FX tables but are not canonical sources.
NON_CANONICAL_SLUG_PENALTY: dict[str, tuple[str, ...]] = {
    "exchange_rate": ("CategorySubCategory", "ProductCategory", "ProductDetails"),
}


def apply_intent_scoring(
    chunks: list[RetrievedChunk],
    intent: QueryIntent,
    query: str,
) -> list[RetrievedChunk]:
    if intent.intent == "general_faq" or not chunks:
        return chunks

    boosts = INTENT_DOC_TYPE_BOOST.get(intent.intent, {})
    penalties = INTENT_DOC_TYPE_PENALTY.get(intent.intent, {})
    canonical = CANONICAL_SLUG_BOOST.get(intent.intent, "")
    normalized_query = query.lower()

    scored: list[RetrievedChunk] = []
    for chunk in chunks:
        score = chunk.score
        doc_type = getattr(chunk, "doc_type", "general") or "general"

        score += boosts.get(doc_type, 0.0)
        score -= penalties.get(doc_type, 0.0)

        if canonical and canonical in (chunk.url or ""):
            score += 0.35

        slug = getattr(chunk, "canonical_url_slug", "") or ""
        for bad_slug in NON_CANONICAL_SLUG_PENALTY.get(intent.intent, ()):
            if bad_slug in slug or bad_slug in (chunk.url or ""):
                score -= 0.45
                break

        if getattr(chunk, "is_stub", False) and doc_type not in intent.allowed_doc_types:
            score -= 0.25

        if intent.negative_terms:
            text_lower = chunk.text.lower()
            for term in intent.negative_terms:
                if term.lower() in text_lower or term in chunk.title:
                    score -= 0.10
                    break

        if intent.intent == "exchange_rate" and "شهاد" in chunk.title:
            score -= 0.30

        scored.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                url=chunk.url,
                language=chunk.language,
                text=chunk.text,
                score=max(0.0, min(1.0, score)),
                lexical_weights=chunk.lexical_weights,
                doc_type=getattr(chunk, "doc_type", "general"),
                category=getattr(chunk, "category", "general"),
                is_stub=getattr(chunk, "is_stub", False),
                canonical_url_slug=getattr(chunk, "canonical_url_slug", ""),
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    return _dedupe_by_url(scored, canonical)


def _dedupe_by_url(chunks: list[RetrievedChunk], canonical_slug: str) -> list[RetrievedChunk]:
    """Keep the best-scoring chunk per URL; prefer canonical slug ties."""
    best: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        existing = best.get(chunk.url)
        if existing is None:
            best[chunk.url] = chunk
            continue
        chunk_is_canonical = canonical_slug and canonical_slug in chunk.url
        existing_is_canonical = canonical_slug and canonical_slug in existing.url
        if chunk_is_canonical and not existing_is_canonical:
            best[chunk.url] = chunk
        elif chunk.score > existing.score and chunk_is_canonical == existing_is_canonical:
            best[chunk.url] = chunk

    ordered = sorted(best.values(), key=lambda item: item.score, reverse=True)
    if canonical_slug:
        ordered.sort(
            key=lambda item: (
                0 if canonical_slug in (item.url or "") else 1,
                -item.score,
            ),
        )
    return ordered
