"""Staged metadata filter policy (L0 exact → L4 broad)."""

from __future__ import annotations

from dataclasses import dataclass

from services.search_service.intent_classifier import QueryIntent
from shared.config import settings

# Parent / related expansions when exact inventory is thin.
RELATED_DOC_TYPES: dict[str, tuple[str, ...]] = {
    "exchange_rate": ("exchange_rate", "currency_converter"),
    "personal_loan": ("loan",),
    "credit_card": ("credit_card",),
    "card_types": ("credit_card", "debit_card"),
    "account_open": ("account",),
    "certificate_rate": ("certificate_rate", "certificate"),
    "certificate_types": ("certificate", "certificate_rate"),
    "certificate_buy": ("certificate",),
}

# Intents that must not broaden when at least one exact hit exists.
NO_BROAD_IF_ANY_HIT: frozenset[str] = frozenset(
    {
        "exchange_rate",
        "personal_loan",
        "credit_card",
        "certificate_rate",
    }
)

CANONICAL_URL_MARKERS: dict[str, tuple[str, ...]] = {
    "exchange_rate": ("ExchangeRatesAndCurrencyConverter",),
    "personal_loan": ("PersonalLoansCatID", "/Loans", "Loans"),
    "credit_card": ("CreditCardsID", "/CreditCards"),
    "account_open": ("CurrentAccountsID", "AccountsID"),
    "certificate_rate": ("LocalCertificatesID", "CertificatesRatesForeignCurrency"),
}


@dataclass(frozen=True)
class FilterStageResult:
    stage: str  # L0 | L1 | L2 | L3 | L4
    doc_types: list[str] | None
    allow_broad: bool
    min_hits: int


def min_hits_for_intent(intent: QueryIntent) -> int:
    if intent.confidence >= settings.filter_high_confidence:
        return settings.filter_min_hits_high_conf
    return settings.filter_min_hits_low_conf


def exact_doc_types(intent: QueryIntent) -> list[str]:
    return list(intent.allowed_doc_types)


def related_doc_types(intent: QueryIntent) -> list[str]:
    related = RELATED_DOC_TYPES.get(intent.intent)
    if related:
        return list(related)
    return list(intent.allowed_doc_types)


def should_broaden(
    intent: QueryIntent,
    *,
    dense_count: int,
    bm25_count: int,
) -> bool:
    """Return True only when broad fallback is permitted."""
    hit_count = max(dense_count, bm25_count)
    required = min_hits_for_intent(intent)

    if hit_count >= required:
        return False

    # High-confidence banking intents: keep exact filter if we have ANY hit.
    if (
        intent.confidence >= settings.filter_high_confidence
        and hit_count >= 1
        and intent.intent in NO_BROAD_IF_ANY_HIT
    ):
        return False

    if (
        intent.confidence >= settings.filter_high_confidence
        and hit_count >= 1
        and not settings.filter_allow_broad_fallback
    ):
        return False

    return settings.filter_allow_broad_fallback


def canonical_markers_for_intent(intent_name: str) -> tuple[str, ...]:
    return CANONICAL_URL_MARKERS.get(intent_name, ())
