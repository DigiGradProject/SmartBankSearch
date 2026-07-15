"""Unit tests for layered semantic-first intent classification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services.rag.critical_rules import classify_critical
from services.rag.intent_fallback import classify_layered, classify_with_fallback
from services.search_service.intent_classifier import QueryIntent, classify_query


def _semantic(intent: str, confidence: float) -> QueryIntent:
    return QueryIntent(
        intent=intent,
        category="exchange_rates" if intent == "exchange_rate" else "general",
        confidence=confidence,
        allowed_doc_types=("exchange_rate", "currency_converter"),
        source="semantic",
    )


def test_critical_customer_service_bypasses_semantic():
    result = classify_critical("رقم خدمة العملاء 19623", "ar")
    assert result is not None
    assert result.intent == "customer_service"
    assert result.source == "critical"
    assert result.confidence >= 0.99


def test_critical_forgot_password():
    result = classify_critical("نسيت الباسورد", "ar")
    assert result is not None
    assert result.intent == "password_reset"


@patch("services.rag.intent_fallback.classify_semantic")
@patch("services.rag.intent_fallback.classify_critical")
def test_high_semantic_confidence_wins(mock_critical, mock_semantic):
    mock_critical.return_value = None
    mock_semantic.return_value = _semantic("exchange_rate", 0.92)

    result = classify_layered("الدولار بكام النهاردة", "ar")
    assert result.intent == "exchange_rate"
    assert result.source == "semantic"
    mock_semantic.assert_called_once()


@patch("services.rag.intent_fallback.classify_semantic")
@patch("services.rag.intent_fallback.classify_critical")
def test_mid_band_uses_regex_when_regex_matches(mock_critical, mock_semantic):
    mock_critical.return_value = None
    mock_semantic.return_value = _semantic("exchange_rate", 0.75)

    result = classify_layered("سعر الصرف", "ar")
    assert result.intent == "exchange_rate"
    assert result.source == "regex"


@patch("services.rag.intent_fallback.classify_semantic")
@patch("services.rag.intent_fallback.classify_critical")
def test_mid_band_uses_semantic_when_regex_misses(mock_critical, mock_semantic):
    mock_critical.return_value = None
    mock_semantic.return_value = _semantic("exchange_rate", 0.72)

    result = classify_layered("عايز أعرف سعر بيع الدولار النهاردة", "ar")
    assert result.intent == "exchange_rate"
    assert result.source == "semantic"
    assert classify_query("عايز أعرف سعر بيع الدولار النهاردة", "ar").intent == "general_faq"


@patch("services.rag.intent_fallback.classify_semantic")
@patch("services.rag.intent_fallback.classify_critical")
def test_low_semantic_falls_back_to_regex(mock_critical, mock_semantic):
    mock_critical.return_value = None
    mock_semantic.return_value = _semantic("general_faq", 0.35)

    result = classify_layered("قرض شخصي", "ar")
    assert result.intent == "personal_loan"
    assert result.source == "regex"


@patch("services.rag.intent_fallback.classify_semantic")
@patch("services.rag.intent_fallback.classify_critical")
def test_both_fail_returns_general_faq(mock_critical, mock_semantic):
    mock_critical.return_value = None
    mock_semantic.return_value = _semantic("general_faq", 0.20)

    result = classify_layered("مرحبا كيف الحال", "ar")
    assert result.intent == "general_faq"
    assert result.source == "general"


@patch("services.rag.intent_fallback.settings")
def test_legacy_mode_regex_primary(mock_settings):
    mock_settings.semantic_intent_enabled = False
    mock_settings.minilm_intent_fallback_enabled = False

    result = classify_with_fallback("سعر الصرف", "ar")
    assert result.intent == "exchange_rate"
    assert result.source == "regex"


@pytest.mark.parametrize(
    "query,language,expected",
    [
        ("19623", "ar", "customer_service"),
        ("forgot password NBE", "en", "password_reset"),
    ],
)
def test_critical_rules_parametrized(query, language, expected):
    result = classify_layered(query, language)
    assert result.intent == expected
    assert result.source == "critical"
