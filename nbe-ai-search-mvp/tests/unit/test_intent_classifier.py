from services.search_service.intent_classifier import classify_query


def test_exchange_rate_intent_ar():
    intent = classify_query("أسعار العملات بما يعادل الجنيه المصري", "ar")
    assert intent.intent == "exchange_rate"
    assert intent.confidence >= 0.75
    assert "exchange_rate" in intent.allowed_doc_types


def test_exchange_rate_blocks_certificate_query_context():
    intent = classify_query("أسعار العملات", "ar")
    assert intent.intent == "exchange_rate"
    assert "شهاد" in intent.negative_terms


def test_certificate_rate_not_exchange_rate():
    intent = classify_query("كم فايده شهادة سنة بالجنيه", "ar")
    assert intent.intent == "certificate_rate"


def test_personal_loan_intent():
    intent = classify_query("قرض شخصي", "ar")
    assert intent.intent == "personal_loan"
    assert "loan" in intent.allowed_doc_types


def test_credit_card_intent():
    intent = classify_query("بطاقات ائتمان", "ar")
    assert intent.intent == "credit_card"


def test_branch_intent():
    intent = classify_query("الفروع", "ar")
    assert intent.intent == "branch_locator"


def test_atm_intent():
    intent = classify_query("الصراف الآلي", "ar")
    assert intent.intent == "atm_locator"


def test_exchange_rate_en():
    intent = classify_query("exchange rates", "en")
    assert intent.intent == "exchange_rate"
