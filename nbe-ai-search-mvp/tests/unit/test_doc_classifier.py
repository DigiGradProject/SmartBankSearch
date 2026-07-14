from ingestion.classification.doc_classifier import classify_document


def test_classify_exchange_rate_page():
    result = classify_document(
        url="https://www.nbe.com.eg/NBE/E/#/AR/ExchangeRatesAndCurrencyConverter",
        title="سعر الصرف و تحويل العملات",
        content="دولار امريكي شراء 49.51 بيع 49.61",
        language="ar",
    )
    assert result.doc_type == "exchange_rate"
    assert result.category == "exchange_rates"


def test_classify_certificate_rates_page():
    result = classify_document(
        url="https://www.nbe.com.eg/NBE/E/#/AR/CertificatesRatesForeignCurrency",
        title="أسعار الشهادات بالعملة الأجنبية",
        content="شهادات بلادي بالدولار",
        language="ar",
    )
    assert result.doc_type == "certificate_rate"
    assert result.category == "certificates"


def test_classify_loan_page():
    result = classify_document(
        url="https://www.nbe.com.eg/NBE/E/#/AR/Loans",
        title="القروض",
        content="قرض شخصي",
        language="ar",
    )
    assert result.doc_type == "loan"
