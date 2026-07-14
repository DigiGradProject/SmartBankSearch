from ingestion.embedding.vector_store import RetrievedChunk
from services.search_service.account_rank import prioritize_account_chunks


def _chunk(url: str, title: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=url,
        document_id=url,
        title=title,
        url=url,
        language="ar",
        text=text,
        score=score,
        doc_type="account",
        category="accounts",
    )


def test_retail_query_prefers_current_over_initiative():
    chunks = [
        _chunk(
            "https://nbe/#/AR/ProductCategory?inParams={\"CategoryID\":\"OpenYourBankAccountInEgypt\"}",
            "مبادرة افتح حسابك في مصر",
            "الأوراق المطلوبة جواز سفر بطاقة رقم قومي سفارة",
            0.95,
        ),
        _chunk(
            "https://nbe/#/AR/ProductDetails?inParams={\"CategoryID\":\"CurrentAccountsID\",\"ProductID\":\"jar\"}",
            "الحساب الجاري",
            "الأوراق المطلوبة نسخة من بطاقة الهوية",
            0.70,
        ),
        _chunk(
            "https://nbe/#/AR/ProductCategory?inParams={\"CategoryID\":\"AccountsID\"}",
            "الحسابات",
            "أنواع الحسابات",
            0.80,
        ),
    ]
    ranked = prioritize_account_chunks(
        "لو عايز افتح حساب بنكي اي الاوراق المطلوبة؟",
        "ar",
        chunks,
    )
    assert "CurrentAccountsID" in ranked[0].url
    assert "OpenYourBankAccountInEgypt" not in ranked[0].url


def test_diaspora_query_keeps_initiative():
    chunks = [
        _chunk(
            "https://nbe/#/AR/ProductCategory?inParams={\"CategoryID\":\"OpenYourBankAccountInEgypt\"}",
            "مبادرة افتح حسابك في مصر",
            "سفارة قنصلية الأوراق المطلوبة",
            0.70,
        ),
        _chunk(
            "https://nbe/#/AR/ProductDetails?inParams={\"CategoryID\":\"CurrentAccountsID\"}",
            "الحساب الجاري",
            "الأوراق المطلوبة بطاقة هوية",
            0.90,
        ),
    ]
    ranked = prioritize_account_chunks(
        "مبادرة افتح حسابك في مصر من السفارة",
        "ar",
        chunks,
    )
    assert "OpenYourBankAccountInEgypt" in ranked[0].url
