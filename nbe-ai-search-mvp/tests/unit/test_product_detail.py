from ingestion.embedding.vector_store import RetrievedChunk
from services.search_service.product_detail import (
    build_product_detail_answer,
    extract_target_product_label,
    prioritize_product_chunks,
)


def _chunk(title: str, text: str, url: str = "https://example.com/product") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        title=title,
        url=url,
        language="ar",
        text=text,
        score=0.9,
    )


SAMPLE_A = """شهادة استثمار ' أ '
فئات الشهادة
500
جم
ومضاعفاتها
المدة
سنوات
10
دورية صرف العائد
فوائد تراكمية
سعر العائد
12%
من يحق لهم الشراء
تصدر الشهادات للأشخاص الطبيعيين المصريين والأجانب
مميزات أخري
العائد مجمع يمنح في نهاية مدة الشهادة
"""

SAMPLE_CATEGORY = """شهادات الاستثمار
الخصائص المشتركة للشهادات
جميع الشهادات اسمية ولا يجوز تداولها
"""


def test_extract_investment_certificate_label():
    q = "ما هي البنك الأهلى المصرى - شهادة استثمار ' أ '؟"
    assert extract_target_product_label(q, "ar") == "شهادة استثمار 'أ'"


def test_build_product_detail_answer_from_named_chunk():
    chunks = [
        _chunk("البنك الأهلى المصرى - شهادات الاستثمار", SAMPLE_CATEGORY),
        _chunk("البنك الأهلى المصرى - شهادة استثمار ' أ '", SAMPLE_A),
    ]
    q = "ما هي شهادة استثمار ' أ '؟"
    result = build_product_detail_answer(q, "ar", chunks)
    assert result is not None
    answer, chunk = result
    assert "12%" in answer
    assert "10" in answer
    assert "فوائد تراكمية" in answer
    assert "استثمار" in chunk.title


def test_prioritize_product_chunk_moves_named_product_first():
    chunks = [
        _chunk("category", SAMPLE_CATEGORY, "https://example.com/category"),
        _chunk("product", SAMPLE_A, "https://example.com/productdetails"),
    ]
    ordered = prioritize_product_chunks("شهادة استثمار ' أ '", "ar", chunks)
    assert ordered[0].title == "product"
