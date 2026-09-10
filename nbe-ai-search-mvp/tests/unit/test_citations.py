from ingestion.embedding.vector_store import RetrievedChunk
from services.context_builder.builder import ContextBuilder
from shared.document_quality import is_junk_document, title_from_document


def test_junk_document_detection():
    assert is_junk_document("AR_newcontent")
    assert is_junk_document("EN_testqc")
    assert not is_junk_document("EN_AccountsFAQs")


def test_title_from_product_page():
    doc_id = 'AR_ProductDetails_inParams={_CategoryID___selfservices_,_ProductID___انضم لعملاء البنك الأهلي المصري_18782_}'
    title = title_from_document(doc_id, "https://example.com/#/AR/ProductDetails")
    assert "انضم لعملاء البنك الأهلي المصري" in title


def test_citations_exclude_junk_and_limit_count():
    builder = ContextBuilder()
    chunks = [
        RetrievedChunk("1", "AR_newcontent", "newcontent", "https://x/1", "ar", "menu - a - b - c - d - e - f - g", 0.9),
        RetrievedChunk("2", "AR_AccountsFAQs", "Accounts FAQs", "https://x/2", "ar", "كيفية فتح حساب بنكي", 0.7),
        RetrievedChunk("3", "AR_Accounts", "Accounts", "https://x/3", "ar", "تفاصيل الحسابات الجارية", 0.65),
        RetrievedChunk("4", "AR_CreditCards", "Credit Cards", "https://x/4", "ar", "بطاقات الائتمان", 0.5),
    ]
    built = builder.build("فتح حساب", chunks)
    urls = [c.url for c in built.citations]
    assert "https://x/1" not in urls
    assert len(built.citations) <= 3


def test_broken_productdetails_url_blocked():
    from shared.document_quality import is_broken_citation_url

    bad = "https://www.nbe.com.eg/NBE/E/#/AR/ProductDetails?inParams={\"CategoryID\":\"beladyoneyear\",\"ProductID\":\"Belady USD_16615\"}"
    good = "https://www.nbe.com.eg/NBE/E/#/AR/ProductCategory?inParams={\"CategoryID\":\"beladyoneyear\"}"
    assert is_broken_citation_url(bad)
    assert not is_broken_citation_url(good)


def test_al_ahly_points_query_only_cites_official_program_page():
    builder = ContextBuilder()
    chunks = [
        RetrievedChunk("fx", "fx", "Exchange Rates", "https://nbe/#/EN/ExchangeRatesAndCurrencyConverter", "en", "Al Ahly Points text", 1.0),
        RetrievedChunk("points", "points", "Al Ahly Points", "https://nbe/#/EN/ProductCategory?CategoryID=ahlypoints", "en", "Loyalty program details and redemption", 0.95),
    ]
    built = builder.build("al ahly points", chunks, compress=False)
    assert [citation.url for citation in built.citations] == [chunks[1].url]
