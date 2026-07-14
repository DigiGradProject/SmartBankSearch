from ingestion.cleaning.cleaner import clean_document_text
from ingestion.chunking.chunker import chunk_document, indexable_chunks
from ingestion.classification.metadata_enricher import enrich_document_metadata
from services.rag.confidence import compute_confidence
from services.rag.response_formatter import wrap_plain_answer
from services.search_service.intent_classifier import classify_query
from shared.schemas import Citation, Document, DocumentMetadata
from ingestion.embedding.vector_store import RetrievedChunk


def test_cleaner_removes_nav_and_placeholders():
    raw = "أهلا بك\n{{placeholder}}\nمحتوى الحساب الجاري الأوراق المطلوبة\nجميع الحقوق محفوظة بنك"
    cleaned = clean_document_text(raw, "ar")
    assert "أهلا بك" not in cleaned
    assert "{{placeholder}}" not in cleaned
    assert "الحساب الجاري" in cleaned


def test_parent_child_chunking():
    document = Document(
        id="doc1",
        title="الحساب الجاري",
        url='https://nbe/#/AR/ProductDetails?inParams={"CategoryID":"CurrentAccountsID"}',
        language="ar",
        content="## الأوراق المطلوبة\nبطاقة هوية\n\n## المميزات\nسحب وإيداع\n",
        metadata=DocumentMetadata(extra={"doc_type": "account", "category": "accounts", "page_type": "product"}),
    )
    chunks = chunk_document(document)
    assert any(c.chunk_level == "parent" for c in chunks)
    children = indexable_chunks(chunks)
    assert children
    assert all(c.chunk_level == "child" for c in children)


def test_metadata_enrichment_product_fields():
    enriched = enrich_document_metadata(
        url='https://nbe/#/AR/ProductDetails?inParams={"CategoryID":"CurrentAccountsID","ProductID":"jar_1"}',
        title="البنك الأهلى - الحساب الجاري",
        content="الأوراق المطلوبة بطاقة رقم قومي",
        language="ar",
        metadata={},
    )
    assert enriched.page_type == "product"
    assert enriched.subcategory == "current_accounts"
    assert enriched.doc_type == "account"
    assert "حساب" in enriched.keywords or "required_documents" in enriched.keywords


def test_new_intents():
    assert classify_query("عروض البنك", "ar").intent == "offers"
    assert classify_query("خدمات الشركات", "ar").intent == "corporate"
    assert classify_query("المشروعات الصغيرة", "ar").intent == "sme"
    assert classify_query("الأسئلة الشائعة", "ar").intent == "faq"


def test_confidence_breakdown():
    chunks = [
        RetrievedChunk("1", "d", "t", "u", "ar", "text", 0.8, doc_type="account"),
        RetrievedChunk("2", "d", "t", "u2", "ar", "text", 0.7, doc_type="account"),
    ]
    breakdown = compute_confidence(chunks, llm_confidence=0.8)
    assert 0.5 <= breakdown.final <= 1.0


def test_wrap_plain_answer_structures_product():
    answer = wrap_plain_answer(
        "الأوراق المطلوبة بطاقة هوية",
        language="ar",
        citations=[Citation(title="الحساب الجاري", url="https://nbe/current")],
        intent="account_open",
    )
    assert "### Product" in answer
    assert "### Source" in answer
