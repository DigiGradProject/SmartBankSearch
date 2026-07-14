from ingestion.chunking.chunker import chunk_document
from shared.schemas import Document, DocumentMetadata


def test_chunk_document_preserves_table_block():
    document = Document(
        id="EN_Test",
        title="Test",
        url="https://example.com",
        language="en",
        content="Intro paragraph.\n\n[TABLE]\nFee | 10 EGP\nLimit | 5000 EGP\n[/TABLE]\n\nClosing paragraph.",
        metadata=DocumentMetadata(),
    )
    chunks = chunk_document(document)
    assert len(chunks) >= 2
    assert any("[TABLE]" in chunk.text for chunk in chunks)
