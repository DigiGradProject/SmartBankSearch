from ingestion.document_processing.playwright_clean import clean_playwright_content
from ingestion.document_processing.processor import load_documents_rescrape_dir
from pathlib import Path


def test_clean_playwright_strips_nav_noise():
    raw = "Nab Bar Menu\n\nشهادة البلاتينية\n\nSiteMap\n\nCopyrights"
    cleaned = clean_playwright_content(raw)
    assert "Nab Bar Menu" not in cleaned
    assert "شهادة البلاتينية" in cleaned


def test_load_rescrape_dir_finds_local_certificates():
    root = Path(__file__).resolve().parents[2].parent / "output"
    docs = load_documents_rescrape_dir(root)
    assert docs
    local = [doc for doc in docs if "LocalCertificatesID" in doc.url and "%" in doc.content]
    assert local
