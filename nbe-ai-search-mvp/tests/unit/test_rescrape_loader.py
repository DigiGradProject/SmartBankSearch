import json
from datetime import date

import pytest
from ingestion.document_processing.playwright_clean import clean_playwright_content
from ingestion.document_processing.processor import (
    load_curated_documents,
    load_documents_rescrape_dir,
    load_merged_corpus,
)
from shared.config import settings


def test_clean_playwright_strips_nav_noise():
    raw = (
        "Nab Bar Menu\n\n#CategoryShortLayoutContainer#\n\n"
        "شهادة البلاتينية\n\n#ProductHeaderContainer#\n\nSiteMap\n\nCopyrights"
    )
    cleaned = clean_playwright_content(raw)
    assert "Nab Bar Menu" not in cleaned
    assert "Container#" not in cleaned
    assert "شهادة البلاتينية" in cleaned


def _write_rescrape(path, *, url: str, content: str) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "local-certificates",
                "title": "شهادات الادخار بالعملة المحلية",
                "url": url,
                "language": "ar",
                "content": content,
                "metadata": {
                    "source": "playwright_rescrape",
                    "extracted_at": "2026-07-14T08:39:10+00:00",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_load_rescrape_dir_finds_local_certificates(tmp_path):
    url = 'https://www.nbe.com.eg/#/AR/ProductCategory?CategoryID=LocalCertificatesID'
    _write_rescrape(
        tmp_path / "local.json",
        url=url,
        content=(
            "Nab Bar Menu\n\nشهادات الادخار بالعملة المحلية\n\n"
            "الشهادة البلاتينية ذات العائد المتغير الشهري 19.50%\n\n"
            "الحد الأدنى للشراء 1000 جنيه مصري ومضاعفاتها. "
            "يتم احتساب العائد من يوم العمل التالي للشراء."
        ),
    )

    docs = load_documents_rescrape_dir(tmp_path)

    assert len(docs) == 1
    assert "LocalCertificatesID" in docs[0].url
    assert "19.50%" in docs[0].content


def test_merged_corpus_uses_rescrape_when_cleaned_jsonl_is_missing(
    tmp_path, monkeypatch
):
    output = tmp_path / "output"
    output.mkdir()
    url = 'https://www.nbe.com.eg/#/AR/ProductCategory?CategoryID=LocalCertificatesID'
    _write_rescrape(
        output / "local.json",
        url=url,
        content=(
            "شهادات الادخار بالعملة المحلية وعائدها المعلن. "
            "الشهادة البلاتينية ذات العائد المتغير الشهري 19.50%. "
            "الحد الأدنى للشراء 1000 جنيه مصري ومضاعفاتها."
        ),
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "curated_documents.json").write_text(
        json.dumps(
            [
                {
                    "id": "stub-local-certificates",
                    "title": "صفحة مؤقتة",
                    "url": url,
                    "language": "ar",
                    "content": "محتوى مؤقت لا يحتوي على العائد الحقيقي.",
                    "metadata": {
                        "source": "curated_document",
                        "is_stub": True,
                        "curated_version": "test-v1",
                        "reviewed_at": "2026-08-17",
                        "valid_until": "2099-12-31",
                        "status": "approved",
                        "source_url": url,
                        "doc_type": "certificate_rate",
                        "category": "certificates",
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(settings, "cleaned_jsonl_path", tmp_path / "missing.jsonl")
    monkeypatch.setattr(settings, "rescrape_json_path", output)

    docs = load_merged_corpus(tmp_path)

    assert len(docs) == 1
    assert docs[0].id == "local-certificates"
    assert docs[0].metadata.extra["source"] == "playwright_rescrape"
    assert "19.50%" in docs[0].content


def _write_curated(path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "id": "curated-test",
                    "title": "Curated test",
                    "url": "https://www.nbe.com.eg/test",
                    "language": "en",
                    "content": "Reviewed source content.",
                    "metadata": {
                        "source": "curated_document",
                        "is_stub": True,
                        "curated_version": "v1",
                        "reviewed_at": "2026-08-17",
                        "valid_until": "2026-09-17",
                        "status": "approved",
                        "doc_type": "general",
                        "category": "general",
                        **metadata,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )


def test_curated_documents_preserve_version_and_expire(tmp_path):
    curated_path = tmp_path / "data" / "curated_documents.json"
    _write_curated(curated_path, {})

    active = load_curated_documents(tmp_path, as_of=date(2026, 8, 17))
    expired = load_curated_documents(tmp_path, as_of=date(2026, 9, 18))

    assert len(active) == 1
    assert active[0].metadata.extra["curated_version"] == "v1"
    assert active[0].metadata.extra["reviewed_at"] == "2026-08-17"
    assert expired == []


def test_curated_documents_require_version_metadata(tmp_path):
    curated_path = tmp_path / "data" / "curated_documents.json"
    _write_curated(curated_path, {"curated_version": None})
    payload = json.loads(curated_path.read_text(encoding="utf-8"))
    del payload[0]["metadata"]["curated_version"]
    curated_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="curated_version"):
        load_curated_documents(tmp_path, as_of=date(2026, 8, 17))
