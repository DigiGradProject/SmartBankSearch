"""Extract PDF links from scrape and build documents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote

from shared.schemas import Document, DocumentMetadata

PDF_URL = re.compile(r"https?://[^\s\"'<>]+\.pdf", re.IGNORECASE)


def _language_from_path(path: Path) -> str:
    return "ar" if "AR" in path.parts else "en"


def collect_pdf_links(scrape_root: Path) -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for data_path in scrape_root.glob("**/pages/**/data.json"):
        with data_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        language = _language_from_path(data_path)
        page_url = data.get("url") or ""

        for link in data.get("links") or []:
            href = (link.get("href") or "").strip()
            if ".pdf" not in href.lower():
                continue
            title = (link.get("text") or link.get("title") or "").strip()
            _add_pdf(catalog, href, title, language, page_url)

        for block in data.get("text_blocks") or []:
            text = block.get("text") or ""
            for match in PDF_URL.findall(text):
                _add_pdf(catalog, match, "", language, page_url)

        html = data.get("html") or ""
        for match in PDF_URL.findall(html):
            _add_pdf(catalog, match, "", language, page_url)

    return catalog


def _add_pdf(
    catalog: dict[str, dict],
    url: str,
    title: str,
    language: str,
    source_page: str,
) -> None:
    url = unquote(url.strip())
    entry = catalog.setdefault(
        url,
        {"title": "", "languages": set(), "source_pages": set()},
    )
    if title and (not entry["title"] or len(title) > len(entry["title"])):
        entry["title"] = title
    entry["languages"].add(language)
    if source_page:
        entry["source_pages"].add(source_page)


def _pdf_id(url: str) -> str:
    slug = url.rsplit("/", 1)[-1].replace(".pdf", "")
    return f"PDF_{slug[:40]}"


def pdf_catalog_to_documents(catalog: dict[str, dict], texts: dict[str, str]) -> list[Document]:
    documents: list[Document] = []
    for url, meta in catalog.items():
        content = texts.get(url, "").strip()
        if len(content) < 40:
            continue
        language = "ar" if "ar" in meta["languages"] else "en"
        title = meta["title"] or _pdf_id(url)
        documents.append(
            Document(
                id=_pdf_id(url),
                title=title,
                url=url,
                language=language,  # type: ignore[arg-type]
                content=content,
                metadata=DocumentMetadata(
                    source_folder="pdf",
                    extra={
                        "content_type": "pdf",
                        "source_pages": sorted(meta["source_pages"])[:5],
                    },
                ),
            )
        )
    return documents
