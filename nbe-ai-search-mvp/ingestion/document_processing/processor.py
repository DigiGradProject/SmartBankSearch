import json
import re
from pathlib import Path
from datetime import date
from shared.url_canonical import canonical_url_key

from ingestion.classification.doc_classifier import classify_document
from ingestion.classification.metadata_enricher import enrich_document_metadata, enriched_as_extra
from ingestion.cleaning.cleaner import clean_document_text
from ingestion.document_processing.playwright_clean import clean_playwright_content
from shared.document_quality import is_junk_document, strip_site_chrome, title_from_document
from shared.schemas import Document, DocumentMetadata

SKIP_TAGS = {"noscript", "script", "style", "iframe"}
NAV_PATTERNS = re.compile(
    r"^(welcome|logout|view profile|notifications|favorite list|follow up|"
    r"أهلا بك|تسجيل الخروج|عرض الصفحة الشخصية|إشعارات|القائمة المفضلة|المتابعة)$",
    re.IGNORECASE,
)


def _title_from_id(doc_id: str, url: str) -> str:
    return title_from_document(doc_id, url)


def _language_from_path(path: Path) -> str:
    if "AR" in path.parts:
        return "ar"
    return "en"


def _extract_text_blocks(data: dict) -> str:
    lines: list[str] = []
    for block in data.get("text_blocks", []):
        tag = (block.get("tag") or "").lower()
        if tag in SKIP_TAGS:
            continue
        text = (block.get("text") or "").strip()
        if not text:
            continue
        if NAV_PATTERNS.match(text):
            continue
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            lines.append(f"\n## {text}\n")
        elif tag in {"li", "p", "td", "th", "span", "a", "button", "div"}:
            if len(text) > 2:
                lines.append(text)
    return "\n".join(lines)


def _extract_tables(data: dict) -> str:
    tables = data.get("tables") or []
    rendered: list[str] = []
    for table in tables:
        rows = table.get("rows") or []
        if not rows:
            continue
        rendered.append("\n[TABLE]")
        for row in rows:
            cells = [str(cell).strip() for cell in row if str(cell).strip()]
            if cells:
                rendered.append(" | ".join(cells))
        rendered.append("[/TABLE]\n")
    return "\n".join(rendered)


def clean_document_content(content: str, language: str) -> str:
    return clean_document_text(content, language)


def process_scrape_file(path: Path) -> Document | None:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    url = data.get("url") or ""
    if not url:
        return None

    language = _language_from_path(path)
    folder_name = path.parent.name
    if is_junk_document(folder_name):
        return None

    doc_id = folder_name
    title = _title_from_id(doc_id, url)

    body = _extract_text_blocks(data)
    tables = _extract_tables(data)
    content = f"{body}\n\n{tables}".strip()
    content = strip_site_chrome(content)
    content = clean_document_content(content, language)

    if len(content) < 80:
        return None

    classification = classify_document(
        url=url,
        title=title,
        content=content,
        language=language,
        metadata={},
    )
    enriched = enrich_document_metadata(
        url=url,
        title=title,
        content=content,
        language=language,
        metadata={},
    )

    return Document(
        id=doc_id,
        title=title,
        url=url,
        language=language,  # type: ignore[arg-type]
        content=content,
        metadata=DocumentMetadata(
            path=str(path.parent),
            source_folder=folder_name,
            block_count=len(data.get("text_blocks") or []),
            extra={
                **enriched_as_extra(enriched),
                "doc_type": classification.doc_type,
                "category": classification.category,
                "canonical_url_slug": classification.canonical_url_slug,
                "quality_score": classification.quality_score,
            },
        ),
    )


def _canonical_url_key(url: str) -> str:
    return canonical_url_key(url)


def merge_documents_by_url(base: list[Document], overrides: list[Document]) -> list[Document]:
    """Override base documents when the same canonical URL exists in overrides."""
    merged: dict[str, Document] = {_canonical_url_key(doc.url): doc for doc in base}
    for doc in overrides:
        merged[_canonical_url_key(doc.url)] = doc
    return list(merged.values())


def load_documents_rescrape_dir(
    directory: Path,
    *,
    min_chars: int = 120,
) -> list[Document]:
    """Load Playwright re-scrape JSON files from scarp/output."""
    documents: list[Document] = []
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        content = clean_playwright_content((raw.get("content") or "").strip())
        content = clean_document_text(content, raw.get("language") or "ar")
        if len(content) < min_chars:
            continue
        meta = raw.get("metadata") or {}
        extra = _enrich_document_metadata(
            {
                "url": raw.get("url") or "",
                "title": raw.get("title") or "",
                "language": raw.get("language") or "ar",
                "content": content,
                "metadata": {
                    **meta,
                    "source": "playwright_rescrape",
                    "source_path": str(path),
                },
            }
        )
        extra["quality_score"] = min(1.0, float(extra.get("quality_score", 0.7)) + 0.15)
        documents.append(
            Document(
                id=raw["id"],
                title=raw.get("title") or "Untitled",
                url=raw["url"],
                language=raw["language"],
                content=content,
                metadata=DocumentMetadata(
                    path=str(path),
                    source_folder="playwright_rescrape",
                    extra=extra,
                ),
            )
        )
    return documents


def load_merged_corpus(project_root: Path, limit: int | None = None) -> list[Document]:
    """Merge all available sources, preferring real content over curated fallbacks.

    The cleaned JSONL is optional in local and MVP environments. When it is
    unavailable, Playwright re-scrapes become the primary corpus instead of
    making the whole ingestion run fail. Curated documents are merged last as
    coverage fallbacks and never replace a real document with the same URL.
    """
    from shared.config import settings

    documents: list[Document] = []
    if settings.cleaned_jsonl_path.exists():
        documents = load_documents_jsonl(settings.cleaned_jsonl_path, limit=None)

    if settings.rescrape_json_path.exists():
        rescrape_docs = load_documents_rescrape_dir(settings.rescrape_json_path)
        documents = merge_documents_by_url(documents, rescrape_docs)

    # Curated fallbacks provide missing coverage only. Real documents win on
    # canonical URL collisions because they are passed as overrides.
    documents = merge_documents_by_url(load_curated_documents(project_root), documents)

    if limit:
        documents = documents[:limit]
    return documents


def load_documents_from_scrape(scrape_root: Path, limit: int | None = None) -> list[Document]:
    documents: list[Document] = []
    for data_path in sorted(scrape_root.glob("**/pages/**/data.json")):
        doc = process_scrape_file(data_path)
        if doc:
            documents.append(doc)
        if limit and len(documents) >= limit:
            break
    return documents


def load_documents_json(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    documents: list[Document] = []
    for item in payload:
        doc = Document.model_validate(item)
        meta = item.get("metadata") or {}
        classification = classify_document(
            url=doc.url,
            title=doc.title,
            content=doc.content,
            language=doc.language,
            metadata=meta,
        )
        preserved = {
            key: value
            for key, value in meta.items()
            if key not in {"path", "source_folder", "block_count", "extra"}
        }
        doc.metadata.extra.update(preserved)
        doc.metadata.extra.update(
            {
                "doc_type": classification.doc_type,
                "category": classification.category,
                "canonical_url_slug": classification.canonical_url_slug,
                "is_stub": classification.is_stub,
                "quality_score": classification.quality_score,
                "source_type": classification.source_type,
            }
        )
        documents.append(doc)
    return documents


def _enrich_document_metadata(raw: dict) -> dict[str, object]:
    meta = raw.get("metadata") or {}
    enriched = enrich_document_metadata(
        url=raw.get("url") or "",
        title=raw.get("title") or "",
        content=raw.get("content") or "",
        language=raw.get("language") or "en",
        metadata=meta,
    )
    payload = enriched_as_extra(enriched)
    payload.update(
        {
            "content_hash": meta.get("content_hash"),
            "has_tables": meta.get("has_tables"),
            "extracted_at": meta.get("extracted_at"),
            "duplicate_of": meta.get("duplicate_of"),
            "source": meta.get("source"),
            "source_path": meta.get("source_path"),
        }
    )
    return payload


def load_documents_jsonl(
    path: Path,
    limit: int | None = None,
    *,
    skip_duplicates: bool = True,
    min_chars: int = 120,
) -> list[Document]:
    """Load RAG-ready documents produced by nbe-scrape-cleaner."""
    documents: list[Document] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            meta = raw.get("metadata") or {}
            if skip_duplicates and meta.get("duplicate_of"):
                continue
            content = (raw.get("content") or "").strip()
            content = clean_document_text(content, raw.get("language") or "ar")
            if len(content) < min_chars:
                continue
            extra = _enrich_document_metadata(raw)
            documents.append(
                Document(
                    id=raw["id"],
                    title=raw.get("title") or "Untitled",
                    url=raw["url"],
                    language=raw["language"],
                    content=content,
                    metadata=DocumentMetadata(
                        path=meta.get("source_path"),
                        source_folder="cleaned_jsonl",
                        extra=extra,
                    ),
                )
            )
            if limit and len(documents) >= limit:
                break
    return documents


CURATED_REQUIRED_METADATA = {
    "source",
    "is_stub",
    "curated_version",
    "reviewed_at",
    "valid_until",
    "status",
}


def load_curated_documents(
    project_root: Path,
    *,
    as_of: date | None = None,
) -> list[Document]:
    curated_path = project_root / "data" / "curated_documents.json"
    if not curated_path.exists():
        return []
    with curated_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    effective_date = as_of or date.today()
    active_ids: set[str] = set()
    for item in payload:
        metadata = item.get("metadata") or {}
        missing = CURATED_REQUIRED_METADATA - metadata.keys()
        if missing:
            raise ValueError(
                f"Curated document {item.get('id', '<unknown>')} missing metadata: "
                f"{', '.join(sorted(missing))}"
            )
        if metadata["source"] != "curated_document" or metadata["status"] != "approved":
            continue
        if date.fromisoformat(str(metadata["valid_until"])) < effective_date:
            continue
        active_ids.add(str(item["id"]))
    return [doc for doc in load_documents_json(curated_path) if doc.id in active_ids]
