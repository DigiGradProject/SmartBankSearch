import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

from ingestion.classification.doc_classifier import classify_document
from shared.arabic_normalize import normalize_text
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
    if "<" in content and ">" in content:
        soup = BeautifulSoup(content, "lxml")
        content = soup.get_text("\n", strip=True)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return normalize_text(content, language)


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
                "doc_type": classification.doc_type,
                "category": classification.category,
                "canonical_url_slug": classification.canonical_url_slug,
                "quality_score": classification.quality_score,
            },
        ),
    )


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
    classification = classify_document(
        url=raw.get("url") or "",
        title=raw.get("title") or "",
        content=raw.get("content") or "",
        language=raw.get("language") or "en",
        metadata=meta,
    )
    return {
        "content_hash": meta.get("content_hash"),
        "has_tables": meta.get("has_tables"),
        "extracted_at": meta.get("extracted_at"),
        "duplicate_of": meta.get("duplicate_of"),
        "doc_type": classification.doc_type,
        "category": classification.category,
        "canonical_url_slug": classification.canonical_url_slug,
        "is_stub": classification.is_stub,
        "quality_score": classification.quality_score,
        "source_type": classification.source_type,
    }


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


def load_product_stubs(project_root: Path) -> list[Document]:
    stubs_path = project_root / "data" / "product_stubs.json"
    if not stubs_path.exists():
        return []
    return load_documents_json(stubs_path)
