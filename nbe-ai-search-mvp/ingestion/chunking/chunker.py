"""Product-aware parent/child chunking with overlap."""

from __future__ import annotations

import re

from shared.config import settings
from shared.content_hash import compute_content_hash
from shared.schemas import ChunkRecord, Document

TABLE_BLOCK = re.compile(r"\[TABLE\](.*?)\[/TABLE\]", re.DOTALL)
HEADING = re.compile(r"^##\s+(.+)$", re.MULTILINE)
PRODUCT_HEADING = re.compile(
    r"(الأوراق المطلوبة|الاوراق المطلوبة|المستندات|مميزات|الشروط|الرسوم|العائد|الاهليه|الأهلية|"
    r"eligibility|required documents|features|fees|interest|documents)",
    re.I,
)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _split_paragraphs(content: str) -> list[str]:
    sections: list[str] = []
    for table_match in TABLE_BLOCK.finditer(content):
        sections.append(table_match.group(0))
    stripped = TABLE_BLOCK.sub("\n", content)
    for part in re.split(r"\n{2,}", stripped):
        part = part.strip()
        if part:
            sections.append(part)
    return sections


def _window_chunks(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    if _approx_tokens(text) <= max_tokens:
        return [text]

    words = text.split()
    max_words = max(8, max_tokens * 4 // 3)
    overlap_words = max(2, overlap_tokens * 4 // 3)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def _meta(document: Document) -> dict:
    return document.metadata.extra or {}


def _make_chunk(
    document: Document,
    *,
    chunk_index: int,
    text: str,
    chunk_level: str,
    parent_chunk_id: str = "",
    section_heading: str = "",
) -> ChunkRecord:
    extra = _meta(document)
    return ChunkRecord(
        chunk_id=f"{document.id}::{chunk_index}",
        document_id=document.id,
        chunk_index=chunk_index,
        title=document.title,
        url=document.url,
        language=document.language,
        text=text,
        content_hash=compute_content_hash(text),
        doc_type=str(extra.get("doc_type", "general")),
        category=str(extra.get("category", "general")),
        is_stub=bool(extra.get("is_stub", False)),
        canonical_url_slug=str(extra.get("canonical_url_slug", "")),
        quality_score=float(extra.get("quality_score", 0.7)),
        chunk_level=chunk_level,
        parent_chunk_id=parent_chunk_id,
        section_heading=section_heading,
        page_type=str(extra.get("page_type", "web_page")),
        subcategory=str(extra.get("subcategory", "")),
        product_name=str(extra.get("product_name", "")),
        service_name=str(extra.get("service_name", "")),
        document_type=str(extra.get("document_type", "web_page")),
        intent=str(extra.get("intent", "")),
        keywords=str(extra.get("keywords", "")),
        last_updated=str(extra.get("last_updated", "")),
    )


def _iter_sections(content: str) -> list[tuple[str, str]]:
    """Return list of (heading, body) sections."""
    heading_parts = HEADING.split(content)
    if len(heading_parts) <= 1:
        return [("", content.strip())] if content.strip() else []

    sections: list[tuple[str, str]] = []
    current_heading = ""
    # HEADING.split keeps preamble at [0], then heading, body, heading, body...
    preamble = heading_parts[0].strip()
    if preamble:
        sections.append(("", preamble))
    for idx in range(1, len(heading_parts), 2):
        heading = heading_parts[idx].strip()
        body = heading_parts[idx + 1].strip() if idx + 1 < len(heading_parts) else ""
        if heading or body:
            sections.append((heading, body))
            current_heading = heading
    _ = current_heading
    return sections


def chunk_document(document: Document) -> list[ChunkRecord]:
    """
    Parent = section (heading-aware / product-aware).
    Child = overlapping semantic windows used for retrieval.
    Parents are emitted for expansion metadata but typically not indexed.
    """
    parent_max = max(settings.chunk_size_tokens * 2, 800)
    child_max = max(160, min(settings.chunk_size_tokens, 280))
    overlap = max(40, settings.chunk_overlap_tokens)

    records: list[ChunkRecord] = []
    chunk_index = 0

    for heading, body in _iter_sections(document.content):
        section_text = f"## {heading}\n{body}".strip() if heading else body
        if not section_text:
            continue

        # Prefer splitting long sections again on product field markers.
        sub_bodies = [section_text]
        # Extra split on explicit product field markers for long sections.
        if _approx_tokens(section_text) > parent_max:
            marks = list(PRODUCT_HEADING.finditer(section_text))
            if len(marks) >= 2:
                rebuilt: list[str] = []
                for i, match in enumerate(marks):
                    start = match.start()
                    end = marks[i + 1].start() if i + 1 < len(marks) else len(section_text)
                    piece = section_text[start:end].strip()
                    if piece:
                        rebuilt.append(piece)
                preamble = section_text[: marks[0].start()].strip()
                if preamble:
                    rebuilt.insert(0, preamble)
                if rebuilt:
                    sub_bodies = rebuilt

        for sub in sub_bodies:
            parent_id = f"{document.id}::parent::{chunk_index}"
            parent_text = sub if _approx_tokens(sub) <= parent_max else _window_chunks(sub, parent_max, overlap)[0]
            records.append(
                _make_chunk(
                    document,
                    chunk_index=chunk_index,
                    text=parent_text,
                    chunk_level="parent",
                    parent_chunk_id="",
                    section_heading=heading or (PRODUCT_HEADING.search(sub).group(0) if PRODUCT_HEADING.search(sub) else ""),
                )
            )
            parent_index = chunk_index
            chunk_index += 1

            for piece in _split_paragraphs(sub):
                for child_text in _window_chunks(piece, child_max, overlap):
                    normalized = child_text.strip()
                    if not normalized:
                        continue
                    # Prefix heading for retrieval signal.
                    if heading and not normalized.startswith("##"):
                        normalized = f"## {heading}\n{normalized}"
                    records.append(
                        _make_chunk(
                            document,
                            chunk_index=chunk_index,
                            text=normalized,
                            chunk_level="child",
                            parent_chunk_id=f"{document.id}::{parent_index}",
                            section_heading=heading,
                        )
                    )
                    chunk_index += 1

    return records


def indexable_chunks(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    """Prefer children for vector/BM25 indexing; fall back to parents if none."""
    children = [chunk for chunk in chunks if chunk.chunk_level == "child"]
    return children or chunks
