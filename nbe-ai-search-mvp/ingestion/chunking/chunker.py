import re

from shared.config import settings
from shared.content_hash import compute_content_hash
from shared.schemas import ChunkRecord, Document

TABLE_BLOCK = re.compile(r"\[TABLE\](.*?)\[/TABLE\]", re.DOTALL)
HEADING = re.compile(r"^##\s+(.+)$", re.MULTILINE)


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
    max_words = max_tokens * 4 // 3
    overlap_words = overlap_tokens * 4 // 3
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


def _make_chunk(document: Document, chunk_index: int, text: str) -> ChunkRecord:
    extra = document.metadata.extra
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
    )


def chunk_document(document: Document) -> list[ChunkRecord]:
    max_tokens = settings.chunk_size_tokens
    overlap_tokens = settings.chunk_overlap_tokens
    records: list[ChunkRecord] = []
    chunk_index = 0

    heading_parts = HEADING.split(document.content)
    if len(heading_parts) > 1:
        current_heading = ""
        for idx, part in enumerate(heading_parts):
            part = part.strip()
            if not part:
                continue
            if idx % 2 == 1:
                current_heading = part
                continue
            section = f"## {current_heading}\n{part}" if current_heading else part
            for piece in _split_paragraphs(section):
                for chunk_text in _window_chunks(piece, max_tokens, overlap_tokens):
                    normalized = chunk_text.strip()
                    records.append(_make_chunk(document, chunk_index, normalized))
                    chunk_index += 1
        return records

    for piece in _split_paragraphs(document.content):
        for chunk_text in _window_chunks(piece, max_tokens, overlap_tokens):
            normalized = chunk_text.strip()
            records.append(_make_chunk(document, chunk_index, normalized))
            chunk_index += 1
    return records
