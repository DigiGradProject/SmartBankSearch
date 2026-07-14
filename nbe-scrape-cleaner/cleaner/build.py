"""Assemble cleaned document content from blocks + tables."""

from __future__ import annotations

import re

from cleaner.boilerplate import filter_content_blocks, normalize_block_text
from cleaner.noise import strip_base64_noise
from cleaner.tables import tables_to_markdown_blocks

HEADING_TAGS = {"h1", "h2", "h3", "h4"}


def _inline_tables(blocks: list[dict], table_markdowns: list[str]) -> str:
    """Place tables after a matching content block when possible; else append."""
    remaining = list(table_markdowns)
    parts: list[str] = []

    for block in blocks:
        text = block.get("text") or ""
        tag = (block.get("tag") or "").lower()
        if tag in HEADING_TAGS:
            parts.append(f"## {text}" if tag != "h1" else f"# {text}")
        else:
            parts.append(text)

        # If a table's first cell text is contained in this block, insert next.
        if remaining:
            probe = normalize_block_text(remaining[0].split("\n", 1)[0])
            # Extract first non-separator cell-ish snippet from markdown header row.
            header_cells = [c.strip() for c in remaining[0].split("\n")[0].strip("|").split("|")]
            header_probe = normalize_block_text(header_cells[0]) if header_cells else ""
            block_norm = normalize_block_text(text)
            if header_probe and header_probe in block_norm:
                parts.append(remaining.pop(0))
            elif probe and probe[:40] and probe[:40] in block_norm:
                parts.append(remaining.pop(0))

    if remaining:
        parts.append("## Tables")
        parts.extend(remaining)

    content = "\n\n".join(part for part in parts if part and part.strip())
    content = strip_base64_noise(content)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    return content


def build_content(text_blocks: list[dict], tables: list[dict], boilerplate: set[str]) -> tuple[str, bool]:
    kept = filter_content_blocks(text_blocks, boilerplate)
    table_mds = tables_to_markdown_blocks(tables)
    content = _inline_tables(kept, table_mds)
    return content, bool(table_mds)
