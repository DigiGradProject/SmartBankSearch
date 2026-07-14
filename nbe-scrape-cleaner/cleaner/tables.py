"""Convert scraped HTML tables into cleaned markdown tables."""

from __future__ import annotations

import re

from cleaner.noise import strip_base64_noise

WHITESPACE = re.compile(r"\s+")


def clean_cell(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\xa0", " ")
    text = strip_base64_noise(text)
    text = WHITESPACE.sub(" ", text).strip()
    text = text.replace("|", "\\|")
    return text


def rows_to_markdown(rows: list[list[object]]) -> str:
    cleaned_rows: list[list[str]] = []
    for row in rows or []:
        cells = [clean_cell(cell) for cell in row]
        if any(cells):
            cleaned_rows.append(cells)
    if not cleaned_rows:
        return ""

    width = max(len(row) for row in cleaned_rows)
    normalized = [row + [""] * (width - len(row)) for row in cleaned_rows]
    header = normalized[0]
    body = normalized[1:] if len(normalized) > 1 else []

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def tables_to_markdown_blocks(tables: list[dict]) -> list[str]:
    blocks: list[str] = []
    for table in tables or []:
        md = rows_to_markdown(table.get("rows") or [])
        if md:
            blocks.append(md)
    return blocks
