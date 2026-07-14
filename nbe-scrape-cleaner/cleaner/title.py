"""Extract page titles from HTML or heading-like text blocks."""

from __future__ import annotations

import re
from html import unescape

from cleaner.noise import strip_base64_noise

TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
HEADING_TAGS = {"h1", "h2", "h3"}
GENERIC_TITLES = {
    "",
    "nbe",
    "national bank of egypt",
    "البنك الأهلي المصري",
    "home",
    "الرئيسية",
}


def _clean_title(raw: str) -> str:
    text = unescape(raw or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = strip_base64_noise(text)
    text = re.sub(r"\s+", " ", text).strip()
    # Common "Title | Bank" patterns — keep left side if meaningful.
    if "|" in text:
        left = text.split("|", 1)[0].strip()
        if left and left.lower() not in GENERIC_TITLES:
            text = left
    return text


def extract_title(html: str | None, text_blocks: list[dict], tables: list[dict] | None = None) -> str:
    if html:
        match = TITLE_RE.search(html)
        if match:
            title = _clean_title(match.group(1))
            if title.lower() not in GENERIC_TITLES and len(title) >= 2:
                return title

    for block in text_blocks or []:
        tag = (block.get("tag") or "").lower()
        if tag in HEADING_TAGS:
            title = _clean_title(block.get("text") or "")
            if title.lower() not in GENERIC_TITLES and len(title) >= 2:
                return title

    for block in text_blocks or []:
        title = _clean_title(block.get("text") or "")
        if 3 <= len(title) <= 120 and title.lower() not in GENERIC_TITLES:
            return title

    for table in tables or []:
        rows = table.get("rows") or []
        if not rows:
            continue
        first_row = rows[0]
        if first_row:
            title = _clean_title(str(first_row[0]))
            if 3 <= len(title) <= 120:
                return title

    return "Untitled page"
