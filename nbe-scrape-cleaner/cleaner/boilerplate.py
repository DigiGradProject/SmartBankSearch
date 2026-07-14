"""Empirically derive boilerplate text from cross-page frequency."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from cleaner.noise import strip_base64_noise

NORMALIZE_SPACE = re.compile(r"\s+")

# Structural tags that are almost never page body content.
SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "path",
    "meta",
    "link",
    "iframe",
}


def normalize_block_text(text: str) -> str:
    cleaned = strip_base64_noise(text or "")
    cleaned = cleaned.replace("\xa0", " ")
    cleaned = NORMALIZE_SPACE.sub(" ", cleaned).strip().lower()
    return cleaned


def build_boilerplate_set(
    pages_blocks: Iterable[list[str]],
    min_page_fraction: float = 0.55,
    min_absolute: int = 40,
) -> set[str]:
    """
    A normalized line is boilerplate if it appears on enough distinct pages.

    Frequency is computed over unique (page, line) pairs so repeated navbar
    items on one page do not inflate the count.
    """
    page_lists = list(pages_blocks)
    page_count = len(page_lists)
    if page_count == 0:
        return set()

    counter: Counter[str] = Counter()
    for blocks in page_lists:
        unique_on_page = {normalize_block_text(block) for block in blocks if normalize_block_text(block)}
        for line in unique_on_page:
            if len(line) < 2:
                continue
            counter[line] += 1

    threshold = max(min_absolute, int(page_count * min_page_fraction))
    return {line for line, count in counter.items() if count >= threshold}


def is_boilerplate_block(
    tag: str | None,
    text: str,
    boilerplate: set[str],
) -> bool:
    tag_name = (tag or "").lower().strip()
    if tag_name in SKIP_TAGS:
        return True
    normalized = normalize_block_text(text)
    if not normalized:
        return True
    if normalized in boilerplate:
        return True
    # Extremely short UI chrome leftovers after stripping.
    if len(normalized) <= 1:
        return True
    return False


def filter_content_blocks(
    text_blocks: list[dict],
    boilerplate: set[str],
) -> list[dict]:
    kept: list[dict] = []
    seen: set[str] = set()
    for block in text_blocks or []:
        tag = block.get("tag")
        text = strip_base64_noise(block.get("text") or "")
        if is_boilerplate_block(tag, text, boilerplate):
            continue
        tag_name = (tag or "").lower().strip()
        # Prefer body-like tags; keep short anchors only if reasonably long.
        if tag_name in {"a", "button"} and len(normalize_block_text(text)) < 28:
            continue
        normalized = normalize_block_text(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        kept.append({**block, "text": NORMALIZE_SPACE.sub(" ", text.replace("\xa0", " ")).strip()})
    return kept
