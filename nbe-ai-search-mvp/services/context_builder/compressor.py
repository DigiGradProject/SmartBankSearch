"""Context compression before LLM packing.

Top-K → dedupe → strip nav/boilerplate → extract relevant paragraphs → merge overlaps.
"""

from __future__ import annotations

import re
from dataclasses import replace

from ingestion.embedding.vector_store import RetrievedChunk
from shared.config import settings
from shared.document_quality import is_menu_heavy_text
from shared.logging import get_logger
from shared.url_canonical import canonical_url_key

logger = get_logger(__name__)

BOILERPLATE = re.compile(
    r"(جميع الحقوق محفوظة|أهلا\s*بك|welcome\s*to|cookie|subscribe|"
    r"login|تسجيل\s*الدخول|menu|القائمة)",
    re.I,
)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower()), set(b.lower())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _query_terms(query: str) -> set[str]:
    return {t for t in re.split(r"\s+", query.strip().lower()) if len(t) >= 2}


def _paragraph_overlap(text: str, terms: set[str]) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs or not terms:
        return text.strip()
    scored: list[tuple[int, str]] = []
    for para in paragraphs:
        lower = para.lower()
        hits = sum(1 for term in terms if term in lower)
        if hits > 0 and not BOILERPLATE.search(para):
            scored.append((hits, para))
    if not scored:
        cleaned = [p for p in paragraphs if not BOILERPLATE.search(p)]
        return "\n\n".join(cleaned[:3]) if cleaned else text.strip()
    scored.sort(key=lambda item: item[0], reverse=True)
    return "\n\n".join(para for _, para in scored[:4])


def compress_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    max_tokens: int | None = None,
) -> list[RetrievedChunk]:
    """Return compressed, deduplicated chunks ready for ContextBuilder."""
    if not settings.context_compression_enabled or not chunks:
        return chunks

    budget = max_tokens or settings.context_max_tokens
    terms = _query_terms(query)
    kept: list[RetrievedChunk] = []
    seen_keys: set[str] = set()
    used_tokens = 0

    for chunk in chunks:
        text = (chunk.text or "").strip()
        if not text:
            continue
        if is_menu_heavy_text(text):
            continue
        text = BOILERPLATE.sub(" ", text)
        text = _paragraph_overlap(text, terms)
        text = re.sub(r"\s{2,}", " ", text).strip()
        if len(text) < 20:
            continue

        url_key = canonical_url_key(chunk.url or "") or chunk.chunk_id
        heading = getattr(chunk, "section_heading", "") or ""
        dedupe_key = f"{url_key}|{heading[:40]}"

        # Near-duplicate against already kept
        if any(_char_jaccard(text[:400], (k.text or "")[:400]) >= 0.82 for k in kept):
            continue
        if dedupe_key in seen_keys:
            # Merge into previous same-url chunk when overlapping
            for idx, prev in enumerate(kept):
                prev_key = canonical_url_key(prev.url or "") or prev.chunk_id
                if prev_key == url_key:
                    merged = f"{prev.text}\n{text}" if text not in prev.text else prev.text
                    tokens = _approx_tokens(merged)
                    if used_tokens - _approx_tokens(prev.text) + tokens <= budget:
                        used_tokens = used_tokens - _approx_tokens(prev.text) + tokens
                        kept[idx] = replace(prev, text=merged)
                    break
            continue

        tokens = _approx_tokens(text)
        if used_tokens + tokens > budget and kept:
            break
        kept.append(replace(chunk, text=text))
        seen_keys.add(dedupe_key)
        used_tokens += tokens

    logger.info(
        "context_compressed",
        input=len(chunks),
        output=len(kept),
        tokens=used_tokens,
    )
    return kept or chunks[:3]
