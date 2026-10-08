"""Autocomplete suggestions while the user types."""

from __future__ import annotations

import re

from ingestion.embedding.vector_store import RetrievedChunk, VectorStore
from ingestion.lexical.bm25_index import get_bm25_index, tokenize_text
from services.rag.language import detect_language, prepare_query
from services.search_service.keyword_rank import apply_spacing_aliases, extract_query_terms, keyword_overlap_score
from services.search_service.query_catalog import TOPIC_QUERIES_AR, TOPIC_QUERIES_EN, catalog_for_language
from services.search_service.query_expand import expand_query, expand_query_intent
from shared.arabic_normalize import normalize_arabic
from shared.schemas import SearchSuggestion

MIN_QUERY_LENGTH = 2

# Corporate prefix the scraper prepends to most page titles ("National Bank
# of Egypt - Al Ahly points"). Suggestions feed the search box, where the
# prefix is dead weight that dilutes BM25 — keep only the meaningful part.
_TITLE_PREFIXES = (
    "National Bank of Egypt - ",
    "البنك الأهلى المصرى - ",
    "البنك الأهلي المصرى - ",
)


def _clean_title(title: str) -> str:
    for prefix in _TITLE_PREFIXES:
        if title.startswith(prefix):
            cleaned = title[len(prefix) :].strip()
            # Keep the full title when the rest is generic or too short
            # ("Home", "FAQs", "ارشادات") so the suggestion still identifies
            # the page; multi-word remainders are always specific enough.
            is_specific = len(cleaned) >= 4 and (" " in cleaned or "\u0600" <= cleaned[0] <= "\u06FF")
            return cleaned if is_specific else title
    return title


def _normalize_for_match(text: str, language: str) -> str:
    lowered = text.lower().strip()
    if language == "ar":
        return normalize_arabic(lowered)
    return lowered


def _catalog_score(query_norm: str, entry_query: str, entry_keywords: tuple[str, ...], language: str) -> float:
    entry_norm = _normalize_for_match(entry_query, language)
    if not query_norm:
        return 0.0
    if entry_norm.startswith(query_norm):
        return 1.0
    if query_norm in entry_norm:
        return 0.82
    for keyword in entry_keywords:
        keyword_norm = _normalize_for_match(keyword, language)
        if keyword_norm.startswith(query_norm) or query_norm in keyword_norm:
            return 0.72
    terms = extract_query_terms(query_norm, language)
    if terms:
        overlap = keyword_overlap_score(entry_norm, terms, language)
        if overlap >= 0.34:
            return 0.55 + overlap * 0.3
    return 0.0


def _bm25_title_matches(query: str, language: str, limit: int) -> list[SearchSuggestion]:
    # "alahly points" must match pages titled "Al Ahly Points": re-space
    # before BM25 so compound brand words tokenize like the corpus does.
    """Suggest real page titles from the standalone BM25 index (no embedder).

    Traditional mode fetches autocomplete on every keystroke (catalog_only=True)
    so it must never touch the vector store; this source keeps suggestions
    grounded in the actual indexed corpus instead of only the small static
    query catalog.
    """
    index = get_bm25_index()
    if index.size == 0:
        return []

    query_tokens = tokenize_text(query, language)
    if not query_tokens:
        return []

    # Exact BM25 ranking, then fall back to normalized-substring title match
    # for partial/inflected prefixes that tokenize differently.
    scored: list[tuple[str, str, str | None, float]] = []
    for chunk, bm25_score in index.query(apply_spacing_aliases(query), language, top_k=limit * 4):
        title = _clean_title(chunk.title.strip())
        if title:
            # Normalize raw BM25 doc scores into a stable 0..1 band below the
            # static-catalog scores (catalog stays authoritative up top).
            scored.append((title, chunk.language, chunk.url, min(bm25_score, 12.0) / 12.0 * 0.85))

    if not scored:
        query_norm = _normalize_for_match(query, language)
        spacing_norm = _normalize_for_match(apply_spacing_aliases(query), language)
        seen_substring: set[str] = set()
        for chunk in index._chunks:
            title_norm = _normalize_for_match(chunk.title, chunk.language or language)
            matched = (
                (query_norm and query_norm in title_norm)
                or (spacing_norm != query_norm and spacing_norm in title_norm)
            )
            if matched:
                title = _clean_title(chunk.title.strip())
                if title and title not in seen_substring:
                    seen_substring.add(title)
                    # Prefix hits rank above mid-title hits; other-language
                    # titles rank below same-language ones so the dropdown
                    # stays clean unless the corpus is nearly empty.
                    same_language = language == "auto" or chunk.language == language
                    starts = title_norm.startswith(spacing_norm or query_norm)
                    base = 0.35 if starts else 0.25
                    scored.append((title, chunk.language, chunk.url, base if same_language else base - 0.2))

    suggestions: list[SearchSuggestion] = []
    seen: set[str] = set()
    scored.sort(key=lambda item: item[3], reverse=True)
    for title, title_language, url, score in scored:
        if title.lower() in seen:
            continue
        seen.add(title.lower())
        suggestions.append(
            SearchSuggestion(
                query=_query_from_title(title, title_language or language),
                label=title,
                url=url or None,
                reason="bm25_title_match",
                score=round(score, 3),
            )
        )
        if len(suggestions) >= limit:
            break
    return suggestions


def _query_from_title(title: str, language: str) -> str:
    # Suggestion chips feed back into the search box; keep them as raw terms
    # (no "What is ...?" wrapper) so the keyword engine ranks BM25 tokens.
    return title


def _topic_matches(query: str, language: str) -> list[SearchSuggestion]:
    catalog = TOPIC_QUERIES_AR if language == "ar" else TOPIC_QUERIES_EN
    query_norm = _normalize_for_match(query, language)
    suggestions: list[SearchSuggestion] = []
    for key, suggested_query in catalog.items():
        key_norm = _normalize_for_match(key, language)
        if key_norm in query_norm or query_norm in key_norm:
            suggestions.append(
                SearchSuggestion(
                    query=suggested_query,
                    label=suggested_query,
                    reason="topic_match",
                    score=0.9,
                )
            )
    return suggestions


def _catalog_matches(query: str, language: str, limit: int) -> list[SearchSuggestion]:
    query_norm = _normalize_for_match(query, language)
    scored: list[SearchSuggestion] = []
    for entry in catalog_for_language(language):
        score = _catalog_score(query_norm, entry.query, entry.keywords, language)
        if score <= 0:
            continue
        scored.append(
            SearchSuggestion(
                query=entry.query,
                label=entry.label,
                reason="catalog_match",
                score=round(score, 3),
            )
        )
    scored.sort(key=lambda item: item.score or 0.0, reverse=True)
    return scored[:limit]


def _semantic_matches(
    query: str,
    language: str,
    vector_store: VectorStore,
    limit: int,
) -> list[SearchSuggestion]:
    expanded = expand_query(query, language)
    embed_query = prepare_query(expanded, language)
    chunks = vector_store.query(embed_query, max(limit * 2, 6))
    suggestions: list[SearchSuggestion] = []
    seen_titles: set[str] = set()

    for chunk in chunks:
        title = chunk.title.strip()
        if not title or len(title) < 3:
            continue
        title_key = title.lower()
        if title_key in seen_titles:
            continue

        terms = extract_query_terms(query, language)
        overlap = keyword_overlap_score(f"{title} {chunk.text[:240]}", terms, language) if terms else 0.0
        if overlap < 0.18:
            continue

        seen_titles.add(title_key)

        suggested_query = _query_from_chunk(chunk, language)
        suggestions.append(
            SearchSuggestion(
                query=suggested_query,
                label=title,
                url=chunk.url,
                reason="semantic_match",
                score=round(chunk.score, 3),
            )
        )
    return suggestions[:limit]


def _query_from_chunk(chunk: RetrievedChunk, language: str) -> str:
    title = chunk.title.strip()
    if language == "ar":
        if len(title) >= 4 and re.search(r"[\u0600-\u06FF]", title):
            return f"ما هي {title}؟"
        return "شهادات الادخار في البنك الأهلي"
    if title:
        return f"What is {title}?"
    return "NBE certificates and services"


def build_autocomplete(
    query: str,
    language: str = "auto",
    limit: int = 8,
    vector_store: VectorStore | None = None,
    *,
    catalog_only: bool = False,
) -> list[SearchSuggestion]:
    """Suggest queries while typing.

    catalog_only=True skips the vector-store semantic pass — used by the
    traditional-search UX to keep keystroke latency low. The store must be
    created lazily: VectorStore() eagerly builds a Chroma client and loads
    the BGE-M3 embedder, which catalog-only keystrokes must never pay for.
    """
    store = None if catalog_only else (vector_store or VectorStore())
    resolved_language = detect_language(query, language)
    trimmed = query.strip()

    if not trimmed:
        popular = catalog_for_language(resolved_language)[:limit]
        return [
            SearchSuggestion(query=item.query, label=item.label, reason="popular", score=1.0)
            for item in popular
        ]

    if len(trimmed) < MIN_QUERY_LENGTH:
        return []

    merged: list[SearchSuggestion] = []
    seen: set[str] = set()

    expanded = expand_query_intent(trimmed, resolved_language)
    if expanded and expanded.strip() != trimmed:
        merged.append(
            SearchSuggestion(
                query=expanded,
                label=expanded,
                reason="expanded_query",
                score=0.95,
            )
        )
        seen.add(expanded.lower())

    semantic_items = (
        []
        if catalog_only
        else _semantic_matches(trimmed, resolved_language, store, limit)  # type: ignore[arg-type]
    )
    for item in (
        _topic_matches(trimmed, resolved_language)
        + _catalog_matches(trimmed, resolved_language, limit)
        + _bm25_title_matches(trimmed, resolved_language, limit)
        + semantic_items
    ):
        key = item.query.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)

    merged.sort(key=lambda item: item.score or 0.0, reverse=True)
    return merged[:limit]
