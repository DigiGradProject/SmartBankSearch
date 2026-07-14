"""Build related search suggestions when no confident answer."""

from __future__ import annotations

import re

from ingestion.embedding.vector_store import RetrievedChunk
from services.search_service.keyword_rank import extract_query_terms, keyword_overlap_score
from services.search_service.query_catalog import TOPIC_QUERIES_AR, TOPIC_QUERIES_EN
from services.search_service.query_expand import expand_query
from services.search_service.rate_guidance import rate_suggestions
from shared.schemas import SearchSuggestion


def _topic_suggestions(query: str, language: str) -> list[SearchSuggestion]:
    catalog = TOPIC_QUERIES_AR if language == "ar" else TOPIC_QUERIES_EN
    haystack = query.lower()
    suggestions: list[SearchSuggestion] = []
    for key, suggested_query in catalog.items():
        if key in haystack:
            suggestions.append(
                SearchSuggestion(
                    query=suggested_query,
                    label=suggested_query,
                    reason="topic_match",
                )
            )
    return suggestions


def _chunk_suggestions(chunks: list[RetrievedChunk], query: str, language: str) -> list[SearchSuggestion]:
    terms = extract_query_terms(expand_query(query, language), language)
    suggestions: list[SearchSuggestion] = []

    for chunk in chunks[:8]:
        overlap = keyword_overlap_score(chunk.text, terms, language) if terms else 0.0
        if chunk.score < 0.18 and overlap < 0.2:
            continue
        label = chunk.title.strip() or chunk.url
        suggested_query = _query_from_chunk(chunk, language)
        suggestions.append(
            SearchSuggestion(
                query=suggested_query,
                label=label,
                url=chunk.url,
                reason="related_content",
                score=round(max(chunk.score, overlap), 3),
            )
        )
    return suggestions


def _query_from_chunk(chunk: RetrievedChunk, language: str) -> str:
    title = chunk.title.strip()
    if language == "ar":
        if len(title) >= 4 and re.search(r"[\u0600-\u06FF]", title):
            return f"ما هي {title}؟"
        return "شهادات الادخار في البنك الأهلي"
    if title:
        return f"What is {title}?"
    return "NBE certificates and services"


def build_suggestions(
    query: str,
    language: str,
    chunks: list[RetrievedChunk],
    limit: int = 5,
) -> list[SearchSuggestion]:
    merged: list[SearchSuggestion] = []
    seen: set[str] = set()

    rate_items = rate_suggestions(query, language)
    for item in rate_items:
        key = (item.url or item.query).lower().strip()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    expanded = expand_query(query, language)
    if not rate_items and expanded.strip() != query.strip():
        key = expanded.lower()
        if key not in seen:
            merged.append(
                SearchSuggestion(
                    query=expanded,
                    label=expanded,
                    reason="expanded_query",
                )
            )
            seen.add(key)

    for item in _topic_suggestions(query, language) + _chunk_suggestions(chunks, query, language):
        key = (item.url or item.query).lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)

    merged.sort(key=lambda item: item.score or 0.0, reverse=True)
    return merged[:limit]
