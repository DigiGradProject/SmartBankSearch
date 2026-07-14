"""Deterministic query rewriting for NBE banking search.

Optional LLM rewrite is feature-flagged and used only when retrieval is weak.
See docs/enterprise-model-decisions.md Task 4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from services.rag.entity_extractor import EntityExtractionResult, extract_entities
from services.search_service.query_expand import expand_query
from services.search_service.synonyms import expand_with_synonyms
from shared.arabic_normalize import normalize_arabic
from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)

COLLOQUIAL_MAP = [
    (re.compile(r"\bعايز\b|\bعاوز\b", re.I), "أريد"),
    (re.compile(r"\bازاي\b|\bإزاي\b", re.I), "كيف"),
    (re.compile(r"\bبكام\b", re.I), "كم سعر"),
    (re.compile(r"\bفايده\b|\bفايدة\b", re.I), "عائد"),
    (re.compile(r"\bاوراق\b|\bأوراق\b", re.I), "الأوراق المطلوبة"),
]


@dataclass(frozen=True)
class RewriteResult:
    original: str
    rewritten: str
    language: str
    entities: EntityExtractionResult
    used_llm: bool = False


def _rule_rewrite(query: str, language: str, intent_expand: str) -> tuple[str, EntityExtractionResult]:
    text = query.strip()
    if language == "ar":
        text = normalize_arabic(text)
        for pattern, replacement in COLLOQUIAL_MAP:
            text = pattern.sub(replacement, text)

    entities = extract_entities(text, language)
    expanded = expand_query(text, language)
    expanded = expand_with_synonyms(expanded, language)
    if intent_expand and intent_expand not in expanded:
        expanded = f"{expanded} {intent_expand}".strip()

    hints: list[str] = []
    hints.extend(entities.currencies[:2])
    hints.extend(entities.tenors[:2])
    if hints:
        expanded = f"{expanded} {' '.join(hints)}".strip()
    return expanded, entities


async def _llm_rewrite(query: str, language: str) -> str | None:
    if not settings.llm_rewrite_enabled:
        return None
    prompt = (
        "Rewrite the bank customer query into formal Modern Standard Arabic or clear English "
        "banking terminology. Return ONLY the rewritten query.\n"
        f"Language: {language}\nQuery: {query}"
    )
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/generate",
                json={
                    "model": settings.ollama_rewrite_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
            )
            response.raise_for_status()
            text = (response.json().get("response") or "").strip()
            return text.splitlines()[0].strip() if text else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("llm_rewrite_failed", error=str(exc))
        return None


def rewrite_query(
    query: str,
    language: str,
    *,
    intent_expand: str = "",
) -> RewriteResult:
    expanded, entities = _rule_rewrite(query, language, intent_expand)
    return RewriteResult(
        original=query,
        rewritten=expanded,
        language=language,
        entities=entities,
        used_llm=False,
    )


async def rewrite_query_async(
    query: str,
    language: str,
    *,
    intent_expand: str = "",
    retrieval_hit_count: int | None = None,
) -> RewriteResult:
    base = rewrite_query(query, language, intent_expand=intent_expand)
    should_try_llm = (
        settings.llm_rewrite_enabled
        and retrieval_hit_count is not None
        and retrieval_hit_count < 3
    )
    if not should_try_llm:
        return base
    llm_text = await _llm_rewrite(query, language)
    if not llm_text:
        return base
    expanded, entities = _rule_rewrite(llm_text, language, intent_expand)
    return RewriteResult(
        original=query,
        rewritten=expanded,
        language=language,
        entities=entities,
        used_llm=True,
    )
