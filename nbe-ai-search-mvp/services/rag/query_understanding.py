"""Unified Query Understanding pipeline for enterprise AI Search.

Language → Intent → Entities → Expansion → Rewriting → Search Query

The original user query is never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass

from services.rag.entity_extractor import ExtractedEntity, extract_entities
from services.rag.intent_fallback import classify_with_fallback
from services.rag.language import detect_language, prepare_query
from services.rag.query_rewrite import rewrite_query
from services.search_service.intent_classifier import QueryIntent
from services.search_service.query_expand import expand_query
from services.search_service.synonyms import expand_with_synonyms
from shared.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class QueryUnderstanding:
    original_query: str
    language: str
    intent: QueryIntent
    entities: list[ExtractedEntity]
    expanded_terms: list[str]
    search_query: str
    normalized_query: str = ""

    def entity_dicts(self) -> list[dict[str, str]]:
        return [{"type": e.type, "value": e.value} for e in self.entities]


def understand_query(query: str, language: str = "auto") -> QueryUnderstanding:
    """Run the full QU pipeline; keep `original_query` unchanged."""
    original = query.strip()
    resolved_language = detect_language(original, language)
    normalized = prepare_query(original, resolved_language)
    intent = classify_with_fallback(normalized, resolved_language)

    entities_result = extract_entities(normalized, resolved_language)
    expand = intent.expand_ar if resolved_language == "ar" else intent.expand_en
    rewritten = rewrite_query(normalized, resolved_language, intent_expand=expand)

    # Capture synonym/expansion delta for explainability.
    expanded_base = expand_with_synonyms(expand_query(normalized, resolved_language), resolved_language)
    expanded_terms: list[str] = []
    for token in expanded_base.replace(",", " ").split():
        if token and token not in normalized and token not in expanded_terms:
            expanded_terms.append(token)

    search_query = prepare_query(rewritten.rewritten, resolved_language)

    logger.info(
        "query_understanding",
        language=resolved_language,
        intent=intent.intent,
        intent_confidence=intent.confidence,
        entity_count=len(entities_result.entities),
        search_query_len=len(search_query),
    )

    return QueryUnderstanding(
        original_query=original,
        language=resolved_language,
        intent=intent,
        entities=list(entities_result.entities),
        expanded_terms=expanded_terms,
        search_query=search_query,
        normalized_query=normalized,
    )