"""Multi-intent query planner — split conjunction queries into sub-intents."""

from __future__ import annotations

import re
from dataclasses import dataclass

from services.rag.intent_fallback import classify_with_fallback
from services.rag.query_understanding import QueryUnderstanding, understand_query
from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)

# Split on Arabic/English conjunctions, keep max 2 sides.
# Arabic often attaches و to the next word (وطلب) — allow optional space after و.
SPLIT_PATTERN = re.compile(
    r"\s+(?:و|ثم)\s*|\s+(?:and|then|also|&)\s+|،\s*(?:و|and)\s*",
    re.I,
)


@dataclass(frozen=True)
class PlannedSubQuery:
    text: str
    intent_name: str
    understanding: QueryUnderstanding


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    language: str
    subqueries: list[PlannedSubQuery]
    is_multi: bool


def _distinct_intents(parts: list[str], language: str) -> list[tuple[str, str]]:
    """Return (text, intent) pairs with distinct intents, max configured."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for part in parts:
        text = part.strip(" ؟?،,.")
        if len(text) < 4:
            continue
        intent = classify_with_fallback(text, language)
        # Do not turn a weak semantic guess from a sentence fragment into a
        # separate retrieval branch. Colloquial Arabic commonly uses an
        # attached conjunction (for example, "وعايز"), and the text before
        # it may be too vague to carry an intent on its own.
        if (
            intent.source == "semantic"
            and intent.confidence < settings.semantic_intent_regex_fallback_threshold
        ):
            continue
        if intent.intent in seen:
            continue
        if intent.intent == "general_faq" and out:
            continue
        seen.add(intent.intent)
        out.append((text, intent.intent))
        if len(out) >= settings.query_planner_max_intents:
            break
    return out


def plan_query(query: str, language: str = "auto") -> QueryPlan:
    primary = understand_query(query, language)
    if not settings.query_planner_enabled:
        return QueryPlan(
            original_query=primary.original_query,
            language=primary.language,
            subqueries=[
                PlannedSubQuery(primary.original_query, primary.intent.intent, primary)
            ],
            is_multi=False,
        )

    parts = [p.strip() for p in SPLIT_PATTERN.split(primary.original_query) if p.strip()]
    if len(parts) < 2:
        return QueryPlan(
            original_query=primary.original_query,
            language=primary.language,
            subqueries=[
                PlannedSubQuery(primary.original_query, primary.intent.intent, primary)
            ],
            is_multi=False,
        )

    distinct = _distinct_intents(parts, primary.language)
    if len(distinct) < 2:
        return QueryPlan(
            original_query=primary.original_query,
            language=primary.language,
            subqueries=[
                PlannedSubQuery(primary.original_query, primary.intent.intent, primary)
            ],
            is_multi=False,
        )

    subqueries: list[PlannedSubQuery] = []
    for text, intent_name in distinct:
        qu = understand_query(text, primary.language)
        subqueries.append(PlannedSubQuery(text, intent_name, qu))

    logger.info(
        "query_plan_multi",
        intents=[s.intent_name for s in subqueries],
        count=len(subqueries),
    )
    return QueryPlan(
        original_query=primary.original_query,
        language=primary.language,
        subqueries=subqueries,
        is_multi=True,
    )
