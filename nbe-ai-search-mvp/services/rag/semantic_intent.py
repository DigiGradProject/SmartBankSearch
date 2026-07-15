"""BGE-M3 semantic intent classification — primary intent layer."""

from __future__ import annotations

from functools import lru_cache

from ingestion.embedding.bge_m3 import get_embedder
from services.rag.intent_prototypes import INTENT_CATEGORIES, INTENT_PROTOTYPES
from services.search_service.intent_classifier import INTENT_RULES, INTENT_DOC_TYPES, QueryIntent
from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@lru_cache(maxsize=1)
def _prototype_index() -> tuple[list[str], list[list[float]]]:
    """Pre-embed all intent prototypes once (BGE-M3)."""
    labels: list[str] = []
    texts: list[str] = []
    for intent_name, lang_map in INTENT_PROTOTYPES.items():
        for lang_texts in lang_map.values():
            for text in lang_texts:
                labels.append(intent_name)
                texts.append(text)
    if not texts:
        return [], []
    vectors = get_embedder().embed_dense(texts)
    logger.info("semantic_intent_prototypes_loaded", count=len(texts), intents=len(INTENT_PROTOTYPES))
    return labels, vectors


def _rule_metadata(intent_name: str) -> tuple[str, str, tuple[str, ...]]:
    """Pull expand/negative metadata from regex rules for semantic hits."""
    for rule in INTENT_RULES:
        if rule.intent == intent_name:
            return rule.expand_ar, rule.expand_en, rule.negative_terms
    return "", "", ()


def classify_semantic(query: str, language: str) -> QueryIntent:
    """Classify intent by comparing query embedding to BGE-M3 prototypes."""
    labels, vectors = _prototype_index()
    if not labels:
        return QueryIntent(
            intent="general_faq",
            category="general",
            confidence=0.0,
            allowed_doc_types=INTENT_DOC_TYPES["general_faq"],
            source="semantic",
        )

    try:
        query_vec = get_embedder().embed_dense([query])[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic_intent_embed_failed", error=str(exc))
        return QueryIntent(
            intent="general_faq",
            category="general",
            confidence=0.0,
            allowed_doc_types=INTENT_DOC_TYPES["general_faq"],
            source="semantic",
        )

    best_intent = "general_faq"
    best_score = 0.0
    per_intent: dict[str, float] = {}

    for label, proto_vec in zip(labels, vectors):
        score = _cosine(query_vec, proto_vec)
        per_intent[label] = max(per_intent.get(label, 0.0), score)

    if per_intent:
        best_intent, best_score = max(per_intent.items(), key=lambda item: item[1])

    if best_score < settings.semantic_intent_min_score:
        return QueryIntent(
            intent="general_faq",
            category="general",
            confidence=best_score,
            allowed_doc_types=INTENT_DOC_TYPES["general_faq"],
            source="semantic",
        )

    expand_ar, expand_en, negative_terms = _rule_metadata(best_intent)
    category = INTENT_CATEGORIES.get(best_intent, "general")
    allowed = INTENT_DOC_TYPES.get(best_intent, INTENT_DOC_TYPES["general_faq"])

    return QueryIntent(
        intent=best_intent,
        category=category,
        confidence=round(min(0.98, best_score), 3),
        allowed_doc_types=allowed,
        expand_ar=expand_ar,
        expand_en=expand_en,
        negative_terms=negative_terms,
        source="semantic",
    )
