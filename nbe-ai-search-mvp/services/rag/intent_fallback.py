"""Layered intent classification — semantic first, regex fallback, critical always.

Decision layers (enterprise banking):
  1. Critical rules (contact, password) — deterministic, always first
  2. BGE-M3 semantic intent — primary for generalization
  3. Regex rules — fallback when semantic confidence is in the middle band
  4. general_faq — broad hybrid retrieval (no intent filter)

Legacy mode (semantic_intent_enabled=False): regex primary, optional MiniLM fallback.
"""

from __future__ import annotations

from functools import lru_cache

from services.rag.critical_rules import classify_critical
from services.rag.semantic_intent import classify_semantic
from services.search_service.intent_classifier import INTENT_DOC_TYPES, QueryIntent, classify_query
from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)

# Re-export prototypes for tests/docs
from services.rag.intent_prototypes import INTENT_PROTOTYPES  # noqa: E402

__all__ = ["classify_with_fallback", "classify_layered", "INTENT_PROTOTYPES"]


@lru_cache(maxsize=1)
def _minilm_model():
    from sentence_transformers import SentenceTransformer

    logger.info("loading_minilm_intent_model", model=settings.minilm_intent_model)
    return SentenceTransformer(settings.minilm_intent_model)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _classify_minilm_fallback(query: str, language: str, rule_intent: QueryIntent) -> QueryIntent:
    """Legacy MiniLM path when semantic layer is disabled."""
    if not settings.minilm_intent_fallback_enabled:
        return rule_intent
    try:
        model = _minilm_model()
        prototypes: list[str] = []
        labels: list[str] = []
        for intent_name, texts in INTENT_PROTOTYPES.items():
            lang_texts = texts.get(language) or texts.get("en") or ()
            if not lang_texts:
                continue
            labels.append(intent_name)
            prototypes.append(lang_texts[0])
        if not prototypes:
            return rule_intent
        vectors = model.encode([query, *prototypes], normalize_embeddings=True)
        query_vec = vectors[0].tolist()
        best_label = "general_faq"
        best_score = 0.0
        for label, proto in zip(labels, vectors[1:]):
            score = _cosine(query_vec, proto.tolist())
            if score > best_score:
                best_score = score
                best_label = label
        if best_score < 0.45:
            return rule_intent
        from services.rag.intent_prototypes import INTENT_CATEGORIES

        category = INTENT_CATEGORIES.get(best_label, "general")
        return QueryIntent(
            intent=best_label,
            category=category,
            confidence=min(0.74, best_score),
            allowed_doc_types=INTENT_DOC_TYPES.get(best_label, INTENT_DOC_TYPES["general_faq"]),
            source="minilm",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("minilm_intent_fallback_failed", error=str(exc))
        return rule_intent


def classify_layered(query: str, language: str) -> QueryIntent:
    """Semantic-first layered intent (BGE-M3 → regex fallback → broad)."""
    critical = classify_critical(query, language)
    if critical is not None:
        logger.info("intent_critical", intent=critical.intent, confidence=critical.confidence)
        return critical

    semantic = classify_semantic(query, language)
    rule = classify_query(query, language)

    high = settings.semantic_intent_high_confidence
    mid = settings.semantic_intent_regex_fallback_threshold

    if semantic.confidence >= high:
        logger.info(
            "intent_semantic_route",
            intent=semantic.intent,
            confidence=semantic.confidence,
            layer="high",
        )
        return semantic

    if semantic.confidence >= mid:
        if rule.intent != "general_faq":
            logger.info(
                "intent_regex_confirm",
                intent=rule.intent,
                semantic_confidence=semantic.confidence,
                layer="mid_regex",
            )
            return rule
        logger.info(
            "intent_semantic_route",
            intent=semantic.intent,
            confidence=semantic.confidence,
            layer="mid_semantic",
        )
        return semantic

    if rule.intent != "general_faq":
        logger.info(
            "intent_regex_fallback",
            intent=rule.intent,
            semantic_confidence=semantic.confidence,
            layer="low_regex",
        )
        return rule

    if semantic.confidence >= settings.semantic_intent_min_score:
        logger.info(
            "intent_semantic_weak",
            intent=semantic.intent,
            confidence=semantic.confidence,
            layer="low_semantic",
        )
        return semantic

    logger.info("intent_broad_hybrid", semantic_confidence=semantic.confidence)
    return QueryIntent(
        intent="general_faq",
        category="general",
        confidence=0.0,
        allowed_doc_types=INTENT_DOC_TYPES["general_faq"],
        source="general",
    )


def classify_with_fallback(query: str, language: str) -> QueryIntent:
    """Main intent entry point used by query understanding."""
    if settings.semantic_intent_enabled:
        return classify_layered(query, language)

    # Legacy: regex primary
    rule_intent = classify_query(query, language)
    if rule_intent.intent != "general_faq":
        return rule_intent
    return _classify_minilm_fallback(query, language, rule_intent)
