"""Optional MiniLM intent fallback when rule confidence is low.

Feature-flagged. Rules remain primary for bank auditability.
"""

from __future__ import annotations

from functools import lru_cache

from services.search_service.intent_classifier import INTENT_DOC_TYPES, QueryIntent, classify_query
from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)

INTENT_PROTOTYPES: dict[str, dict[str, str]] = {
    "exchange_rate": {
        "ar": "أسعار العملات سعر الصرف تحويل العملات دولار يورو",
        "en": "exchange rates currency converter USD EGP",
    },
    "certificate_rate": {
        "ar": "عائد شهادة فايدة نسبة شهادات ادخار",
        "en": "certificate yield interest rate",
    },
    "certificate_types": {
        "ar": "أنواع الشهادات شهادات ادخار بلادي استثمار",
        "en": "types of certificates saving certificates",
    },
    "personal_loan": {
        "ar": "قرض شخصي تمويل قروض",
        "en": "personal loan financing",
    },
    "credit_card": {
        "ar": "بطاقات ائتمان كريدت كارد",
        "en": "credit cards",
    },
    "card_types": {
        "ar": "أنواع البطاقات ائتمان خصم مدفوعة مقدما",
        "en": "types of bank cards debit credit prepaid",
    },
    "account_open": {
        "ar": "فتح حساب بنكي الأوراق المطلوبة حساب جاري",
        "en": "open bank account required documents current account",
    },
    "branch_locator": {
        "ar": "الفروع أقرب فرع",
        "en": "branch locator find branch",
    },
    "offers": {
        "ar": "عروض البنك خصومات",
        "en": "bank offers promotions",
    },
    "corporate": {
        "ar": "خدمات الشركات",
        "en": "corporate banking",
    },
    "sme": {
        "ar": "المشروعات الصغيرة والمتوسطة",
        "en": "SME small medium enterprises",
    },
    "faq": {
        "ar": "أسئلة شائعة",
        "en": "frequently asked questions FAQ",
    },
    "digital_banking": {
        "ar": "الأهلي نت موبايل بانكنج",
        "en": "internet banking mobile banking",
    },
}


@lru_cache(maxsize=1)
def _model():
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


def classify_with_fallback(query: str, language: str) -> QueryIntent:
    rule_intent = classify_query(query, language)
    if rule_intent.intent != "general_faq" or not settings.minilm_intent_fallback_enabled:
        return rule_intent

    try:
        model = _model()
        prototypes = []
        labels: list[str] = []
        for intent_name, texts in INTENT_PROTOTYPES.items():
            labels.append(intent_name)
            prototypes.append(texts.get(language, texts["en"]))
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
        category = {
            "exchange_rate": "exchange_rates",
            "certificate_rate": "certificates",
            "certificate_types": "certificates",
            "personal_loan": "loans",
            "credit_card": "cards",
            "card_types": "cards",
            "account_open": "accounts",
            "branch_locator": "branches",
            "offers": "offers",
            "corporate": "corporate",
            "sme": "sme",
            "faq": "general",
            "digital_banking": "digital_banking",
        }.get(best_label, "general")
        return QueryIntent(
            intent=best_label,
            category=category,
            confidence=min(0.74, best_score),
            allowed_doc_types=INTENT_DOC_TYPES.get(best_label, INTENT_DOC_TYPES["general_faq"]),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("minilm_intent_fallback_failed", error=str(exc))
        return rule_intent
