"""Banking entity extraction — dictionary/regex primary (deterministic).

Optional GLiNER path is feature-flagged and disabled by default for latency.
See docs/enterprise-model-decisions.md Task 3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from shared.arabic_normalize import normalize_arabic
from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)

CURRENCY = re.compile(r"(USD|EUR|EGP|GBP|SAR|AED|دولار|يورو|جنيه|ريال)", re.I)
TENOR = re.compile(
    r"(\d+)\s*(سن[ةه]|سنوات|شهر|اشهر|أشهر|year|years|month|months)|(?<!\w)سنه(?!\w)|(?<!\w)سنة(?!\w)",
    re.I,
)
CERTIFICATE = re.compile(
    r"(شهاد[ةه]\s*(?:الادخار|الاستثمار|بلادي)?|certificate(?:s)?|belady|بلادي)",
    re.I,
)
LOAN = re.compile(r"(قرض\s*(?:شخصي|عقاري|سيارات)?|تمويل|personal\s*loan|auto\s*loan|mortgage)", re.I)
CARD = re.compile(
    r"(بطاق[ةه]\s*(?:ائتمان|خصم|مدين|مسبقة)?|credit\s*card|debit\s*card|prepaid\s*card|"
    r"فيزا|ماستركارد|visa|mastercard)",
    re.I,
)
ACCOUNT = re.compile(r"(حساب\s*(?:جاري|توفير|ادخار)?|current\s*account|savings?\s*account)", re.I)
BRANCH = re.compile(r"(فرع|فروع|branch(?:es)?)", re.I)
LOCATION = re.compile(
    r"(القاهره|القاهرة|الجيزه|الجيزة|الإسكندريه|الإسكندرية|الاسكندريه|الاسكندرية|المعادي|مدينه\s*نصر|مدينة\s*نصر|"
    r"cairo|giza|alexandria|maadi|nasr\s*city)",
    re.I,
)
PRODUCT_HINTS = re.compile(
    r"(شهاد[ةه]|شهادات|بطاق[ةه]|بطاقات|حساب|قرض|تمويل|certificate|card|account|loan)",
    re.I,
)


@dataclass(frozen=True)
class ExtractedEntity:
    type: str
    value: str
    start: int | None = None
    end: int | None = None


@dataclass
class EntityExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    product_mentions: list[str] = field(default_factory=list)
    currencies: list[str] = field(default_factory=list)
    tenors: list[str] = field(default_factory=list)
    certificates: list[str] = field(default_factory=list)
    loans: list[str] = field(default_factory=list)
    cards: list[str] = field(default_factory=list)
    branches: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _gliner():
    from gliner import GLiNER

    logger.info("loading_gliner_model", model=settings.gliner_model)
    return GLiNER.from_pretrained(settings.gliner_model)


def _extract_gliner(text: str) -> list[ExtractedEntity]:
    if not settings.gliner_enabled:
        return []
    try:
        model = _gliner()
        labels = ["product", "currency", "tenor", "city", "document", "card", "loan", "certificate"]
        spans = model.predict_entities(text, labels, threshold=0.4)
        return [
            ExtractedEntity(type=str(span.get("label", "ENTITY")).upper(), value=str(span.get("text", "")))
            for span in spans
            if span.get("text")
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("gliner_extract_failed", error=str(exc))
        return []


def _add_typed(
    result: EntityExtractionResult,
    pattern: re.Pattern[str],
    text: str,
    entity_type: str,
    bucket: list[str],
) -> None:
    for match in pattern.finditer(text):
        value = match.group(0).strip()
        if not value:
            continue
        bucket.append(value)
        result.entities.append(ExtractedEntity(entity_type, value, match.start(), match.end()))


def extract_entities(query: str, language: str = "ar") -> EntityExtractionResult:
    text = normalize_arabic(query) if language == "ar" else query
    result = EntityExtractionResult()

    _add_typed(result, CURRENCY, text, "currency", result.currencies)
    _add_typed(result, TENOR, text, "tenor", result.tenors)
    _add_typed(result, CERTIFICATE, text, "certificate", result.certificates)
    _add_typed(result, LOAN, text, "loan", result.loans)
    _add_typed(result, CARD, text, "card", result.cards)
    _add_typed(result, ACCOUNT, text, "product", result.product_mentions)
    _add_typed(result, BRANCH, text, "branch", result.branches)
    _add_typed(result, LOCATION, text, "location", result.locations)

    for match in PRODUCT_HINTS.finditer(text):
        value = match.group(0)
        if value not in result.product_mentions:
            result.product_mentions.append(value)
            result.entities.append(ExtractedEntity("product", value, match.start(), match.end()))

    for entity in _extract_gliner(query):
        result.entities.append(entity)

    return result


def entities_as_dicts(result: EntityExtractionResult) -> list[dict[str, str]]:
    return [{"type": entity.type, "value": entity.value} for entity in result.entities]
