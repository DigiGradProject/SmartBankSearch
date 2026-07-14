from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from services.context_builder.builder import ContextBuilder
from services.llm_service.llm import LLMService
from services.search_service.suggestions import build_suggestions
from services.search_service.certificate_catalog import (
    build_certificate_types_answer,
    is_certificate_types_query,
)
from services.search_service.rate_guidance import (
    context_has_applicable_rate,
    is_rate_query,
    rate_answer,
    rate_citations,
    rate_guidance,
)
from services.search_service.search import SearchService
from shared.logging import get_logger
from shared.schemas import SearchResponse

logger = get_logger(__name__)


class QueryCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, dict[str, Any]]] = {}

    def get(self, key: str, ttl_seconds: int) -> dict[str, Any] | None:
        item = self._store.get(key)
        if not item:
            return None
        expires_at, payload = item
        if time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return payload

    def set(self, key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        self._store[key] = (time.time() + ttl_seconds, payload)


class Orchestrator:
    def __init__(self) -> None:
        self.search_service = SearchService()
        self.context_builder = ContextBuilder()
        self.llm_service = LLMService()
        self.cache = QueryCache()

    @staticmethod
    def _cache_key(query: str, language: str) -> str:
        raw = f"{language}:{query.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def search(self, query: str, language: str = "auto") -> SearchResponse:
        from shared.config import settings

        cache_key = self._cache_key(query, language)
        if settings.cache_enabled:
            cached = self.cache.get(cache_key, settings.cache_ttl_seconds)
            if cached:
                return SearchResponse.model_validate(cached)

        retrieval = self.search_service.retrieve(query, language)
        suggestions = build_suggestions(retrieval.query, retrieval.language, retrieval.chunks)

        if not retrieval.should_answer:
            response = SearchResponse(
                answer=None,
                confidence=retrieval.confidence,
                citations=[],
                answered=False,
                language=retrieval.language,  # type: ignore[arg-type]
                abstention_reason=retrieval.abstention_reason,
                suggestions=suggestions,
                guidance=self._guidance_message(
                    retrieval.query,
                    retrieval.language,
                    suggestions,
                    retrieval.abstention_reason,
                ),
            )
            return response

        built = self.context_builder.build(retrieval.query, retrieval.chunks)
        if not built.context_text:
            return SearchResponse(
                answer=None,
                confidence=retrieval.confidence,
                citations=[],
                answered=False,
                language=retrieval.language,  # type: ignore[arg-type]
                abstention_reason="empty_context",
                suggestions=suggestions,
                guidance=self._guidance_message(
                    retrieval.query,
                    retrieval.language,
                    suggestions,
                    "empty_context",
                ),
            )

        # Certificate catalog questions: answer from indexed product stubs/pages directly.
        if is_certificate_types_query(retrieval.query, retrieval.language):
            catalog_answer = build_certificate_types_answer(retrieval.chunks, retrieval.language)
            if catalog_answer:
                return SearchResponse(
                    answer=catalog_answer,
                    confidence=round(max(0.72, retrieval.confidence * 0.85), 3),
                    citations=built.citations,
                    answered=True,
                    language=retrieval.language,  # type: ignore[arg-type]
                    suggestions=suggestions,
                )

        # Yield/rate questions: only use LLM when evidence contains an applicable numeric rate.
        if is_rate_query(retrieval.query, retrieval.language) and not context_has_applicable_rate(
            retrieval.query, retrieval.language, built.context_text
        ):
            answer = rate_answer(retrieval.query, retrieval.language)
            citations = rate_citations(retrieval.query, retrieval.language)
            return SearchResponse(
                answer=answer,
                confidence=round(max(0.55, retrieval.confidence * 0.75), 3),
                citations=citations,
                answered=True,
                language=retrieval.language,  # type: ignore[arg-type]
                abstention_reason=None,
                suggestions=suggestions,
                guidance=None,
            )

        answer, llm_confidence = await self.llm_service.generate_answer(
            retrieval.query,
            built.context_text,
            retrieval.language,
        )

        if not answer:
            answer = self.llm_service._fallback_answer(retrieval.query, built.context_text, retrieval.language)
            llm_confidence = 0.5 if answer else 0.0

        if not answer:
            response = SearchResponse(
                answer=None,
                confidence=min(retrieval.confidence, llm_confidence),
                citations=built.citations,
                answered=False,
                language=retrieval.language,  # type: ignore[arg-type]
                abstention_reason="insufficient_context",
                suggestions=suggestions,
                guidance=self._guidance_message(
                    retrieval.query,
                    retrieval.language,
                    suggestions,
                    "insufficient_context",
                ),
            )
            return response

        final_confidence = round((retrieval.confidence * 0.6) + (llm_confidence * 0.4), 3)
        response = SearchResponse(
            answer=answer,
            confidence=final_confidence,
            citations=built.citations,
            answered=True,
            language=retrieval.language,  # type: ignore[arg-type]
        )

        if settings.cache_enabled:
            self.cache.set(cache_key, response.model_dump(), settings.cache_ttl_seconds)

        logger.info(
            "search_answered",
            language=retrieval.language,
            confidence=final_confidence,
            citations=len(built.citations),
        )
        return response

    @staticmethod
    def _guidance_message(
        query: str,
        language: str,
        suggestions: list,
        abstention_reason: str | None = None,
    ) -> str:
        special = rate_guidance(query, language)
        if special:
            return special
        if language == "ar":
            if suggestions:
                return "لم نجد إجابة مؤكدة من المحتوى المفهرس، وهذه روابط/مواضيع قريبة قد تفيدك:"
            return "لم نجد محتوى كافياً. جرّب صياغة السؤال بطريقة أخرى أو استخدم البحث التقليدي على موقع البنك."
        if suggestions:
            return "No confident answer in the indexed content. Try these related pages/topics:"
        return "Not enough evidence found. Try rephrasing or use traditional search on the bank website."
