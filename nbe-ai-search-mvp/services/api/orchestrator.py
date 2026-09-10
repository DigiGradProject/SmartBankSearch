from __future__ import annotations

import hashlib
import time
from typing import Any

from ingestion.embedding.vector_store import RetrievedChunk
from services.context_builder.builder import ContextBuilder
from services.llm_service.llm import LLMService
from services.rag.analytics import AnalyticsEvent, write_analytics_event
from services.rag.audit import build_audit_event, write_audit_event
from services.rag.confidence import compute_confidence
from services.rag.explainability import build_explain_payload
from services.rag.metrics import LLM_LATENCY, RERANK_LATENCY, RETRIEVE_LATENCY, SEMANTIC_CACHE_HITS
from services.rag.query_planner import QueryPlan, plan_query
from services.rag.response_formatter import format_structured_sections, wrap_plain_answer
from services.rag.self_eval import STRICT_REGEN_HINT, evaluate_answer
from services.rag.semantic_cache import SemanticCache
from services.search_service.card_catalog import (
    build_card_citations,
    build_card_types_answer,
    build_credit_cards_answer,
    is_card_types_query,
    is_credit_cards_overview_query,
)
from services.search_service.certificate_catalog import (
    build_certificate_buy_answer,
    build_certificate_types_answer,
    is_certificate_types_query,
)
from services.search_service.intent_classifier import INTENT_DOC_TYPES, QueryIntent
from services.search_service.product_detail import try_product_detail_answer
from services.search_service.rate_guidance import (
    context_has_applicable_rate,
    is_rate_query,
    rate_answer,
    rate_guidance,
)
from services.search_service.search import RetrievalResult, SearchService
from services.search_service.suggestions import build_suggestions
from shared.config import settings
from shared.logging import get_logger
from shared.schemas import Citation, SearchResponse
from shared.url_canonical import canonical_url_key

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


def _cached_response_is_supported(
    response: SearchResponse,
    retrieval: RetrievalResult,
) -> bool:
    """Accept cached answers only when current retrieval supports a cited URL."""
    current_urls = {canonical_url_key(chunk.url) for chunk in retrieval.chunks if chunk.url}
    cached_urls = {
        canonical_url_key(citation.url)
        for citation in response.citations
        if citation.url
    }
    return bool(current_urls & cached_urls)


def merge_retrieval_results(results: list[RetrievalResult]) -> RetrievalResult:
    """Merge multi-intent retrieval into one result (best score per URL)."""
    if len(results) == 1:
        return results[0]
    primary = results[0]
    by_url: dict[str, RetrievedChunk] = {}
    citation_pool: list[RetrievedChunk] = []
    for result in results:
        for chunk in result.chunks:
            key = canonical_url_key(chunk.url or "") or chunk.chunk_id
            existing = by_url.get(key)
            if existing is None or chunk.score > existing.score:
                by_url[key] = chunk
        citation_pool.extend(result.citation_chunks or result.chunks)
    merged_chunks = sorted(by_url.values(), key=lambda c: c.score, reverse=True)
    intents = "+".join(dict.fromkeys(r.intent for r in results))
    conf = max(r.confidence for r in results)
    return RetrievalResult(
        query=primary.query,
        language=primary.language,
        chunks=merged_chunks[: settings.rerank_keep_size * 2],
        confidence=conf,
        should_answer=bool(results) and all(r.should_answer for r in results),
        abstention_reason=next(
            (r.abstention_reason for r in results if not r.should_answer),
            None,
        ),
        intent=intents,
        category=primary.category,
        intent_confidence=max(r.intent_confidence for r in results),
        filter_applied=any(r.filter_applied for r in results),
        hybrid_used=any(r.hybrid_used for r in results),
        citation_chunks=citation_pool,
        rewritten_query=" | ".join(r.rewritten_query for r in results if r.rewritten_query),
        entities=list({e for r in results for e in (r.entities or [])}),
        decision=primary.decision,
        confidence_reason=primary.confidence_reason,
        original_query=primary.original_query,
        entity_payload=primary.entity_payload,
        understand=primary.understand,
        retrieve_ms=sum(r.retrieve_ms for r in results),
        rerank_ms=sum(r.rerank_ms for r in results),
    )


class Orchestrator:
    def __init__(
        self,
        search_service: SearchService | None = None,
        context_builder: ContextBuilder | None = None,
        llm_service: LLMService | None = None,
        semantic_cache: SemanticCache | None = None,
        exact_cache: QueryCache | None = None,
    ) -> None:
        self.search_service = search_service or SearchService()
        self.context_builder = context_builder or ContextBuilder()
        self.llm_service = llm_service or LLMService()
        self.semantic_cache = semantic_cache
        self._semantic_cache_failed = False
        self.cache = exact_cache or QueryCache()

    def _get_semantic_cache(self) -> SemanticCache | None:
        if not settings.semantic_cache_enabled:
            return None
        if self.semantic_cache is not None:
            return self.semantic_cache
        if self._semantic_cache_failed:
            return None
        try:
            self.semantic_cache = SemanticCache()
            return self.semantic_cache
        except Exception as exc:  # noqa: BLE001
            logger.warning("semantic_cache_init_failed", error=str(exc))
            self._semantic_cache_failed = True
            return None

    @staticmethod
    def _cache_key(query: str, language: str) -> str:
        raw = f"{language}:{query.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _enrich_response_meta(
        self,
        response: SearchResponse,
        retrieval: RetrievalResult,
        *,
        cache_hit: bool = False,
        faithfulness: str | None = None,
        explain: dict[str, Any] | None = None,
    ) -> SearchResponse:
        response.rewritten_query = retrieval.rewritten_query or None
        response.entities = retrieval.entity_payload
        response.confidence_reason = response.confidence_reason or retrieval.confidence_reason or None
        response.cache_hit = cache_hit
        if faithfulness:
            response.faithfulness = faithfulness
        if explain is not None:
            response.explain = explain
        return response

    def _audit(self, query: str, retrieval: RetrievalResult, response: SearchResponse) -> SearchResponse:
        structured = format_structured_sections(response.answer or "")
        response.structured = structured
        response.intent = retrieval.intent
        response.query_hash = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:16]
        write_audit_event(
            build_audit_event(
                query=query,
                language=retrieval.language,
                intent=retrieval.intent,
                intent_confidence=retrieval.intent_confidence,
                chunk_ids=[chunk.chunk_id for chunk in retrieval.chunks],
                urls=[chunk.url for chunk in retrieval.chunks if chunk.url],
                confidence=response.confidence,
                answered=response.answered,
                abstention_reason=response.abstention_reason,
                model_embedding=settings.embedding_model,
                model_reranker=settings.reranker_model,
                model_llm=settings.ollama_model,
                rewritten_query=retrieval.rewritten_query or "",
                entities=retrieval.entities or [],
            )
        )
        return response

    def _write_analytics(
        self,
        *,
        query: str,
        retrieval: RetrievalResult,
        response: SearchResponse,
        llm_ms: float,
        total_ms: float,
        cache_hit: bool,
    ) -> None:
        write_analytics_event(
            AnalyticsEvent(
                query=query,
                intent=retrieval.intent,
                rewritten_query=retrieval.rewritten_query or "",
                entities=retrieval.entity_payload or [],
                retrieve_ms=retrieval.retrieve_ms,
                rerank_ms=retrieval.rerank_ms,
                llm_ms=llm_ms,
                total_ms=total_ms,
                cache_hit=cache_hit,
                top_documents=[
                    {"title": c.title, "url": c.url, "score": c.score, "category": getattr(c, "category", "")}
                    for c in retrieval.chunks[:5]
                ],
                confidence=response.confidence,
                confidence_reason=response.confidence_reason or "",
                decision=retrieval.decision,
                faithfulness=response.faithfulness,
                language=retrieval.language,
                answered=response.answered,
            )
        )

    def _retrieve_planned(self, plan: QueryPlan) -> RetrievalResult:
        results: list[RetrievalResult] = []
        for sub in plan.subqueries:
            results.append(
                self.search_service.retrieve(
                    sub.text,
                    plan.language,
                    understanding=sub.understanding,
                )
            )
        merged = merge_retrieval_results(results)
        merged.original_query = plan.original_query
        return merged

    async def search(self, query: str, language: str = "auto", *, debug: bool = False) -> SearchResponse:
        t_total = time.perf_counter()
        cache_key = self._cache_key(query, language)
        llm_ms = 0.0
        cache_hit = False

        plan = plan_query(query, language)
        retrieval = self._retrieve_planned(plan)
        if retrieval.retrieve_ms:
            RETRIEVE_LATENCY.observe(retrieval.retrieve_ms / 1000.0)
        if retrieval.rerank_ms:
            RERANK_LATENCY.observe(retrieval.rerank_ms / 1000.0)
        suggestions = build_suggestions(retrieval.query, retrieval.language, retrieval.chunks)
        understanding = retrieval.understand or plan.subqueries[0].understanding
        semantic: SemanticCache | None = None

        def _finalize(
            response: SearchResponse,
            *,
            faithfulness: str | None = None,
        ) -> SearchResponse:
            response = self._enrich_response_meta(
                response,
                retrieval,
                cache_hit=cache_hit,
                faithfulness=faithfulness,
            )
            if debug:
                response.explain = build_explain_payload(
                    understanding=understanding,
                    chunks=retrieval.chunks,
                    confidence=response.confidence,
                    confidence_reason=response.confidence_reason or "",
                    decision=retrieval.decision,
                    filter_applied=retrieval.filter_applied,
                    cache_hit=cache_hit,
                    plan_intents=[s.intent_name for s in plan.subqueries],
                )
            response = self._audit(query, retrieval, response)
            total_ms = (time.perf_counter() - t_total) * 1000.0
            self._write_analytics(
                query=query,
                retrieval=retrieval,
                response=response,
                llm_ms=llm_ms,
                total_ms=total_ms,
                cache_hit=cache_hit,
            )
            if response.answered and semantic and not cache_hit:
                semantic.store(
                    query,
                    retrieval.language,
                    response.model_dump(),
                    retrieved_urls=[c.url for c in retrieval.chunks if c.url],
                )
            if settings.cache_enabled and response.answered and not cache_hit:
                self.cache.set(cache_key, response.model_dump(), settings.cache_ttl_seconds)
            return response

        # Reliability boundary: no cache or answer builder may bypass the
        # current retrieval decision.
        if not retrieval.should_answer:
            return _finalize(
                SearchResponse(
                    answer=None,
                    confidence=retrieval.confidence,
                    confidence_reason=retrieval.confidence_reason,
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
            )

        if settings.cache_enabled:
            cached = self.cache.get(cache_key, settings.cache_ttl_seconds)
            if cached:
                response = SearchResponse.model_validate(cached)
                if _cached_response_is_supported(response, retrieval):
                    cache_hit = True
                    return _finalize(response)
                logger.info("search_exact_cache_rejected", reason="retrieval_source_mismatch")

        semantic = self._get_semantic_cache()
        # These deterministic responses evolve with response formatting and
        # must not be replaced by an older semantic-cache payload.
        certificate_intents = {"certificate_buy", "certificate_types"}
        skip_semantic_cache = bool(certificate_intents.intersection(retrieval.intent.split("+")))
        skip_semantic_cache = skip_semantic_cache or "ahly points" in " ".join(query.lower().split())
        if semantic and not skip_semantic_cache:
            hit = semantic.lookup(query, retrieval.language)
            if hit:
                response = SearchResponse.model_validate(hit.payload)
                if _cached_response_is_supported(response, retrieval):
                    cache_hit = True
                    SEMANTIC_CACHE_HITS.inc()
                    logger.info("search_semantic_cache_hit", similarity=hit.similarity)
                    return _finalize(response)
                logger.info(
                    "search_semantic_cache_rejected",
                    similarity=hit.similarity,
                    reason="retrieval_source_mismatch",
                )

        if is_certificate_types_query(retrieval.query, retrieval.language):
            catalog_answer = build_certificate_types_answer(retrieval.chunks, retrieval.language)
            if catalog_answer:
                built = self.context_builder.build(retrieval.query, retrieval.chunks)
                return _finalize(
                    SearchResponse(
                        answer=catalog_answer,
                        confidence=round(max(0.72, retrieval.confidence * 0.85), 3),
                        confidence_reason=retrieval.confidence_reason,
                        citations=built.citations,
                        answered=True,
                        language=retrieval.language,  # type: ignore[arg-type]
                        suggestions=suggestions,
                    )
                )

        if "certificate_buy" in retrieval.intent.split("+"):
            buy_answer = build_certificate_buy_answer(retrieval.chunks, retrieval.language)
            if buy_answer:
                built = self.context_builder.build(retrieval.query, retrieval.chunks)
                return _finalize(
                    SearchResponse(
                        answer=buy_answer,
                        confidence=round(max(0.72, retrieval.confidence * 0.85), 3),
                        confidence_reason=retrieval.confidence_reason,
                        citations=built.citations,
                        answered=True,
                        language=retrieval.language,  # type: ignore[arg-type]
                        suggestions=suggestions,
                    )
                )

        if is_card_types_query(retrieval.query, retrieval.language):
            card_answer = build_card_types_answer(retrieval.chunks, retrieval.language)
            if card_answer:
                citation_pool = retrieval.citation_chunks or retrieval.chunks
                citations = build_card_citations(citation_pool, max_items=5, language=retrieval.language)
                return _finalize(
                    SearchResponse(
                        answer=card_answer,
                        confidence=round(max(0.75, retrieval.confidence * 0.88), 3),
                        confidence_reason=retrieval.confidence_reason,
                        citations=citations,
                        answered=True,
                        language=retrieval.language,  # type: ignore[arg-type]
                        suggestions=suggestions,
                    )
                )

        if is_credit_cards_overview_query(retrieval.query, retrieval.language):
            credit_answer = build_credit_cards_answer(retrieval.chunks, retrieval.language)
            if credit_answer:
                citation_pool = retrieval.citation_chunks or retrieval.chunks
                citations = build_card_citations(
                    citation_pool,
                    max_items=5,
                    include_families=("credit",),
                    language=retrieval.language,
                )
                return _finalize(
                    SearchResponse(
                        answer=credit_answer,
                        confidence=round(max(0.78, retrieval.confidence * 0.9), 3),
                        confidence_reason=retrieval.confidence_reason,
                        citations=citations,
                        answered=True,
                        language=retrieval.language,  # type: ignore[arg-type]
                        suggestions=suggestions,
                    )
                )

        built = self.context_builder.build(retrieval.query, retrieval.chunks)
        if not built.context_text:
            return _finalize(
                SearchResponse(
                    answer=None,
                    confidence=retrieval.confidence,
                    confidence_reason=retrieval.confidence_reason,
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
            )

        product_detail = try_product_detail_answer(
            retrieval.query, retrieval.language, retrieval.chunks
        )
        if product_detail:
            answer, source_chunk = product_detail
            citations = (
                [Citation(title=source_chunk.title, url=source_chunk.url, category=getattr(source_chunk, "category", None), relevance_score=source_chunk.score, reranker_score=source_chunk.score)]
                if source_chunk.url
                else built.citations
            )
            return _finalize(
                SearchResponse(
                    answer=answer,
                    confidence=round(max(0.78, retrieval.confidence * 0.9), 3),
                    confidence_reason=retrieval.confidence_reason,
                    citations=citations,
                    answered=True,
                    language=retrieval.language,  # type: ignore[arg-type]
                    suggestions=suggestions,
                )
            )

        if is_rate_query(retrieval.query, retrieval.language) and not context_has_applicable_rate(
            retrieval.query, retrieval.language, built.context_text
        ):
            return _finalize(
                SearchResponse(
                    answer=None,
                    confidence=retrieval.confidence,
                    confidence_reason=retrieval.confidence_reason,
                    citations=[],
                    answered=False,
                    language=retrieval.language,  # type: ignore[arg-type]
                    abstention_reason="missing_applicable_rate",
                    suggestions=suggestions,
                    guidance=rate_answer(retrieval.query, retrieval.language),
                )
            )

        t_llm = time.perf_counter()
        answer, llm_confidence = await self.llm_service.generate_answer(
            retrieval.query,
            built.context_text,
            retrieval.language,
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000.0
        LLM_LATENCY.observe(llm_ms / 1000.0)

        if not answer:
            answer = self.llm_service._fallback_answer(retrieval.query, built.context_text, retrieval.language)
            llm_confidence = 0.5 if answer else 0.0

        faithfulness_label = None
        if answer:
            eval_result = await evaluate_answer(
                query=retrieval.query,
                answer=answer,
                context=built.context_text,
                language=retrieval.language,
            )
            faithfulness_label = eval_result.label.value
            if eval_result.should_regenerate:
                strict_context = f"{STRICT_REGEN_HINT}\n\n{built.context_text}"
                t_llm = time.perf_counter()
                regen, regen_conf = await self.llm_service.generate_answer(
                    retrieval.query,
                    strict_context,
                    retrieval.language,
                )
                llm_ms += (time.perf_counter() - t_llm) * 1000.0
                if regen:
                    answer = regen
                    llm_confidence = regen_conf
                    recheck = await evaluate_answer(
                        query=retrieval.query,
                        answer=answer,
                        context=built.context_text,
                        language=retrieval.language,
                    )
                    faithfulness_label = recheck.label.value
                    if recheck.should_regenerate:
                        answer = ""
                else:
                    # Never publish the original answer when the strict
                    # regeneration itself failed.
                    answer = ""

        if not answer:
            return _finalize(
                SearchResponse(
                    answer=None,
                    confidence=min(retrieval.confidence, llm_confidence),
                    confidence_reason=retrieval.confidence_reason,
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
                ),
                faithfulness=faithfulness_label,
            )

        answer = wrap_plain_answer(
            answer,
            language=retrieval.language,
            citations=built.citations,
            intent=retrieval.intent.split("+")[0],
        )
        intent_name = retrieval.intent.split("+")[0]
        breakdown = compute_confidence(
            retrieval.chunks,
            llm_confidence=llm_confidence,
            embedding_similarity=retrieval.confidence,
            intent=QueryIntent(
                intent=intent_name,
                category=retrieval.category,
                confidence=retrieval.intent_confidence,
                allowed_doc_types=INTENT_DOC_TYPES.get(
                    intent_name,
                    INTENT_DOC_TYPES["general_faq"],
                ),
            ),
        )
        response = SearchResponse(
            answer=answer,
            confidence=breakdown.final,
            confidence_reason=breakdown.reason or retrieval.confidence_reason,
            citations=built.citations,
            answered=True,
            language=retrieval.language,  # type: ignore[arg-type]
            suggestions=suggestions,
        )
        logger.info(
            "search_answered",
            language=retrieval.language,
            confidence=breakdown.final,
            citations=len(built.citations),
            intent=retrieval.intent,
            faithfulness=faithfulness_label,
        )
        return _finalize(response, faithfulness=faithfulness_label)

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
