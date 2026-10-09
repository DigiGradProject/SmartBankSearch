"""Hybrid page search — dense + BM25 fusion for the Hybrid tab + grounded RAG answer.

Pipeline (user-approved plan, docs/plan-hybrid-search.md §1b Hybrid RAG):

    BM25 Top 50  +  BGE-M3 dense Top 50
              ↓  RRF (reciprocal rank fusion)
            Top 30
              ↓  BGE reranker (cross-encoder)
              ↓  light title / exact-match boost (controlled, NOT a hard override)
            Top 10 page-level results  ← FROZEN ranking; generation never reorders it
              ↓  grounded context from top ranked chunks (ContextBuilder)
              ↓  existing LLMService.generate_answer (grounding + NO_ANSWER abstention)
            answer + citations (URLs from retrieved pages only) → shown FIRST in the UI,
            followed by the ranked page list.

NOT inherited from AI mode (deliberate): intent business-rule boosts, query
expansion/planning, exact + semantic caches, confidence gate, catalog answer
builders, self-eval regeneration loop (one LLM call keeps latency bounded on
the already rerank-heavy pipeline). The LLM call fails OPEN: on any error the
request still returns the ranked pages.

Everything downstream of the channel queries reuses existing, tested code:
`reciprocal_rank_fusion`, `get_reranker`, `traditional.prepare_keyword_query`,
`traditional.group_pages` (page dedup + title phrase bonus),
`traditional.normalize_scores`, `traditional.build_snippet`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace

from ingestion.embedding.vector_store import RetrievedChunk, VectorStore
from ingestion.lexical.bm25_index import BM25Index, IndexedChunk, get_bm25_index
from services.context_builder.builder import BuiltContext, ContextBuilder
from services.llm_service.llm import LLMService
from services.rag.response_formatter import wrap_plain_answer
from services.search_service.keyword_rank import extract_query_terms
from services.search_service.reranker import get_reranker
from services.search_service.hybrid_retriever import reciprocal_rank_fusion
from services.search_service.traditional import (
    TraditionalSearchOutcome,
    build_snippet,
    normalize_scores,
    prepare_keyword_query,
)
from shared.arabic_normalize import normalize_arabic
from shared.config import settings
from shared.logging import get_logger
from shared.schemas import Citation, TraditionalResult
from shared.url_canonical import canonical_url_key

logger = get_logger(__name__)

# Pipeline stage sizes (plan §2): fusion keeps 30 candidates for the reranker,
# which returns rerank_keep for the final title boost + page grouping.
_HYBRID_FUSED_KEEP = 30
# Controlled title boost AFTER reranking — multiplicative, capped, never a
# hard override (review note #6): a page whose title covers the query's
# content tokens rises, a body-heavy page with a bare "platinum" mention
# wins on relevance alone if its rerank score is genuinely higher.
_HYBRID_TITLE_BOOST = 0.35  # ×(1 + 0.35·coverage) → up to ×1.35
# Exact normalized-title match gets a slightly stronger, still-controlled bump.
_HYBRID_TITLE_EXACT_BONUS = 0.25  # ×(1 + 0.25) on top of coverage boost

# Number of top-ranked chunks fed to the grounded LLM context.
# CHUNK-level (not page-deduped): the exchange-rates page carries one chunk
# per currency, so page dedup would collapse the context to a single currency
# (user-reported: hybrid answered USD-only while AI mode listed most
# currencies). Chunk-level context mirrors AI mode's grounded granularity;
# ContextBuilder still dedupes near-duplicates and caps the token budget.
_HYBRID_CONTEXT_TOP_CHUNKS = settings.retrieval_top_k

# The ranked page list below the answer IS the source list (user feedback) —
# the "### Source" section is stripped from the generated answer.
_SOURCE_SECTION = re.compile(r"\n###\s*Source\b.*$", re.DOTALL | re.IGNORECASE)


def _strip_source_section(answer: str) -> str:
    return _SOURCE_SECTION.sub("", answer).strip()


@dataclass(frozen=True)
class HybridPagesOutcome:
    """Hybrid-tab payload: FROZEN page ranking + optional grounded answer."""

    results: list[TraditionalResult] = field(default_factory=list)
    total: int = 0
    language: str = "en"
    effective_query: bool = True
    bm25_available: bool = True
    dense_candidates: int = 0
    bm25_candidates: int = 0
    fused_candidates: int = 0
    reranked: bool = False
    title_boost_applied: bool = False
    # --- Phase 2: grounded RAG answer (answers generate AFTER ranking) ---
    answer: str | None = None
    citations: list[Citation] = field(default_factory=list)
    answer_confidence: float = 0.0
    # True only when generation itself threw (context build / LLM client);
    # LLM refusals (insufficient evidence) are NOT errors.
    generation_error: bool = False
    # Top normalized page chunks feeding the grounded context (internal only,
    # never serialized to the API response).
    context_chunks: list[RetrievedChunk] = field(default_factory=list, repr=False)
    # Latency bookkeeping (ms): retrieval vs generation vs total.
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    total_ms: float | None = None


def _rerank_query_terms(query: str, language: str) -> list[str]:
    terms = extract_query_terms(query, language)
    return terms if terms else []


def rerank_query_text(query: str, language: str) -> str:
    """Query text for the reranker — raw is fine (cross-encoders are robust)."""
    return query.strip()


def _hybrid_fusion(
    dense_chunks: list[RetrievedChunk],
    bm25_chunks: list[tuple[object, float]],
    to_retrieved,
) -> tuple[list[RetrievedChunk], bool]:
    """RRF-fuse the dense and BM25 channels; True when both channels ran.

    Kept as a thin, deterministic wrapper so the fusion math lives only in
    the tested `hybrid_retriever.reciprocal_rank_fusion`.
    """
    dense_ranking = [chunk.chunk_id for chunk in dense_chunks]
    bm25_ranking = [chunk.chunk_id for chunk, _ in bm25_chunks]
    if not dense_ranking and not bm25_ranking:
        return [], False
    if not bm25_ranking:
        return list(dense_chunks), False
    if not dense_ranking:
        return [to_retrieved(chunk, score) for chunk, score in bm25_chunks], False

    fused_scores = reciprocal_rank_fusion([dense_ranking, bm25_ranking])
    merged = {chunk.chunk_id: chunk for chunk in dense_chunks}
    for chunk, score in bm25_chunks:
        merged.setdefault(chunk.chunk_id, to_retrieved(chunk, score))
    fused: list[RetrievedChunk] = []
    for chunk_id, rrf_score in sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True):
        fused.append(replace(merged[chunk_id], score=rrf_score))
    return fused, True


def _title_boost_ranking(
    fused: list[RetrievedChunk],
    terms: list[str],
    language: str,
) -> list[RetrievedChunk]:
    """Controlled multiplicative title boost after reranking (note #6).

    Coverage = fraction of the query's content tokens present in the title
    (frequency-agnostic); exact normalized-title match adds a small bonus.
    Rerank score stays the base — the boost reorders near-ties, it does not
    force a page to #1 unconditionally.
    """
    if not terms:
        return list(fused)
    match_tokens = _title_match_tokens(terms, language)
    query_keys = _normalized_query_keys(" ".join(match_tokens), language)

    boosted: list[RetrievedChunk] = []
    for chunk in fused:
        title = _title_haystack(chunk.title, language)
        if not title:
            boosted.append(chunk)
            continue
        hits = sum(1 for term in match_tokens if term in title)
        coverage = hits / len(match_tokens)
        factor = 1.0 + _HYBRID_TITLE_BOOST * coverage
        if title in query_keys:
            factor += _HYBRID_TITLE_EXACT_BONUS
        boosted.append(replace(chunk, score=chunk.score * factor))
    return sorted(boosted, key=lambda item: item.score, reverse=True)


def _title_match_tokens(terms: list[str], language: str) -> list[str]:
    """Corporate-prefix tokens excluded from the exact-match key comparison."""
    from services.search_service.traditional import _title_match_tokens as _tmt

    return _tmt(terms, language)


def _title_haystack(title: str, language: str) -> str:
    # Strip the corporate prefix the same way autocomplete/traditional do, so
    # "National Bank of Egypt - Platinum" matches a query for "platinum".
    for prefix in _CORPORATE_PREFIXES:
        if title.startswith(prefix):
            remainder = title[len(prefix):].strip()
            if remainder:
                title = remainder
            break
    lowered = title.lower().strip()
    return normalize_arabic(lowered) if language == "ar" else lowered


_CORPORATE_PREFIXES = (
    "National Bank of Egypt - ",
    "البنك الأهلى المصرى - ",
    "البنك الأهلي المصرى - ",
)


def _normalized_query_keys(text: str, language: str) -> set[str]:
    keys = {normalize_arabic(text.lower().strip()) if language == "ar" else text.lower().strip()}
    return {key for key in keys if key}


def grouped_best_pages(
    chunks: list[IndexedChunk],
    scores: list[float],
    terms: list[str],
    language: str,
) -> list[tuple[IndexedChunk, float]]:
    """Canonical-URL best-chunk-per-page dedup (no extra boosting: the hybrid
    title boost already ran; BM25's own `_title_boosted_score` must not
    double-apply). Local to avoid importing group_pages with its express-
    tier expectations."""
    best: dict[str, tuple[IndexedChunk, float]] = {}
    for chunk, score in zip(chunks, scores):
        key = canonical_url_key(chunk.url) or chunk.chunk_id
        current = best.get(key)
        if current is None or score > current[1]:
            best[key] = (chunk, score)
    return sorted(best.values(), key=lambda pair: pair[1], reverse=True)


class HybridPageSearchService:
    """Dense + BM25 fusion behind the `mode="hybrid"` API contract.

    Phase 2: `search_with_answer` wraps `search` (FROZEN ranking) and adds a
    grounded LLM answer from the top ranked chunks. Retrieval ordering is
    never touched by generation; an LLM failure degrades to pages-only.
    LLM infra is injected (test seams) but defaults to the SAME service AI
    mode uses — no duplicate implementation.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
        reranker=None,  # noqa: ANN001 — test seam
        *,
        bm25_disabled: bool = False,
        context_builder: ContextBuilder | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._reranker = reranker
        self._context_builder = context_builder
        self._llm_service = llm_service
        # True when tests deliberately pass bm25_index=None: BM25 is then
        # "configured off" for this instance, so dense-only is a degradation,
        # not an index-load fallthrough (which would retry the real pickle).
        self._bm25_disabled = bm25_disabled

    def _get_bm25(self) -> BM25Index | None:
        if self._bm25_disabled:
            return None
        if self._bm25_index is not None:
            return self._bm25_index if self._bm25_index.size > 0 else None
        if not settings.bm25_enabled:
            return None
        index = get_bm25_index()
        return index if index.size > 0 else None

    def _get_vector_store(self) -> VectorStore:
        return self._vector_store or VectorStore()

    def _get_context_builder(self) -> ContextBuilder:
        return self._context_builder or ContextBuilder()

    def _get_llm(self) -> LLMService:
        return self._llm_service or LLMService()

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        language: str = "auto",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> HybridPagesOutcome:
        """Retrieval only — frozen page ranking, no LLM (Phase 1 contract)."""
        t0 = time.perf_counter()
        page_size = limit or settings.keyword_results_limit
        page_size = max(1, min(page_size, settings.keyword_results_max_limit))
        offset = max(0, offset)

        # Keyword-style prep (stopwords + light stems) — NO intent rewriting,
        # NO synonym expansion (review note #4: fair retrieval comparison).
        kw = prepare_keyword_query(query, language)
        if not kw.effective:
            return HybridPagesOutcome(
                language=kw.language, effective_query=False, bm25_available=True
            )
        resolved_language = kw.language
        embed_text = kw.original_query  # dense channel sees the raw query
        bm25_text = kw.bm25_query

        bm25 = self._get_bm25()
        dense_limit = max(page_size * 4, settings.dense_top_k)
        bm25_limit = max(page_size * 4, settings.bm25_top_k)

        dense_chunks: list[RetrievedChunk] = []
        try:
            dense_chunks = self._get_vector_store().query(
                embed_text, dense_limit, language=resolved_language
            )
        except Exception as exc:  # noqa: BLE001 — dense channel is optional
            logger.warning("hybrid_dense_channel_failed", error=str(exc))

        bm25_hits: list[tuple[object, float]] = []
        if bm25 is not None:
            try:
                bm25_hits = list(bm25.query(bm25_text, resolved_language, bm25_limit))
            except Exception as exc:  # noqa: BLE001 — lexical channel is optional
                logger.warning("hybrid_bm25_channel_failed", error=str(exc))

        def _to_retrieved(chunk, score):  # noqa: ANN001 — BM25Index/IndexedChunk pair
            return bm25.to_retrieved_chunk(chunk, score)

        fused, both_channels = _hybrid_fusion(dense_chunks, bm25_hits, _to_retrieved)
        if not fused:
            return HybridPagesOutcome(
                language=resolved_language,
                effective_query=True,
                dense_candidates=len(dense_chunks),
                bm25_candidates=len(bm25_hits),
                fused_candidates=0,
                bm25_available=bm25 is not None,
            )

        fused_keep = fused[: min(_HYBRID_FUSED_KEEP, max(page_size, _HYBRID_FUSED_KEEP))]

        # Reranker: cross-encoder over the fused pool, deterministic keyword
        # overlap boost as fallback when the model is unavailable/fails.
        reranked = False
        try:
            reranker = self._reranker
            if reranker is None and settings.reranker_enabled:
                reranker = get_reranker()
            if reranker is not None:
                pool = fused_keep[: max(page_size * 3, settings.rerank_pool_size)]
                keep = max(page_size, settings.rerank_pool_size)
                fused_keep = reranker.rerank(
                    rerank_query_text(bm25_text or embed_text, resolved_language),
                    pool,
                    top_k=keep,
                )
                reranked = True
        except Exception as exc:  # noqa: BLE001 — fallback to keyword overlap
            logger.warning("hybrid_reranker_failed_fallback", error=str(exc))
            reranked = False

        primary_terms = _rerank_query_terms(kw.original_query, resolved_language)
        ranked = _title_boost_ranking(fused_keep, primary_terms, resolved_language)

        # Page-level dedup via the tested group_pages: it needs IndexedChunk-
        # shaped objects only for .url/.title/.text/.language fields.
        chunkish = [
            IndexedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                title=c.title,
                url=c.url,
                language=c.language,
                text=c.text,
                doc_type=c.doc_type,
                category=c.category,
                is_stub=False,
                canonical_url_slug="",
            )
            for c in ranked
        ]
        pages = grouped_best_pages(chunkish, [c.score for c in ranked], primary_terms, resolved_language)
        candidates = normalize_scores(pages[: max(page_size, 1) * 2])

        total = len(candidates)
        window = candidates[offset : offset + page_size]

        results: list[TraditionalResult] = []
        for page, score in window:
            url = page.url or ""
            if not url.startswith(("http://", "https://")):
                continue
            result_language = page.language if page.language in {"ar", "en"} else resolved_language
            terms = _rerank_query_terms(kw.original_query, result_language)
            results.append(
                TraditionalResult(
                    title=page.title,
                    url=url,
                    snippet=build_snippet(page.text, terms, result_language),
                    score=round(float(score), 3),
                    terms=terms,
                    language=result_language,  # type: ignore[arg-type]
                    category=page.category or None,
                    doc_type=page.doc_type or None,
                )
            )

        candidate_chunks = list(ranked)  # CHUNK-level: same-page chunks all eligible for context

        outcome = HybridPagesOutcome(
            results=results,
            total=total,
            language=resolved_language,
            dense_candidates=len(dense_chunks),
            bm25_candidates=len(bm25_hits),
            fused_candidates=len(fused),
            reranked=reranked,
            title_boost_applied=True,
            bm25_available=bm25 is not None,
            context_chunks=candidate_chunks,
            retrieval_ms=round((time.perf_counter() - t0) * 1000.0, 1),
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "hybrid_search_completed",
            language=resolved_language,
            effective=kw.effective,
            dense=len(dense_chunks),
            bm25=len(bm25_hits),
            fused=len(fused),
            fused_candidates=len(fused),
            reranked=reranked,
            pages=total,
            returned=len(results),
            offset=offset,
            limit=page_size,
            retrieval_ms=round(retrieval_ms, 1),
        )
        return outcome

    # ------------------------------------------------------------------
    # Phase 2 — grounded answer generation (Hybrid RAG)
    # ------------------------------------------------------------------

    async def search_with_answer(
        self,
        query: str,
        language: str = "auto",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> HybridPagesOutcome:
        """`search` (frozen ranking) + grounded LLM answer from the top pages.

        The results list is built BEFORE generation and never reordered —
        the LLM only adds `answer` + `citations` on top. Any generation
        failure degrades to pages-only; the request never fails because of
        the LLM.
        """
        t_total = time.perf_counter()
        outcome = self.search(query, language, limit=limit, offset=offset)
        if not outcome.context_chunks or offset > 0:
            # No retrieval evidence (or a pagination page): answers are only
            # generated for the first page of results.
            return outcome

        t_gen = time.perf_counter()
        try:
            answer, citations, llm_confidence = await self._generate_grounded_answer(
                query, outcome.context_chunks, outcome.language
            )
        except Exception as exc:  # noqa: BLE001 — generation must never fail the request
            logger.warning("hybrid_answer_generation_failed", error=str(exc))
            outcome = replace(
                outcome,
                generation_error=True,
                total_ms=round((time.perf_counter() - t_total) * 1000.0, 1),
                generation_ms=None,
            )
            logger.info(
                "hybrid_search_with_answer_completed",
                answered=False,
                generation_error=True,
                total_ms=outcome.total_ms,
            )
            return outcome
        generation_ms = (time.perf_counter() - t_gen) * 1000.0

        outcome = replace(
            outcome,
            answer=answer,
            citations=citations,
            answer_confidence=llm_confidence if answer else 0.0,
            generation_error=False,
            total_ms=round((time.perf_counter() - t_total) * 1000.0, 1),
            generation_ms=round(generation_ms, 1),
        )
        logger.info(
            "hybrid_search_with_answer_completed",
            answered=bool(answer),
            generation_error=False,
            citations=len(citations),
            generation_ms=outcome.generation_ms,
            total_ms=outcome.total_ms,
        )
        return outcome

    async def _generate_grounded_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        language: str,
    ) -> tuple[str | None, list[Citation], float]:
        """Grounded answer strictly from the retrieved page chunks.

        Reuses AI-mode building blocks verbatim (user requirement #5):
        `ContextBuilder.build` for grounded context + citation selection and
        `LLMService.generate_answer` for the grounding/NO_ANSWER contract.
        Compression is OFF — the reranker already picked and ordered these
        top chunks, so the context order preserves the frozen ranking.
        """
        built = self._get_context_builder().build(
            query,
            chunks[:_HYBRID_CONTEXT_TOP_CHUNKS],
            compress=False,
        )
        if not built.context_text:
            # Insufficient evidence — abstain; pages are still returned.
            return None, [], 0.0

        answer, llm_confidence = await self._generate_with_hybrid_prompt(
            query,
            built.context_text,
            language,
        )
        if not answer:
            # LLM refused (NO_ANSWER / insufficient evidence) or is down:
            # abstain gracefully, keep the ranked pages.
            return None, built.citations, llm_confidence
        answer = wrap_plain_answer(
            answer,
            language=language,
            citations=built.citations,
            intent="hybrid_search",
        )
        # The ranked page list below the answer is the source list — drop the
        # duplicated "### Source" section from the generated answer.
        return _strip_source_section(answer), built.citations, llm_confidence

    async def _generate_with_hybrid_prompt(
        self,
        query: str,
        context: str,
        language: str,
    ) -> tuple[str | None, float]:
        """`LLMService.generate_answer` with the hybrid RAG prompt override.

        Broad rate questions must list every currency in the retrieved table,
        not just USD (the shared prompt's single-currency template). AI mode
        keeps its exact prompt — the override flag is set only for this call
        and restored immediately (no shared-state leakage).
        """
        llm = self._get_llm()
        try:
            llm.hybrid_broad_answer = True
            return await llm.generate_answer(query, context, language)
        finally:
            llm.hybrid_broad_answer = False
