"""Hybrid page search — dense + BM25 fusion for the Hybrid tab, no LLM.

Pipeline (user-approved plan, docs/plan-hybrid-search.md):

    BM25 Top 50  +  BGE-M3 dense Top 50
              ↓  RRF (reciprocal rank fusion)
            Top 30
              ↓  BGE reranker (cross-encoder)
              ↓  light title / exact-match boost (controlled, NOT a hard override)
            Top 10 page-level results

Retrieval-only: no LLM answer, no business-rule ranking, no confidence gate,
no AI-mode caching. Query prep reuses the *traditional* keyword pipeline
(stopwords + light stems only) so hybrid behaves like a pure dense+lexical
experiment — no intent rewriting, no synonym expansion.

Everything downstream of the channel queries reuses existing, tested code:
`reciprocal_rank_fusion`, `get_reranker`, `traditional.prepare_keyword_query`,
`traditional.group_pages` (page dedup + title phrase bonus),
`traditional.normalize_scores`, `traditional.build_snippet`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ingestion.embedding.vector_store import RetrievedChunk, VectorStore
from ingestion.lexical.bm25_index import BM25Index, IndexedChunk, get_bm25_index
from ingestion.lexical.bm25_index import IndexedChunk
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
from shared.schemas import TraditionalResult
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


@dataclass(frozen=True)
class HybridPagesOutcome:
    """Ranked page list for the Hybrid tab (TraditionalResult contract)."""

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
    """Dense + BM25 fusion behind the `mode="hybrid"` API contract."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
        reranker=None,  # noqa: ANN001 — test seam
        *,
        bm25_disabled: bool = False,
    ) -> None:
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._reranker = reranker
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

        logger.info(
            "hybrid_search_completed",
            language=resolved_language,
            effective=kw.effective,
            dense=len(dense_chunks),
            bm25=len(bm25_hits),
            fused=len(fused),
            reranked=reranked,
            pages=total,
            returned=len(results),
            offset=offset,
            limit=page_size,
        )
        return HybridPagesOutcome(
            results=results,
            total=total,
            language=resolved_language,
            dense_candidates=len(dense_chunks),
            bm25_candidates=len(bm25_hits),
            fused_candidates=len(fused),
            reranked=reranked,
            title_boost_applied=True,
            bm25_available=bm25 is not None,
        )
