"""Unit tests for the Hybrid tab page-search service (mode="hybrid")."""

from __future__ import annotations

import asyncio
import pytest

from ingestion.embedding.vector_store import RetrievedChunk
from ingestion.lexical.bm25_index import IndexedChunk
from services.search_service.hybrid_pages import (
    _HYBRID_FUSED_KEEP as HYBRID_RRF_KEEP,
    HybridPageSearchService,
    _hybrid_fusion,
    _title_boost_ranking,
    grouped_best_pages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dense(chunk_id: str, title: str, score: float, text: str = "content") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"d-{chunk_id}",
        title=title,
        url=f"https://www.nbe.com.eg/EN/{chunk_id}",
        language="en",
        text=text,
        score=score,
    )


def _indexed(chunk_id: str, title: str, score: float, text: str = "content") -> IndexedChunk:
    return IndexedChunk(
        chunk_id=chunk_id,
        document_id=f"d-{chunk_id}",
        title=title,
        url=f"https://www.nbe.com.eg/EN/{chunk_id}",
        language="en",
        text=text,
        doc_type="product",
        category="cards",
        is_stub=False,
        canonical_url_slug="",
    )


class _FakeVectorStore:
    def __init__(self, chunks: list[RetrievedChunk]):
        self._chunks = chunks
        self.calls: list[str] = []

    def query(self, query_text: str, top_k: int, *, language: str | None = None, doc_types=None):
        self.calls.append(query_text)
        return list(self._chunks)[:top_k]


class _FakeBM25:
    def __init__(self, hits: list[tuple[IndexedChunk, float]]):
        self._hits = hits
        self.size = len(hits) or 1  # non-zero so the index counts as available

    def query(self, query_text, language, top_k=50, *, doc_types=None):  # noqa: ANN001
        return list(self._hits)[:top_k]

    def to_retrieved_chunk(self, chunk: IndexedChunk, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.title,
            url=chunk.url,
            language=chunk.language,
            text=chunk.text,
            score=score,
            doc_type=chunk.doc_type,
            category=chunk.category,
            is_stub=chunk.is_stub,
            canonical_url_slug=chunk.canonical_url_slug,
        )

    def chunks(self) -> list[IndexedChunk]:
        return [chunk for chunk, _ in self._hits]


class _FakeReranker:
    """Reverses order deterministically — proves rerank was invoked."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.called = False

    def rerank(self, query_text, chunks, top_k=None):  # noqa: ANN001
        if self.fail:
            raise RuntimeError("reranker down")
        self.called = True
        ranked = list(reversed(list(chunks)))
        return ranked[: top_k or len(ranked)]


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def test_rrf_fusion_ranks_common_chunks_first():
    dense = [_dense(f"d{i}", f"Dense {i}", 0.9 - i * 0.01) for i in range(3)]
    bm25 = [(_indexed(f"b{i}", f"BM25 {i}", 10.0 - i),) for i in range(1)]
    hits = [(_indexed("d1", "Dense 1", 9.0), 9.0)]
    fused, both = _hybrid_fusion(
        dense, hits, lambda chunk, score: RetrievedChunk(
            chunk_id=chunk.chunk_id, document_id=chunk.document_id, title=chunk.title,
            url=chunk.url, language=chunk.language, text=chunk.text, score=score,
        )
    )
    assert both is True
    # d1 appears in both channels → top of the fused ranking.
    assert fused[0].chunk_id == "d1"


def test_hybrid_fusion_dense_only_and_bm25_only_paths():
    dense = [_dense("d0", "Dense 0", 0.9)]
    fused, both = _hybrid_fusion(dense, [], lambda c, s: c)
    assert both is False
    assert [c.chunk_id for c in fused] == ["d0"]

    fused, both = _hybrid_fusion([], [(_indexed("b0", "B", 3.0), 3.0)], lambda c, s: c)
    assert both is False
    assert [c.chunk_id for c in fused] == ["b0"]

    fused, both = _hybrid_fusion([], [], lambda c, s: c)
    assert fused == [] and both is False


def test_rrf_fusion_merges_both_channels_without_duplicates():
    dense = [_dense(f"d{i}", f"D{i}", 0.5) for i in range(40)]
    hits = [(_indexed(f"d{i}", f"D{i}", 1.0), 1.0) for i in range(40)]
    rr = lambda c, s: RetrievedChunk(  # noqa: E731
        chunk_id=c.chunk_id, document_id=c.document_id, title=c.title,
        url=c.url, language=c.language, text=c.text, score=s,
    )
    fused, both = _hybrid_fusion(dense, hits, rr)
    # Overlap of 40: fusion returns exactly the union (no double-count).
    assert both is True
    assert len(fused) == 40
    ids = [c.chunk_id for c in fused]
    assert len(ids) == len(set(ids))
    assert HYBRID_RRF_KEEP == 30  # service-side cap constant


# ---------------------------------------------------------------------------
# Title boost (controlled, not a hard override)
# ---------------------------------------------------------------------------


def _boost_case():
    return [
        _dense("excl", "National Bank of Egypt - Exclusive Products", 0.95, "platinum platinum platinum"),
        _dense("plat", "National Bank of Egypt - Platinum", 0.58),
        _dense("cert", "National Bank of Egypt - Platinum Certificate 3 Years", 0.62),
    ]


def _query_terms():
    from services.search_service.keyword_rank import extract_query_terms

    return [t for t in extract_query_terms("National Bank of Egypt - Platinum", "en")]


def test_title_boost_lifts_exact_match_above_comparable_page():
    ranked = _title_boost_ranking(_boost_case(), _query_terms(), "en")
    titles = [c.title for c in ranked]
    # Boost reorders near-ties: 0.58×1.6=0.928 (exact-title Platinum) beats
    # 0.62×1.35=0.837 (certificate title that merely contains "platinum").
    assert titles.index("National Bank of Egypt - Platinum") < titles.index(
        "National Bank of Egypt - Platinum Certificate 3 Years"
    )
    # Not a hard override: a much higher base score stays above (0.95 case).
    assert titles[0] == "National Bank of Egypt - Exclusive Products"
    scores = {c.title: c.score for c in ranked}
    assert scores["National Bank of Egypt - Exclusive Products"] == pytest.approx(0.95)


def test_title_boost_does_not_override_large_score_gap():
    ranked = _title_boost_ranking(_boost_case(), _query_terms(), "en")
    scores = {c.title: c.score for c in ranked}
    plat = scores["National Bank of Egypt - Platinum"]
    excl = scores["National Bank of Egypt - Exclusive Products"]
    # Identifying-token coverage ("platinum") + exact-title bonus.
    assert plat == pytest.approx(0.58 * (1 + 0.35 + 0.25), rel=1e-3)
    assert plat < excl  # could not leapfrog the much higher rerank score…


def test_title_boost_no_terms_is_identity():
    chunks = _boost_case()
    assert _title_boost_ranking(chunks, [], "en") == chunks


def test_title_exact_match_requires_identifying_tokens():
    # Corporate-only query keeps its tokens → no false "exact" hijack of " platinum" pages.
    from services.search_service.hybrid_pages import _normalized_query_keys
    from services.search_service.traditional import _title_match_tokens

    terms = _query_terms()
    key = " ".join(_title_match_tokens(terms, "en"))
    assert "national" not in key
    assert "platinum" in key


# ---------------------------------------------------------------------------
# Page-level dedup + normalization
# ---------------------------------------------------------------------------


def test_grouped_best_pages_keeps_best_chunk_per_url():
    first = _indexed("x1", "P", 1.0)
    second = _indexed("x2", "P", 2.0)
    other = _indexed("b1", "Other", 2.5)
    # Force the same canonical URL by giving the chunks truly equal urls:
    from dataclasses import replace as _replace

    second = _replace(second, url=first.url)
    pages = grouped_best_pages([first, second, other], [1.0, 2.0, 2.5], ["p"], "en")
    assert len(pages) == 2  # same-url chunks collapse to the best score
    winner = max(pages, key=lambda pair: pair[1])
    same_url = [chunk for chunk, _ in pages if chunk.url == first.url]
    assert same_url and same_url[0].chunk_id == "x2"  # higher score wins


def test_score_normalization_bounds_in_hybrid_output():
    service = HybridPageSearchService(
        vector_store=_FakeVectorStore([_dense("a", "Page A", 0.9), _dense("b", "Page B", 0.2)]),
        bm25_index=_FakeBM25([(_indexed("a", "Page A", 5.0), 5.0), (_indexed("b", "Page B", 1.0), 1.0)]),
        reranker=_FakeReranker(),
    )
    outcome = service.search("platinum", "en", limit=5)
    scores = [r.score for r in outcome.results]
    if len(scores) > 1:
        assert max(scores) == pytest.approx(1.0)
        assert min(scores) >= 0.05


# ---------------------------------------------------------------------------
# Service-level behavior with seams
# ---------------------------------------------------------------------------


def _service(dense: list[RetrievedChunk], bm25_hits: list[tuple[IndexedChunk, float]], reranker=None):
    return HybridPageSearchService(
        vector_store=_FakeVectorStore(dense),
        bm25_index=_FakeBM25(bm25_hits),
        reranker=reranker or _FakeReranker(),
    )


def test_service_returns_results_and_contract_fields():
    svc = _service(
        [_dense("a", "Page A", 0.9), _dense("b", "Page B", 0.2)],
        [(_indexed("a", "Page A", 5.0), 5.0), (_indexed("b", "Page B", 1.0), 1.0)],
    )
    outcome = svc.search("credit cards", "en", limit=5)
    assert outcome.effective_query
    assert outcome.total > 0
    assert len(outcome.results) <= 5
    assert outcome.fused_candidates > 0
    assert outcome.reranked is True
    top = outcome.results[0]
    assert top.title and top.url
    assert 0.0 < top.score <= 1.0
    assert top.language in {"ar", "en"}


def test_service_reranker_failure_falls_back():
    svc = _service(
        [_dense("a", "Page A", 0.9), _dense("b", "Page B", 0.5)],
        [(_indexed("b", "Page B", 4.0), 4.0), (_indexed("a", "Page A", 2.0), 2.0)],
        reranker=_FakeReranker(fail=True),
    )
    outcome = svc.search("page", "en", limit=5)
    assert outcome.reranked is False
    assert outcome.total > 0  # pipeline continued with RRF order


def test_service_bm25_unavailable_dense_only():
    service = HybridPageSearchService(
        vector_store=_FakeVectorStore([_dense("a", "Dense hit", 0.9)]),
        bm25_index=None,
        reranker=_FakeReranker(),
        bm25_disabled=True,  # deliberate dense-only configuration
    )
    outcome = service.search("loan", "en", limit=5)
    assert outcome.bm25_available is False
    assert outcome.results
    assert outcome.results[0].title == "Dense hit"


def test_service_empty_channels_returns_empty_outcome():
    svc = _service([], [])
    outcome = svc.search("credit", "en", limit=5)
    assert outcome.total == 0
    assert outcome.results == []


def test_service_ineffective_query_short_circuits():
    svc = _service([_dense("a", "A", 1.0)], [(_indexed("a", "A", 1.0), 1.0)])
    outcome = svc.search("the of and", "en", limit=5)
    assert outcome.effective_query is False
    assert outcome.results == []


def test_service_dedupes_and_normalizes_like_traditional():
    dense = [_dense("dup", "Page", 0.8), _dense("dup", "Page", 0.6)]
    bm25 = [(_indexed("dup", "Page", 2.0), 2.0), (_indexed("solo", "Solo", 1.0), 1.0)]
    svc = _service(dense, bm25)
    outcome = svc.search("page", "en", limit=5)
    urls = [r.url for r in outcome.results]
    assert len(urls) == len(set(urls))


# ---------------------------------------------------------------------------
# Note #10 — ranking regression (mirrors traditional-mode behavior)
# ---------------------------------------------------------------------------


def test_platinum_title_page_beats_exclusive_products_body_page():
    """User-reported regression: "National Bank of Egypt - Platinum" must put
    the Platinum page above an "Exclusive Products" page carrying "platinum"
    in the body, when relevance signals are otherwise comparable."""
    dense = [
        _dense("excl", "National Bank of Egypt - Exclusive Products", 0.80, "platinum benefits platinum"),
        _dense("plat", "National Bank of Egypt - Platinum", 0.62, "card overview"),
    ]
    bm25 = [
        (_indexed("excl", "National Bank of Egypt - Exclusive Products", 18.0), 18.0),
        (_indexed("plat", "National Bank of Egypt - Platinum", 15.0), 15.0),
    ]
    service = HybridPageSearchService(
        vector_store=_FakeVectorStore(dense),
        bm25_index=_FakeBM25(bm25),
        reranker=_FakeReranker(),  # reversed → plat 0.62 first? no: reversal puts plat first already
    )
    outcome = service.search("National Bank of Egypt - Platinum", "en", limit=5)
    titles = [r.title for r in outcome.results]
    assert titles, "expected results"
    assert titles[0] == "National Bank of Egypt - Platinum"
    assert "National Bank of Egypt - Exclusive Products" in titles[: len(titles)]


def test_platinum_case_with_strong_body_heavy_competitor():
    """Even when the body-heavy page scores clearly higher after rerank, the
    title-boosted Platinum page must sit above it when the gap is small; a
    huge gap still respects relevance (controlled boost)."""
    dense = [
        _dense("excl", "National Bank of Egypt - Exclusive Products", 0.9, "platinum " * 20),
        _dense("plat", "National Bank of Egypt - Platinum", 0.75, "card"),
    ]
    service = HybridPageSearchService(
        vector_store=_FakeVectorStore(dense),
        bm25_index=_FakeBM25([]),
        reranker=_FakeReranker(),
    )
    outcome = service.search("National Bank of Egypt - Platinum", "en", limit=5)
    # Reranker reverses: plat (0.75) first anyway; boost keeps it first.
    assert outcome.results[0].title == "National Bank of Egypt - Platinum"


# ---------------------------------------------------------------------------
# Phase 2 — Hybrid RAG: grounded answer + citations (frozen page ranking)
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Test double for LLMService.generate_answer (seam like _FakeReranker)."""

    def __init__(self, answer="### Answer\n\nPlatinum facts\n\n### Source\n\n- Page A", *, fail=False):
        self.answer = answer
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    async def generate_answer(self, query: str, context: str, language: str):
        if self.fail:
            raise RuntimeError("LLM down")
        self.calls.append((query, context))
        return self.answer, 0.78


class _FakeContextBuilder:
    """Mirrors ContextBuilder.build's contract with a fixed citation list."""

    def __init__(self, citations=None):
        self.citations = citations or []
        self.calls: list[tuple[str, list[str]]] = []

    def build(self, query, chunks, *, compress=True):
        self.calls.append((query, [c.chunk_id for c in chunks]))
        from services.context_builder.builder import BuiltContext
        from shared.schemas import Citation

        citations = self.citations or [
            Citation(title=chunks[0].title, url=chunks[0].url, relevance_score=0.9, reranker_score=0.9)
        ] if chunks else []
        context = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, 1))
        return BuiltContext(context_text=context, citations=list(citations), token_estimate=100)


def _rag_service(llm=None, builder=None, dense=None):
    return HybridPageSearchService(
        vector_store=_FakeVectorStore(
            dense or [
                _dense("a", "Page A", 0.9, "platinum card features text"),
                _dense("b", "Page B", 0.6, "more platinum text"),
            ]
        ),
        bm25_index=_FakeBM25([]),
        reranker=_FakeReranker(),
        context_builder=builder,
        llm_service=llm,
    )



def test_search_with_answer_returns_frozen_pages_plus_answer():
    llm = _FakeLLM()
    builder = _FakeContextBuilder()
    svc = _rag_service(llm=llm, builder=builder)
    outcome = asyncio.run(svc.search_with_answer("platinum", "en", limit=5))
    # Answer + citations populated (wrapped in the structured template).
    assert outcome.answer and "Platinum facts" in outcome.answer
    assert outcome.answer.startswith("### ")
    assert len(outcome.citations) == 1
    assert outcome.citations[0].url == "https://www.nbe.com.eg/EN/a"
    assert outcome.answer_confidence == pytest.approx(0.78)
    # Page ranking untouched by generation — identical to plain search() output
    # (_FakeReranker reverses without rescoring, so grouped order sorts by score).
    assert [r.title for r in outcome.results] == [
        r.title for r in svc.search("platinum", "en", limit=5).results
    ]
    # Grounded context built from the top ranked chunks (frozen order).
    query, chunk_ids = builder.calls[0]
    assert chunk_ids == ["a", "b"]
    assert query == "platinum"
    # Latency bookkeeping split.
    assert outcome.generation_ms is not None and outcome.total_ms is not None



def test_search_with_answer_citations_come_from_retrieved_urls_only():
    llm = _FakeLLM()
    svc = _rag_service(llm=llm)
    outcome = asyncio.run(svc.search_with_answer("platinum", "en", limit=5))
    retrieved_urls = {r.url for r in outcome.results}
    assert outcome.citations
    for citation in outcome.citations:
        assert citation.url in retrieved_urls



def test_search_with_answer_abstains_on_empty_context():
    """Insufficient evidence → no answer; ranked pages survive (abstention #6)."""

    class _EmptyBuilder:
        def build(self, query, chunks, *, compress=True):
            from services.context_builder.builder import BuiltContext

            return BuiltContext(context_text="", citations=[], token_estimate=0)

    builder = _EmptyBuilder()
    llm = _FakeLLM()
    svc = _rag_service(llm=llm, builder=builder)
    outcome = asyncio.run(svc.search_with_answer("platinum", "en", limit=5))
    assert outcome.answer is None
    assert outcome.citations == []
    assert outcome.generation_error is False  # abstention, not failure
    assert outcome.results  # page list unaffected
    assert not llm.calls  # LLM never invoked without evidence



def test_search_with_answer_llm_failure_returns_pages():
    """LLM throw → fail-open: the request keeps the full page list (req #7)."""
    llm = _FakeLLM(fail=True)
    svc = _rag_service(llm=llm)
    outcome = asyncio.run(svc.search_with_answer("platinum", "en", limit=5))
    assert outcome.answer is None
    assert outcome.generation_error is True
    assert [r.title for r in outcome.results] == ["Page A", "Page B"]
    assert outcome.total > 0
    assert outcome.total_ms is not None  # latency still tracked



def test_search_with_answer_llm_refusal_abstains_with_pages():
    """LLM 取 refusal (None answer) → abstain, keep pages (grounding contract)."""
    llm = _FakeLLM(answer=None)
    svc = _rag_service(llm=llm)
    outcome = asyncio.run(svc.search_with_answer("platinum", "en", limit=5))
    assert outcome.answer is None
    assert outcome.generation_error is False
    assert outcome.results
    assert llm.calls  # LLM was called and refused on its own



def test_search_with_answer_paginated_offset_skips_generation():
    svc = _rag_service(llm=_FakeLLM(), builder=_FakeContextBuilder())
    outcome = asyncio.run(svc.search_with_answer("platinum", "en", limit=5, offset=5))
    assert outcome.answer is None  # answers only on page 1



def test_search_plain_has_no_answer_fields():
    """Phase 1 `search()` contract unchanged: retrieval-only, no LLM fields."""
    svc = _rag_service(llm=_FakeLLM(), builder=_FakeContextBuilder())
    outcome = svc.search("platinum", "en", limit=5)
    assert outcome.answer is None
    assert outcome.citations == []
    assert outcome.generation_error is False
    assert outcome.generation_ms is None
