"""Unit tests for traditional (keyword) search — Phase 4.3."""

from __future__ import annotations

import pytest

from ingestion.lexical.bm25_index import IndexedChunk
from services.search_service.traditional import (
    TraditionalSearchService,
    build_snippet,
    group_pages,
    light_stem,
    light_stem_ar,
    light_stem_en,
    normalize_scores,
    prepare_keyword_query,
)
from shared.config import settings
from shared.retrieval_mode import RetrievalMode, business_rules_active, gate_active, parse_retrieval_mode


# ---------------------------------------------------------------------------
# Phase 0 — KEYWORD retrieval mode
# ---------------------------------------------------------------------------


def test_parse_keyword_mode():
    assert parse_retrieval_mode("KEYWORD") is RetrievalMode.KEYWORD
    assert parse_retrieval_mode("keyword") is RetrievalMode.KEYWORD


def test_keyword_mode_has_no_gate_and_no_business_rules():
    assert not gate_active(RetrievalMode.KEYWORD)
    assert not business_rules_active(RetrievalMode.KEYWORD)
    # Enterprise behavior must be untouched.
    assert gate_active(RetrievalMode.ENTERPRISE)
    assert business_rules_active(RetrievalMode.ENTERPRISE)
    assert not gate_active(RetrievalMode.PURE_SEMANTIC)


def test_existing_modes_still_parse():
    assert parse_retrieval_mode("PURE_SEMANTIC") is RetrievalMode.PURE_SEMANTIC
    assert parse_retrieval_mode("ENTERPRISE") is RetrievalMode.ENTERPRISE


def test_config_keyword_defaults():
    assert settings.keyword_search_enabled is True
    assert settings.keyword_results_limit == 10
    assert settings.keyword_results_max_limit == 30


# ---------------------------------------------------------------------------
# Light stemming (dual-token, query-time only)
# ---------------------------------------------------------------------------


def test_arabic_prefix_stripping():
    # Prefix and suffix both apply — the stem of البطاقات is بطاق.
    # Bare ه is also a suffix, so بالجنيه → جني (raw token جنيه is always
    # kept alongside, so this only widens recall, never replaces the raw form).
    assert light_stem_ar("البطاقات") == "بطاق"
    assert light_stem_ar("والشهادات") == "شهاد"
    assert light_stem_ar("بالجنيه") == "جني"


def test_arabic_suffix_stripping():
    assert light_stem_ar("بطاقات") == "بطاق"
    assert light_stem_ar("الحسابات") == "حساب"
    assert light_stem_ar("قرضها") == "قرض"


def test_arabic_stem_matches_inflected_form():
    # Real path normalizes first (ة→ه): بطاقة → بطاقه → بطاق.
    from shared.arabic_normalize import normalize_arabic

    assert light_stem_ar(normalize_arabic("بطاقة")) == light_stem_ar(normalize_arabic("بطاقات"))


def test_english_plural_stripping():
    assert light_stem_en("cards") == "card"
    assert light_stem_en("accounts") == "account"
    assert light_stem_en("certificates") == "certificate"
    assert light_stem_en("classes") == "class"
    assert light_stem_en("card") == "card"
    assert light_stem_en("class") == "class"  # no over-stripping of ss
    # Short words keep their final s (4 chars remaining minimum).
    assert light_stem_en("rates") == "rate"
    assert light_stem_en("fees") == "fees"


def test_light_stem_dispatch():
    assert light_stem("بطاقات", "ar") == "بطاق"
    assert light_stem("cards", "en") == "card"


# ---------------------------------------------------------------------------
# Query preparation
# ---------------------------------------------------------------------------


def test_prepare_query_strips_stopwords_and_adds_stems():
    kw = prepare_keyword_query("how to open a bank account", "en")
    assert kw.effective
    assert kw.language == "en"
    tokens = kw.bm25_query.split()
    assert "how" not in tokens and "to" not in tokens
    assert "account" in tokens
    # dual-token: stems appended
    assert any(token.startswith("open") for token in tokens)


def test_prepare_query_arabic_inflection():
    kw = prepare_keyword_query("بطاقات ائتمان", "ar")
    assert kw.effective
    assert "بطاق" in kw.bm25_query  # stem present → matches بطاقة documents
    assert "بطاقات" in kw.bm25_query  # raw token preserved


def test_prepare_query_stopwords_only_is_ineffective():
    kw = prepare_keyword_query("ازاي", "ar")
    assert not kw.effective
    assert kw.bm25_query == ""


def test_prepare_query_keeps_original_for_highlight():
    kw = prepare_keyword_query("بطاقات ائتمان", "ar")
    assert kw.original_query == "بطاقات ائتمان"


# ---------------------------------------------------------------------------
# Snippets and highlighting terms
# ---------------------------------------------------------------------------


def test_build_snippet_contains_term():
    text = "شهادة الادخار تمنح عائد شهري ويمكن شراء شهادة الادخار من أي فرع."
    snippet = build_snippet(text, ["شهاده", "شهادة"], "ar")
    assert "شهاد" in snippet


def test_build_snippet_fallback_without_hits():
    snippet = build_snippet("Plain opening text of the page here.", ["zzzz"], "en")
    # English snippets come from the lowercased haystack.
    assert snippet.startswith("plain opening")


def test_snippet_length_is_bounded():
    text = "word " * 400
    snippet = build_snippet(text, ["word"], "en", max_chars=240)
    assert len(snippet) <= 260


# ---------------------------------------------------------------------------
# Page grouping, normalization
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, url: str, text: str = "text", title: str = "Title") -> IndexedChunk:
    return IndexedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        title=title,
        url=url,
        language="en",
        text=text,
        doc_type="product",
        category="cards",
        is_stub=False,
        canonical_url_slug="",
    )


def test_group_pages_keeps_best_chunk_per_url():
    url = "https://www.nbe.com.eg/EN/CreditCards"
    hits = [(_chunk("a", url, "low"), 1.0), (_chunk("b", url, "high"), 5.0), (_chunk("c", url), 3.0)]
    pages = group_pages(hits, terms=["credit"], language="en")
    assert len(pages) == 1
    assert pages[0][0].chunk_id == "b"


def test_group_pages_title_boost():
    url = "https://www.nbe.com.eg/EN/CreditCards"
    plain = _chunk("plain", url, title="Some page", text="credit stuff")
    titled = _chunk("titled", url + "-x", title="Credit Cards page", text="other stuff")
    pages = group_pages([(plain, 1.0), (titled, 1.0)], terms=["credit"], language="en")
    assert pages[0][0].chunk_id == "titled"


def test_group_pages_phrase_bonus_lifts_exact_title():
    # Bag-of-words BM25 scores both pages alike; the verbatim bigram in the
    # title must lift the intended page (stretch item 4.5).
    belady = _chunk("belady", "u-belady", title="bank - شهادات بلادي", text="شهادات بلادي سنه")
    sibling = _chunk("sibling", "u-sibling", title="bank - شهادات الادخار", text="شهادات ادخار عائد")
    pages = group_pages(
        [(belady, 10.0), (sibling, 10.05)],
        terms=["شهادات", "بلادي"],
        language="ar",
        phrases=["شهادات بلادي"],
    )
    assert pages[0][0].chunk_id == "belady"


def test_group_pages_without_phrases_is_backward_compatible():
    a = _chunk("a", "u1", title="t", text="x")
    pages = group_pages([(a, 1.0)], terms=["t"], language="en")
    assert len(pages) == 1


# ---------------------------------------------------------------------------
# Dominant title similarity
# ---------------------------------------------------------------------------


def test_group_pages_title_similarity_beats_body_frequency():
    # User-reported case: "Platinum" must rank the Platinum page first even
    # though the Exclusive-Products body mentions "platinum" more often.
    exclusive = _chunk(
        "excl",
        "u-excl",
        title="exclusive products",
        text="platinum " * 40,
    )
    platinum = _chunk(
        "plat",
        "u-plat",
        title="national bank of egypt - platinum",
        text="platinum card benefits",
    )
    pages = group_pages(
        [(exclusive, 10.0), (platinum, 5.0)],
        terms=["platinum"],
        language="en",
    )
    assert pages[0][0].chunk_id == "plat"


def test_group_pages_full_title_match_wins_on_tie():
    a = _chunk("a", "u-a", title="savings certificates", text="rate stuff")
    b = _chunk("b", "u-b", title="some other page", text="savings certificates rate stuff savings")
    pages = group_pages(
        [(a, 3.0), (b, 3.0)],
        terms=["savings", "certificates"],
        language="en",
    )
    assert pages[0][0].chunk_id == "a"


def test_title_similarity_ignores_corporate_prefix_tokens():
    # "National Bank of Egypt - Platinum" must be judged by "platinum":
    # the corporate words would otherwise dilute coverage to 1/4 everywhere.
    from services.search_service.traditional import _title_match_tokens

    assert _title_match_tokens(
        ["national", "bank", "egypt", "platinum"], "en"
    ) == ["platinum"]
    # A query that is ONLY corporate words keeps them (no dilution, no crash).
    assert _title_match_tokens(["bank"], "en") == ["bank"]
    assert _title_match_tokens([], "en") == []


def test_service_platinum_page_ranks_first_on_real_index():
    svc = TraditionalSearchService()
    outcome = svc.search("National Bank of Egypt - Platinum", "en", limit=5)
    assert outcome.results, "expected results"
    assert outcome.results[0].title == "National Bank of Egypt - Platinum"
    outcome_short = svc.search("Platinum", "en", limit=5)
    assert outcome_short.results[0].title == "National Bank of Egypt - Platinum"


def test_exact_title_express_tier_beats_body_heavy_page():
    # The intended page is NOT a BM25 candidate (its body/rank would place it
    # far below); only the corpus-wide exact-title scan can surface it at #1.
    gold = IndexedChunk(
        chunk_id="gold", document_id="d-title",
        title="National Bank of Egypt - Platinum",
        url="https://www.nbe.com.eg/EN/Platinum", language="en",
        text="cards overview", doc_type="product", category="cards",
        is_stub=False, canonical_url_slug="",
    )
    body_heavy = IndexedChunk(
        chunk_id="bulk", document_id="d-bulk",
        title="Exclusive Products",
        url="https://www.nbe.com.eg/EN/Exclusive", language="en",
        text="platinum " * 30, doc_type="product", category="cards",
        is_stub=False, canonical_url_slug="",
    )
    # BM25 sees ONLY the body-heavy page; `corpus` gives the exact-title
    # scanner the full page collection it would see on the real index.
    fake = _FakeBM25({"en": [(body_heavy, 20.0)]}, corpus=[gold, body_heavy])
    svc = TraditionalSearchService(bm25_index=fake)
    outcome = svc.search("National Bank of Egypt - Platinum", "en", limit=5)
    titles = [r.title for r in outcome.results]
    assert titles[0] == "National Bank of Egypt - Platinum"
    assert "Exclusive Products" in titles  # still present, just below


def test_exact_title_express_tier_fires_without_bm25_candidates():
    gold = IndexedChunk(
        chunk_id="gold", document_id="d-title",
        title="National Bank of Egypt - Platinum",
        url="https://www.nbe.com.eg/EN/Platinum", language="en",
        text="cards overview", doc_type="product", category="cards",
        is_stub=False, canonical_url_slug="",
    )
    filler = IndexedChunk(
        chunk_id="fill", document_id="d-fill", title="Loan page",
        url="https://www.nbe.com.eg/EN/Loans", language="en", text="loan",
        doc_type="product", category="loans", is_stub=False, canonical_url_slug="",
    )
    # A non-empty BM25 hit list keeps the index "available" (size > 0) while
    # the express tier is the only thing that can surface the gold page.
    fake = _FakeBM25({"en": [(filler, 9.0)]}, corpus=[gold, filler])
    svc = TraditionalSearchService(bm25_index=fake)
    outcome = svc.search("National Bank of Egypt - Platinum", "en", limit=5)
    assert outcome.results[0].title == "National Bank of Egypt - Platinum"


def test_exact_title_express_generic_query_no_match():
    # Queries without an identifying tail must not hijack generic pages: a
    # bare "bank" query keeps its tokens and never enters the express tier.
    home = IndexedChunk(
        chunk_id="home", document_id="d-home",
        title="National Bank of Egypt - Home",
        url="https://www.nbe.com.eg/EN/Home", language="en",
        text="welcome", doc_type="page", category="home",
        is_stub=False, canonical_url_slug="",
    )
    fake = _FakeBM25({"en": []}, corpus=[home])
    svc = TraditionalSearchService(bm25_index=fake)
    outcome = svc.search("bank", "en", limit=5)
    assert outcome.results == []  # express tier must stay empty


def test_normalize_scores_range():
    pairs = [(_chunk("a", "u1"), 3.2), (_chunk("b", "u2"), 9.7)]
    normalized = normalize_scores(pairs)
    scores = [score for _, score in normalized]
    assert max(scores) == 1.0
    assert min(scores) == pytest.approx(0.05)


def test_normalize_scores_single_and_empty():
    assert normalize_scores([]) == []
    single = normalize_scores([(_chunk("a", "u1"), 4.2)])
    assert single[0][1] == 1.0


# ---------------------------------------------------------------------------
# End-to-end service on the real BM25 index
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def service() -> TraditionalSearchService:
    return TraditionalSearchService()


def test_service_english_results(service: TraditionalSearchService):
    outcome = service.search("how to open a bank account", "en", limit=5)
    assert outcome.effective_query
    assert outcome.total > 0
    assert len(outcome.results) <= 5
    top = outcome.results[0]
    assert top.title
    assert top.url
    assert top.language == "en"
    assert 0.0 < top.score <= 1.0
    assert top.terms  # highlight terms present


def test_service_arabic_dual_token_matches_inflection(service: TraditionalSearchService):
    outcome = service.search("بطاقات ائتمان", "ar", limit=5)
    assert outcome.total > 0
    assert all(result.language == "ar" for result in outcome.results)


def test_service_dedupes_pages(service: TraditionalSearchService):
    outcome = service.search("credit cards", "en", limit=30)
    urls = [result.url for result in outcome.results]
    assert len(urls) == len(set(urls))


def test_service_pagination(service: TraditionalSearchService):
    page1 = service.search("certificate", "en", limit=3, offset=0)
    page2 = service.search("certificate", "en", limit=3, offset=3)
    urls1 = {r.url for r in page1.results}
    urls2 = {r.url for r in page2.results}
    assert urls1 and urls2
    assert not urls1 & urls2


def test_service_stopword_only_returns_empty(service: TraditionalSearchService):
    outcome = service.search("ازاي", "ar")
    assert outcome.results == []
    assert outcome.total == 0
    assert not outcome.effective_query


def test_service_nonsense_returns_empty_but_effective(service: TraditionalSearchService):
    outcome = service.search("asdkjhqwe zzz qwerty", "en")
    assert outcome.effective_query
    assert outcome.results == []


def test_service_limit_is_capped(service: TraditionalSearchService):
    outcome = service.search("certificate", "en", limit=999)
    assert len(outcome.results) <= settings.keyword_results_max_limit


def test_prepare_query_spacing_aliases_compound_brand_words():
    """"alahly points" must reach BM25 as "al ahly points" — the corpus
    stores "Al Ahly" as two tokens, so the compound never matched."""
    kq = prepare_keyword_query("alahly points", "en")
    assert "al ahly points" in kq.bm25_query
    assert "alahly" not in kq.bm25_query.split()


def test_prepare_query_spacing_aliases_leave_normal_queries_unchanged():
    kq = prepare_keyword_query("personal loan", "en")
    assert kq.bm25_query.startswith("personal loan")


def test_prepare_query_highlight_terms_exclude_stopwords():
    kw = prepare_keyword_query("Where is the nearest NBE branch?", "en")
    assert "where" not in kw.terms_by_language["en"]
    assert "the" not in kw.terms_by_language["en"]
    assert "branch" in kw.terms_by_language["en"]


class _FakeBM25:
    """Minimal BM25Index stand-in for API-contract tests."""

    def __init__(
        self,
        hits: dict[str, list[tuple[IndexedChunk, float]]],
        *,
        corpus: list[IndexedChunk] | None = None,
    ):
        self._fake_hits = hits
        self._all: list[IndexedChunk] = list(corpus) if corpus is not None else [
            chunk for pairs in hits.values() for chunk, _ in pairs
        ]
        self.size = sum(len(v) for v in hits.values())

    def query(self, query_text, language, top_k=50, *, doc_types=None):  # noqa: ANN001, ANN202
        return self._fake_hits.get(language, [])[:top_k]

    def chunks(self) -> list[IndexedChunk]:
        return list(self._all)


def test_service_rejects_non_http_urls():
    bad = IndexedChunk(
        chunk_id="bad", document_id="d1", title="Evil",
        url="javascript:alert(1)", language="en", text="loan content",
        doc_type="product", category="loans", is_stub=False, canonical_url_slug="",
    )
    good = IndexedChunk(
        chunk_id="good", document_id="d2", title="Loans",
        url="https://www.nbe.com.eg/EN/Loans", language="en", text="loan content",
        doc_type="product", category="loans", is_stub=False, canonical_url_slug="",
    )
    svc = TraditionalSearchService(bm25_index=_FakeBM25({"en": [(bad, 5.0), (good, 4.0)]}))
    outcome = svc.search("loan", "en", limit=5)
    urls = [r.url for r in outcome.results]
    assert urls == ["https://www.nbe.com.eg/EN/Loans"]


def test_service_fill_in_is_capped_at_one_window():
    primary = [
        (IndexedChunk(chunk_id=f"p{i}", document_id=f"dp{i}", title=f"P{i}",
                      url=f"https://x.example/ar/{i}", language="ar", text="محتوى",
                      doc_type="product", category="c", is_stub=False, canonical_url_slug=""), 10.0 - i)
        for i in range(2)
    ]
    other = [
        (IndexedChunk(chunk_id=f"o{i}", document_id=f"do{i}", title=f"O{i}",
                      url=f"https://x.example/en/{i}", language="en", text="content",
                      doc_type="product", category="c", is_stub=False, canonical_url_slug=""), 10.0 - i)
        for i in range(25)
    ]
    svc = TraditionalSearchService(bm25_index=_FakeBM25({"ar": primary, "en": other}))
    outcome = svc.search("محتوى", "ar", limit=10)
    # 2 primary pages + at most one 10-result window of fill-in.
    assert outcome.total == 12
    assert [r.language for r in outcome.results[:2]] == ["ar", "ar"]
    assert all(r.language == "en" for r in outcome.results[2:])


def test_service_index_unavailable_flag():
    svc = TraditionalSearchService(bm25_index=_FakeBM25({"en": []}))
    outcome = svc.search("loan", "en", limit=5)
    assert outcome.results == []
    assert outcome.effective_query
    assert not outcome.bm25_available


def test_service_language_fill_in_from_other_language(service: TraditionalSearchService):
    # An AR-only brand/technical query with (almost) no AR hits should fill
    # from EN pages instead of returning nothing.
    outcome = service.search("tariff fees", "ar", limit=10)
    if outcome.total < 5:
        languages = {result.language for result in outcome.results}
        assert "en" in languages or outcome.total == 0
