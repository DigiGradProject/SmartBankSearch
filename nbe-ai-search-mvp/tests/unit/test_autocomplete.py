from services.search_service.autocomplete import build_autocomplete


def test_autocomplete_popular_when_empty():
    suggestions = build_autocomplete("", "ar", limit=5)
    assert len(suggestions) == 5
    assert all(item.reason == "popular" for item in suggestions)


def test_autocomplete_prefix_match_arabic():
    suggestions = build_autocomplete("شهاد", "ar", limit=5)
    assert len(suggestions) >= 1
    assert any("شهاد" in item.label for item in suggestions)


def test_autocomplete_colloquial_arabic():
    suggestions = build_autocomplete("عايز", "ar", limit=5)
    assert len(suggestions) >= 1
    assert any("شهاد" in item.query or "حساب" in item.query for item in suggestions)


def test_short_arabic_prefix_still_suggests():
    """2-char Arabic input must not be misdetected as English (regression:
    autocomplete used a heuristic detector that required >=3 Arabic chars)."""
    from services.search_service.autocomplete import build_autocomplete

    suggestions = build_autocomplete("شه", "auto", limit=5, catalog_only=True)
    assert suggestions, "short Arabic prefix must return catalog suggestions"
    assert all(s.label for s in suggestions)


def test_traditional_mode_bm25_title_matches(monkeypatch):
    """catalog_only (traditional) mode must suggest real indexed page titles
    via the standalone BM25 index — no embedder, no vector store."""
    from ingestion.lexical.bm25_index import BM25Index, IndexedChunk

    from shared.schemas import ChunkRecord

    index = BM25Index()
    index.build([
        ChunkRecord(
            chunk_id="c1", document_id="d1", chunk_index=0, content_hash="h1",
            title="Auto Loan", url="https://nbe.com.eg/auto",
            language="en", text="Auto loan interest and eligibility", doc_type="page",
            category="", is_stub=False, canonical_url_slug="",
        ),
        ChunkRecord(
            chunk_id="c2", document_id="d2", chunk_index=1, content_hash="h2",
            title="بيانات الحساب", url="https://nbe.com.eg/accounts",
            language="ar", text="بيانات الحساب الجاري", doc_type="page",
            category="", is_stub=False, canonical_url_slug="",
        ),
    ])

    # get_bm25_index() is a process-wide singleton; patch it to return the
    # freshly-built test index so unit tests never load the real pickle.
    import services.search_service.autocomplete as ac
    monkeypatch.setattr(ac, "get_bm25_index", lambda: index)

    suggestions = ac.build_autocomplete("loan", "en", limit=5, catalog_only=True)
    assert any(s.reason == "bm25_title_match" and "Auto Loan" in s.label for s in suggestions)
    assert any(s.url == "https://nbe.com.eg/auto" for s in suggestions)

    # Arabic substring fallback: "حسا" tokenizes differently than the title
    # token "الحساب" but normalized-substring matching must still hit.
    ar_suggestions = ac.build_autocomplete("حسا", "ar", limit=5, catalog_only=True)
    assert any("حساب" in s.label for s in ar_suggestions)


def test_bm25_title_matches_empty_index_returns_nothing(monkeypatch):
    from ingestion.lexical.bm25_index import BM25Index

    import services.search_service.autocomplete as ac
    monkeypatch.setattr(ac, "get_bm25_index", lambda: BM25Index())

    suggestions = ac.build_autocomplete("loan", "en", limit=5, catalog_only=True)
    assert all(s.reason != "bm25_title_match" for s in suggestions)
