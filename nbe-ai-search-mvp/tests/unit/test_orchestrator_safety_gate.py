import asyncio
from types import SimpleNamespace

from ingestion.embedding.vector_store import RetrievedChunk
from services.api.orchestrator import Orchestrator, merge_retrieval_results
from services.search_service.search import RetrievalResult
from shared.schemas import SearchResponse


class _StaticSearch:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls = 0

    def retrieve(self, *args, **kwargs) -> RetrievalResult:
        self.calls += 1
        return self.result


class _SemanticCache:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.lookup_calls = 0
        self.store_calls = 0

    def lookup(self, query: str, language: str):
        self.lookup_calls += 1
        return SimpleNamespace(payload=self.payload, similarity=1.0)

    def store(self, *args, **kwargs):
        self.store_calls += 1
        return "cache-id"


def _result(*, should_answer: bool, url: str = "https://nbe.example/cards") -> RetrievalResult:
    chunk = RetrievedChunk(
        chunk_id="card::0",
        document_id="card",
        title="بطاقات الائتمان",
        url=url,
        language="ar",
        text="بطاقات الائتمان فيزا كلاسيك",
        score=0.9,
        doc_type="credit_card",
        category="cards",
    )
    return RetrievalResult(
        query="انواع البطاقات البنكيه",
        language="ar",
        chunks=[chunk],
        confidence=0.2 if not should_answer else 0.9,
        should_answer=should_answer,
        abstention_reason=None if should_answer else "low_retrieval_confidence",
        intent="card_types",
        category="cards",
        intent_confidence=0.95,
        decision="ANSWER" if should_answer else "NO_ANSWER",
    )


def _cached_payload(url: str) -> dict:
    return SearchResponse(
        answer="إجابة قديمة من الكاش",
        confidence=0.99,
        citations=[{"title": "cached", "url": url}],
        answered=True,
        language="ar",
    ).model_dump()


def test_should_answer_false_blocks_catalog_and_semantic_cache(monkeypatch, tmp_path):
    monkeypatch.setattr("shared.config.settings.semantic_cache_enabled", True)
    monkeypatch.setattr("shared.config.settings.cache_enabled", False)
    monkeypatch.setattr("shared.config.settings.audit_log_enabled", False)
    monkeypatch.setattr("shared.config.settings.analytics_log_enabled", False)
    cache = _SemanticCache(_cached_payload("https://nbe.example/cards"))
    search = _StaticSearch(_result(should_answer=False))
    orchestrator = Orchestrator(search_service=search, semantic_cache=cache)

    response = asyncio.run(orchestrator.search("انواع البطاقات البنكيه", "ar"))

    assert search.calls == 1
    assert cache.lookup_calls == 0
    assert response.answered is False
    assert response.answer is None
    assert response.abstention_reason == "low_retrieval_confidence"


def test_semantic_cache_requires_current_retrieved_source(monkeypatch):
    monkeypatch.setattr("shared.config.settings.semantic_cache_enabled", True)
    monkeypatch.setattr("shared.config.settings.cache_enabled", False)
    monkeypatch.setattr("shared.config.settings.audit_log_enabled", False)
    monkeypatch.setattr("shared.config.settings.analytics_log_enabled", False)
    cache = _SemanticCache(_cached_payload("https://nbe.example/unrelated"))
    orchestrator = Orchestrator(
        search_service=_StaticSearch(_result(should_answer=True)),
        semantic_cache=cache,
    )

    response = asyncio.run(orchestrator.search("انواع البطاقات البنكيه", "ar"))

    assert cache.lookup_calls == 1
    assert response.cache_hit is False
    assert response.answer != "إجابة قديمة من الكاش"


def test_missing_numeric_rate_abstains_after_retrieval_gate(monkeypatch):
    monkeypatch.setattr("shared.config.settings.semantic_cache_enabled", False)
    monkeypatch.setattr("shared.config.settings.cache_enabled", False)
    monkeypatch.setattr("shared.config.settings.audit_log_enabled", False)
    monkeypatch.setattr("shared.config.settings.analytics_log_enabled", False)
    chunk = RetrievedChunk(
        chunk_id="cert::0",
        document_id="cert",
        title="شهادات الادخار بالعملة المحلية",
        url="https://nbe.example/certificates",
        language="ar",
        text="شهادات الادخار بالعملة المحلية دون نسبة عائد رقمية.",
        score=0.9,
        doc_type="certificate_rate",
        category="certificates",
    )
    result = RetrievalResult(
        query="كم عائد شهادة سنة بالجنيه",
        language="ar",
        chunks=[chunk],
        confidence=0.9,
        should_answer=True,
        abstention_reason=None,
        intent="certificate_rate",
        category="certificates",
        intent_confidence=0.95,
        decision="ANSWER",
    )
    orchestrator = Orchestrator(search_service=_StaticSearch(result))

    response = asyncio.run(
        orchestrator.search("كم عائد شهادة سنة بالجنيه", "ar")
    )

    assert response.answered is False
    assert response.answer is None
    assert response.abstention_reason == "missing_applicable_rate"
    assert response.guidance
    assert "غير موثق" in response.guidance


def test_multi_intent_requires_all_subqueries_to_pass():
    merged = merge_retrieval_results([
        _result(should_answer=True),
        _result(should_answer=False, url="https://nbe.example/accounts"),
    ])

    assert merged.should_answer is False
