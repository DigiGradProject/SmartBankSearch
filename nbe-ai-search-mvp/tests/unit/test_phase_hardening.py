import time

from services.rag.language import detect_language
from services.rag.query_rewrite import rewrite_query
from services.rag.synonym_ontology import validate_vocabulary
from services.rag.audit import build_audit_event
from services.search_service.intent_classifier import classify_query
from shared.config import settings


def test_synonym_ontology_valid():
    errors = validate_vocabulary(settings.project_root)
    assert errors == []


def test_query_understanding_latency_smoke():
    start = time.perf_counter()
    for _ in range(20):
        lang = detect_language("عايز افتح حساب بنكي", "auto")
        intent = classify_query("عايز افتح حساب بنكي", lang)
        rewrite_query("عايز افتح حساب بنكي", lang, intent_expand=intent.expand_ar)
    elapsed_ms = (time.perf_counter() - start) * 1000 / 20
    assert elapsed_ms < 50  # generous smoke threshold on shared CI hosts


def test_audit_event_hashes_query():
    event = build_audit_event(
        query="فتح حساب",
        language="ar",
        intent="account_open",
        intent_confidence=0.95,
        chunk_ids=["a"],
        urls=["https://nbe"],
        confidence=0.8,
        answered=True,
        abstention_reason=None,
        model_embedding="bge-m3",
        model_reranker="reranker",
        model_llm="qwen3:8b",
    )
    assert event["query_hash"]
    assert event["intent"] == "account_open"
