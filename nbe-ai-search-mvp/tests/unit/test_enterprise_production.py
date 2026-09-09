"""Unit tests for enterprise production hardening layers."""

from __future__ import annotations

from pathlib import Path

from ingestion.embedding.vector_store import RetrievedChunk
from services.context_builder.builder import ContextBuilder
from services.context_builder.compressor import compress_chunks
from services.feedback.service import FeedbackService
from services.rag.confidence import compute_confidence
from services.rag.explainability import build_explain_payload
from services.rag.query_planner import plan_query
from services.rag.query_understanding import understand_query
from services.rag.self_eval import FaithfulnessLabel, parse_faithfulness_label
from services.rag.semantic_cache import SemanticCache
from services.search_service.intent_classifier import QueryIntent
from shared.schemas import FeedbackRequest


def _chunk(cid: str, url: str, text: str, score: float, category: str = "certificates") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=f"d-{cid}",
        title=f"Title {cid}",
        url=url,
        language="ar",
        text=text,
        score=score,
        doc_type="certificate",
        category=category,
    )


def test_query_understanding_keeps_original():
    original = "عايز اعرف سعر الصرف"
    qu = understand_query(original, "ar")
    assert qu.original_query == original
    assert qu.search_query != ""
    assert qu.language == "ar"
    assert qu.intent.intent == "exchange_rate"
    assert isinstance(qu.entity_dicts(), list)


def test_entity_types_extracted():
    qu = understand_query("قرض شخصي وبطاقة ائتمان في القاهرة", "ar")
    types = {e.type for e in qu.entities}
    assert "loan" in types or any("قرض" in e.value for e in qu.entities)
    assert "card" in types or any("بطاق" in e.value for e in qu.entities)
    assert "location" in types or any("القاهرة" in e.value for e in qu.entities)


def test_context_compression_dedupes_and_strips():
    chunks = [
        _chunk("1", "https://nbe/a", "عائد الشهادة السنوي ١٨٪ للمستثمرين.\n\nتسجيل الدخول للقائمة", 0.9),
        _chunk("2", "https://nbe/a", "عائد الشهادة السنوي ١٨٪ للمستثمرين.", 0.85),
        _chunk("3", "https://nbe/b", "شروط الاسترداد بعد ٦ أشهر من الشراء.", 0.8),
    ]
    compressed = compress_chunks("عائد شهادة", chunks)
    assert len(compressed) <= 2
    assert all("تسجيل الدخول" not in (c.text or "") for c in compressed)


def test_calibrated_confidence_has_reason():
    chunks = [
        _chunk("1", "https://nbe/c1", "نص", 0.92, "certificates"),
        _chunk("2", "https://nbe/c2", "نص", 0.90, "certificates"),
        _chunk("3", "https://nbe/c3", "نص", 0.88, "certificates"),
    ]
    intent = QueryIntent(
        intent="certificate_rate",
        category="certificates",
        confidence=0.95,
        allowed_doc_types=("certificate_rate", "certificate"),
    )
    breakdown = compute_confidence(chunks, intent=intent)
    assert 0.0 < breakdown.final <= 1.0
    assert "Certificates" in breakdown.reason or "certificate" in breakdown.reason.lower()
    assert breakdown.category_consistency > 0.5


def test_citation_ranking_includes_scores():
    chunks = [
        _chunk("1", "https://nbe/c1", "عائد الشهادة التفصيلي للمستثمرين المحليين ١٨٪", 0.95),
        _chunk("2", "https://nbe/c2", "عائد الشهادة التفصيلي للمستثمرين المحليين ١٧٪", 0.9),
    ]
    built = ContextBuilder().build("عائد شهادة", chunks, compress=False)
    assert built.citations
    assert built.citations[0].relevance_score is not None
    assert built.citations[0].category == "certificates"


def test_planner_splits_multi_intent():
    plan = plan_query("أريد فتح حساب وطلب بطاقة ائتمان", "ar")
    assert plan.is_multi is True
    assert len(plan.subqueries) == 2
    intents = {s.intent_name for s in plan.subqueries}
    assert "account_open" in intents or "credit_card" in intents


def test_planner_single_intent_unchanged():
    plan = plan_query("سعر الصرف", "ar")
    assert plan.is_multi is False
    assert len(plan.subqueries) == 1


def test_planner_does_not_promote_weak_fragment_for_investment_query():
    plan = plan_query("انا معايا فلوس وعايز استثمر بيهم", "ar")
    assert plan.is_multi is False
    assert [sub.intent_name for sub in plan.subqueries] == ["certificate_buy"]


def test_self_eval_label_parse():
    assert parse_faithfulness_label("The answer is SUPPORTED.") == FaithfulnessLabel.SUPPORTED
    assert parse_faithfulness_label("UNSUPPORTED material") == FaithfulnessLabel.UNSUPPORTED
    assert parse_faithfulness_label("maybe") == FaithfulnessLabel.PARTIALLY_SUPPORTED


def test_explainability_payload():
    qu = understand_query("سعر الصرف", "ar")
    chunks = [_chunk("1", "https://nbe/fx", "exchange", 0.9, "exchange_rates")]
    payload = build_explain_payload(
        understanding=qu,
        chunks=chunks,
        confidence=0.9,
        confidence_reason="ok",
        decision="ANSWER",
    )
    assert "detected_intent" in payload
    assert "retrieved_documents" in payload
    assert payload["rewritten_query"]


def test_feedback_store(tmp_path: Path):
    path = tmp_path / "feedback.jsonl"
    service = FeedbackService(path=path)
    result = service.submit(
        FeedbackRequest(query_hash="abc123", vote="helpful", question="q", answer="a")
    )
    assert result.feedback_id.startswith("fb_")
    assert path.exists()
    assert "helpful" in path.read_text(encoding="utf-8")


def test_semantic_cache_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("shared.config.settings.semantic_cache_enabled", True)
    monkeypatch.setattr("shared.config.settings.chroma_path", tmp_path / "chroma")
    monkeypatch.setattr("shared.config.settings.semantic_cache_threshold", 0.90)
    cache = SemanticCache(collection_name="test_semantic_cache_unit")
    payload = {
        "answer": "إجابة",
        "confidence": 0.9,
        "citations": [{"title": "t", "url": "u"}],
        "answered": True,
        "language": "ar",
    }
    cache_id = cache.store("سعر الصرف اليوم", "ar", payload, retrieved_urls=["u"])
    assert cache_id
    hit = cache.lookup("سعر الصرف اليوم", "ar")
    assert hit is not None
    assert hit.payload["answer"] == "إجابة"
