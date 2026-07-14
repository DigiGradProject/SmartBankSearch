from services.search_service.hybrid_retriever import HybridRetriever
from services.search_service.search import SearchService
from services.search_service.intent_classifier import QueryIntent


def test_confidence_gate_abstains_without_chunks(monkeypatch):
    class EmptyHybrid(HybridRetriever):
        def retrieve(self, query_text, language, *, candidate_k, intent, apply_filter, doc_types):
            return [], False

    service = SearchService(hybrid_retriever=EmptyHybrid())
    result = service.retrieve("what is the fee for transfers?", "en")
    assert result.should_answer is False
    assert result.confidence == 0.0
