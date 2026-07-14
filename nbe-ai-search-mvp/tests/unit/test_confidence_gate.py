from services.search_service.search import SearchService


def test_confidence_gate_abstains_without_chunks(monkeypatch):
    class EmptyStore:
        def query(self, query_text: str, top_k: int):
            return []

    service = SearchService(vector_store=EmptyStore())  # type: ignore[arg-type]
    result = service.retrieve("what is the fee for transfers?", "en")
    assert result.should_answer is False
    assert result.confidence == 0.0
