from ingestion.embedding.vector_store import VectorStore


class _Collection:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def get(self, *, include):
        assert include == []
        return {"ids": ["doc-a::0", "doc-a::1", "stale::0"]}

    def delete(self, *, ids):
        self.deleted.extend(ids)


def test_delete_chunks_not_in_removes_only_orphans():
    store = VectorStore.__new__(VectorStore)
    store._collection = _Collection()

    deleted = store.delete_chunks_not_in({"doc-a::0", "doc-a::1"})

    assert deleted == 1
    assert store._collection.deleted == ["stale::0"]
