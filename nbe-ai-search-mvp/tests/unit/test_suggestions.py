from services.search_service.suggestions import build_suggestions


def test_build_suggestions_for_certificate_query():
    class Chunk:
        def __init__(self, title, url, text, score):
            self.chunk_id = "1"
            self.document_id = "doc"
            self.title = title
            self.url = url
            self.language = "ar"
            self.text = text
            self.score = score

    chunks = [
        Chunk("شهادات الادخار", "https://example.com/cert", "شراء شهادة شهادات بلادي", 0.4),
        Chunk("Exchange Rates", "https://example.com/rates", "exchange rates", 0.2),
    ]
    suggestions = build_suggestions("عايز اشتري شهاده", "ar", chunks)
    assert len(suggestions) >= 1
    assert any("شهاد" in s.query or "شهاد" in s.label for s in suggestions)
