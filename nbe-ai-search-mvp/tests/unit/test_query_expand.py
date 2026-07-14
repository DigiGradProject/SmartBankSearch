from services.search_service.query_expand import expand_query


def test_expand_buy_certificate_colloquial():
    expanded = expand_query("عايز اشتري شهاده", "ar")
    assert "شراء" in expanded
    assert "شهادة" in expanded or "شهادات" in expanded
