from shared.content_hash import compute_content_hash


def test_content_hash_stable():
    first = compute_content_hash("sample text")
    second = compute_content_hash("sample text")
    assert first == second
    assert first != compute_content_hash("other text")
