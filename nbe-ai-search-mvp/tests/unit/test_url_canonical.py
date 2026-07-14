from shared.url_canonical import canonical_url_key


def test_canonical_url_key_preserves_category_id():
    a = canonical_url_key(
        'https://www.nbe.com.eg/NBE/E/#/AR/ProductCategory?inParams={"CategoryID":"LocalCertificatesID"}'
    )
    b = canonical_url_key(
        'https://www.nbe.com.eg/NBE/E/#/AR/ProductCategory?inParams={"CategoryID":"CertificatesID"}'
    )
    assert a != b
    assert "localcertificatesid" in a
