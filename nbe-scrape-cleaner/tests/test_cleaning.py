from cleaner.ids import document_id
from cleaner.noise import is_base64ish_token, strip_base64_noise
from cleaner.boilerplate import build_boilerplate_set, filter_content_blocks
from cleaner.tables import rows_to_markdown


def test_id_determinism():
    url = "https://www.nbe.com.eg/NBE/E/#/AR/AccountsFAQs"
    assert document_id(url) == document_id(url)
    assert len(document_id(url)) == 16
    assert document_id(url) != document_id(url + "x")


def test_base64_stripping():
    blob = "iVBORw0KGgo" + ("A" * 120) + "AAAAAElFTkSuQmCC"
    text = f"مقدمة {blob} خاتمة"
    cleaned = strip_base64_noise(text)
    assert "iVBORw0KGgo" not in cleaned
    assert "مقدمة" in cleaned
    assert "خاتمة" in cleaned
    assert is_base64ish_token("A" * 50) is False
    assert is_base64ish_token("ABCDEFGHIJKLMNOP" * 10) is True


def test_table_to_markdown():
    rows = [
        ["\n\t\tQuestion\n", "\n\t\tAnswer\n"],
        ["ما هى الحسابات؟", "ودائع تحت الطلب"],
    ]
    md = rows_to_markdown(rows)
    assert "| Question | Answer |" in md
    assert "| --- | --- |" in md
    assert "ما هى الحسابات؟" in md
    assert "\t" not in md


def test_boilerplate_stripping_synthetic():
    pages = [
        ["Home", "Accounts", "UNIQUE PAGE A BODY"],
        ["Home", "Accounts", "UNIQUE PAGE B BODY"],
        ["Home", "Accounts", "UNIQUE PAGE C BODY"],
        ["Home", "Accounts", "UNIQUE PAGE D BODY"],
        ["Home", "Accounts", "UNIQUE PAGE E BODY"],
    ]
    # With 5 pages and fraction 0.55 => threshold max(40, 2)=40 by default absolute.
    # Use low absolute for synthetic unit test via direct call with min_absolute=3.
    boilerplate = build_boilerplate_set(pages, min_page_fraction=0.5, min_absolute=3)
    assert "home" in boilerplate
    assert "accounts" in boilerplate
    assert "unique page a body" not in boilerplate

    blocks = [
        {"tag": "a", "text": "Home"},
        {"tag": "p", "text": "UNIQUE PAGE A BODY"},
        {"tag": "script", "text": "var x=1"},
    ]
    kept = filter_content_blocks(blocks, boilerplate)
    assert len(kept) == 1
    assert kept[0]["text"] == "UNIQUE PAGE A BODY"
