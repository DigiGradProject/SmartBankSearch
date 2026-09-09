from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rescrape_playwright import (  # noqa: E402
    PageResult,
    _documents_from_results,
    _strip_cross_page_boilerplate,
    extract_sitemap_urls,
    normalize_target_url,
)


def test_normalize_target_url_accepts_only_one_official_bilingual_url():
    valid = "https://www.nbe.com.eg/NBE/E/#/AR/Accounts"
    assert normalize_target_url(valid) == valid
    assert normalize_target_url(valid + valid) is None
    assert normalize_target_url("https://evil.example/NBE/E/#/AR/Accounts") is None
    assert normalize_target_url(
        "https://www.nbe.com.eg/NBE/E/#/EN/MultiArticle?inParams=%7B"
    ) is None


def test_extract_sitemap_urls_is_unique_balanced_and_filters_private_routes():
    markup = """
    <loc>https://www.nbe.com.eg/NBE/E/#/AR/Accounts</loc>
    <loc>https://www.nbe.com.eg/NBE/E/#/AR/Accounts</loc>
    <loc>https://www.nbe.com.eg/NBE/E/#/EN/Accounts</loc>
    <loc>https://www.nbe.com.eg/NBE/E/#/EN/CustomerLogin</loc>
    """
    urls = extract_sitemap_urls(markup, {"ar", "en"})
    assert urls == [
        "https://www.nbe.com.eg/NBE/E/#/ar/Accounts".replace("/ar/", "/AR/"),
        "https://www.nbe.com.eg/NBE/E/#/en/Accounts".replace("/en/", "/EN/"),
    ]


def test_cross_page_boilerplate_is_removed_without_dropping_unique_content():
    common = "National Bank of Egypt navigation line"
    results = [
        PageResult(
            url=f"https://www.nbe.com.eg/NBE/E/#/EN/Page{index}",
            language="en",
            title=f"Page {index}",
            content=f"{common}\nUnique banking content for page {index} " + ("x" * 150),
        )
        for index in range(10)
    ]
    removed = _strip_cross_page_boilerplate(results)
    assert removed["en"] == 10
    assert all(common not in result.content for result in results)
    assert all(result.error is None for result in results)


def test_exact_content_duplicates_are_flagged():
    content = "Certificate information " + ("details " * 30)
    results = [
        PageResult(
            url="https://www.nbe.com.eg/NBE/E/#/EN/A",
            language="en",
            title="A",
            content=content,
        ),
        PageResult(
            url="https://www.nbe.com.eg/NBE/E/#/EN/B",
            language="en",
            title="B",
            content=content,
        ),
    ]
    documents, duplicates = _documents_from_results(results)
    assert duplicates == 1
    assert documents[0]["metadata"]["duplicate_of"] is None
    assert documents[1]["metadata"]["duplicate_of"] == documents[0]["id"]
