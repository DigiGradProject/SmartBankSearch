import re
from urllib.parse import unquote

# CMS/test pages that are mostly navigation menus, not real content.
JUNK_DOC_PATTERN = re.compile(
    r"(?i)(^|_)(test|tst|testqc|testtt|newcontent|sadfasdf|researchtest|"
    r"minaaa|marwacontent|fredv|fghg|fffgfcg|ferfrb|cdvc|cdcv|df|oppo|ziad|"
    r"weklyreport2|rana|testone|tst|sadfasdf)",
)

MENU_LIST_PATTERN = re.compile(r" - .+ - .+ - ")

LOW_VALUE_DOC_PATTERN = re.compile(
    r"(?i)(SiteMap|ProductForm|ProductFormSubmit|NEWAPC|SwiperSideMenu|SubPageDynamicContent)",
)

# ProductDetails pages that exist in menus but render empty on the live NBE site.
BROKEN_CITATION_URL_PATTERN = re.compile(
    r"(?i)ProductDetails.*(Belady%20USD|Belady USD|beladyoneyear|beladythreeyears|beladyfiveyears)",
)

OFFICIAL_CATEGORY_IDS = (
    "CreditCardsID",
    "DepitCardsID",
    "PrepaidCardsID",
    "LocalCertificatesID",
    "ForigenCertificatesID",
    "CertificatesRatesForeignCurrency",
    "CardsID",
)


def is_official_product_category_url(url: str) -> bool:
    """NBE SPA category listings are menu-like but are valid citation/retrieval sources."""
    if "ProductCategory" not in url:
        return False
    return any(category_id in url for category_id in OFFICIAL_CATEGORY_IDS)


def is_junk_document(doc_id: str) -> bool:
    return bool(JUNK_DOC_PATTERN.search(doc_id))


def is_low_value_document(doc_id: str) -> bool:
    return bool(LOW_VALUE_DOC_PATTERN.search(doc_id))


def is_broken_citation_url(url: str) -> bool:
    return bool(BROKEN_CITATION_URL_PATTERN.search(url))


def is_menu_heavy_text(text: str) -> bool:
    """Detect sitemap-style / flattened navbar chunks that list many products."""
    if len(text) < 200:
        return False

    # Flattened scrape blobs from documents.json often start with chrome.
    chrome_markers = (
        "اهلا بك | الخروج",
        "اهلا بك |",
        "القوائم الرئيسيه",
        "القوائم الرئيسية",
        "Welcome |",
        "Main Menu",
    )
    if any(marker in text[:180] for marker in chrome_markers):
        return True

    dash_items = text.count(" - ")
    pipe_items = text.count(" | ")
    lines = [line for line in text.splitlines() if line.strip()]
    link_like = sum(1 for line in lines if len(line) < 80 and " - " in line)

    # Dense product-list soup without newlines (common in unclean exports).
    productish = sum(
        1
        for token in (
            "شهادات",
            "حسابات",
            "بطاقات",
            "قروض",
            "Certificates",
            "Accounts",
            "Cards",
            "Loans",
        )
        if token in text
    )
    if productish >= 4 and len(text) > 800 and text.count("\n") < 5:
        return True

    return dash_items >= 12 or pipe_items >= 8 or (len(lines) > 30 and link_like / max(len(lines), 1) > 0.6)


def strip_site_chrome(content: str) -> str:
    """Remove repeated navbar blocks that appear at the top of most scraped pages."""
    markers = [
        "حماية حقوق العملاء",
        "Consumer Protection",
        "شرائح العملاء",
        "Retail Segments",
        "القوائم الرئيسية",
        "القوائم الرئيسيه",
    ]
    for marker in markers:
        idx = content.find(marker)
        if 0 <= idx < len(content) * 0.7:
            content = content[idx + len(marker) :].strip(" |-")
            break
    return content


def title_from_document(doc_id: str, url: str) -> str:
    product_match = re.search(r"ProductID___(.+?)(?:_\d+_?)?\}", doc_id)
    if product_match:
        return _clean_title(product_match.group(1))

    category_match = re.search(r"CategoryID___(.+?)(?:_)?\}", doc_id)
    if category_match:
        return _clean_title(category_match.group(1))

    if "#/" in url:
        path = unquote(url.split("#/", 1)[1]).split("?")[0].strip("/")
        if path and path not in {"AR", "EN"}:
            segment = path.split("/")[-1]
            if segment and not segment.startswith("Product"):
                return _clean_title(segment)

    parts = doc_id.split("_", 1)
    raw = parts[1] if len(parts) > 1 else doc_id
    return _clean_title(raw)


def _clean_title(raw: str) -> str:
    cleaned = re.sub(r"inParams=\{.*", "", raw)
    cleaned = re.sub(r"[_\-]+", " ", cleaned).strip()
    return cleaned or raw
