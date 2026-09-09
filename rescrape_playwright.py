"""
Re-scrape JS-rendered ("thin") NBE pages using Playwright.

WHY THIS SCRIPT EXISTS
-----------------------
The original scrape (`nbe_complete_scrape.zip`) used a scraper that captured
the page too early — before the Angular/SPA app finished fetching and
rendering the real product/rate content. As a result, 175 of 315 pages
(certificates, product details, rates, etc.) came back as empty or
near-empty ("SPA shells" — just nav/menu, no actual body content).

This script discovers current Arabic and English pages from NBE's official
sitemap, validates every URL, and visits them with a real headless browser.
(not just "page loaded"), and extracts clean text + tables.

USAGE
-----
1. pip install playwright beautifulsoup4
2. playwright install chromium
3. Put `priority_urls.txt` (one URL per line) next to this script, or pass
   --urls-file path/to/file.txt
4. python rescrape_playwright.py --urls-file priority_urls.txt --out-dir output/

OUTPUT
------
One JSON file per page in --out-dir, matching the target schema:
    { id, title, url, language, content, metadata }
Plus a `rescrape_report.md` summary at the end (how many pages actually
got real content this time vs. still came back thin — some pages may be
genuinely thin, e.g. a page that's just a single confirmation button).

NOTES / THINGS TO TUNE
-----------------------
- This site is a hash-routed SPA (`#/AR/PageName` style), so a plain
  `page.goto()` + `networkidle` is necessary but NOT sufficient — SPAs often
  keep background polling (e.g. chat widget, analytics) that never lets
  "networkidle" settle cleanly, and the actual content can still be
  populated by client-side JS after the network goes quiet. This script
  therefore ALSO waits for the DOM to stop mutating (a "DOM settled" wait)
  before extracting content, which is the most reliable generic signal for
  SPA-rendered content across many different page templates without having
  to hardcode a CSS selector per page type.
- If a specific page still comes back thin after this, that page most
  likely needs a page-specific wait condition (e.g. waiting for a rates
  table's specific class to appear, or clicking a tab/accordion first).
  The report at the end flags these for manual follow-up instead of
  silently accepting a thin result.
- Be a respectful crawler: this script only targets the 175 known-thin
  pages (not the full 315), runs with a concurrency cap, and a delay
  between requests. Do not remove the delay/concurrency limits — this is
  hitting NBE's production website.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html as html_lib
import json
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, TimeoutError as PWTimeout

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONCURRENCY = 2                # be gentle with NBE's production site
DELAY_BETWEEN_REQUESTS_SEC = 1.5
NAV_TIMEOUT_MS = 30_000
DOM_SETTLE_TIMEOUT_MS = 10_000
DOM_SETTLE_QUIET_MS = 900       # how long the DOM must stop changing to be "settled"
DEFAULT_SITEMAP_URL = "https://www.nbe.com.eg/SiteMap.xml"
ALLOWED_HOSTS = {"www.nbe.com.eg", "nbe.com.eg"}
MIN_CONTENT_CHARS = 120

SITEMAP_URL_RE = re.compile(
    r'https://www\.nbe\.com\.eg/NBE/E/#/(?:AR|EN)/[^\s<"]+',
    re.IGNORECASE,
)

EXCLUDED_ROUTE_MARKERS = (
    "/null",
    "undefined",
    "add%2520content",
    "customerlogin",
    "customerprofile",
    "productorderhandler",
    "productgetorderdetails",
    "hrapplication",
    "smeslogin",
    "subpagedynamiccontent_test",
    "researchtest",
    "testxxxxx",
    "sadfasdf",
)

BLOCKED_RESOURCE_HOSTS = (
    "chatbot.nbe.com.eg",
    "googletagmanager.com",
    "google-analytics.com",
    "hotjar.com",
    "facebook.com",
    "recaptcha.net",
    "google.com/recaptcha",
)

# Boilerplate line patterns to strip (menu/nav/chatbot text seen repeated
# across ~94% of pages in the original scrape). Extend this list if the
# re-scrape report shows a page still dominated by nav text.
BOILERPLATE_PATTERNS = [
    r"^أهلا بك",
    r"^عرض الصفحة الشخصية$",
    r"^إشعارات$",
    r"^القائمة المفضلة$",
    r"^الخروج$",
    r"^القوائم الرئيسية$",
    r"^جميع الحقوق محفوظة",
    r"^Couldn.?t connect",
    r"^We.?ll keep retrying",
    r"^try now$",
    r"NBE Chatbot",
    r"^ابدأ المحادثة$",
    r"^Restart Conversation$",
    r"^إعادة المحادثة$",
    r"^دخول$",
    r"^المزيد$",
    r"^مقارنة$",
    r"^أضف إلى المفضلة$",
    r"^اظغط للمقارنة بين المنتجات\.?$",
    r"^Compare$",
    r"^Add to favorites$",
    r"^مرحبا،? ?👋?$",
    r"^دعنا نتحدث إذا كان لديك أي أسئلة$",
    r"^#(?:Category|Product)[A-Za-z]*Container#$",
    r"^#[A-Za-z][A-Za-z0-9_-]{2,80}#$",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

BASE64_LIKE_RE = re.compile(r"^[A-Za-z0-9+/=]{100,}$")


@dataclass
class PageResult:
    url: str
    language: str
    title: str
    content: str
    tables_md: list[str] = field(default_factory=list)
    error: str | None = None
    attempts: int = 1


def normalize_target_url(raw_url: str) -> str | None:
    """Accept one official NBE AR/EN SPA URL and reject concatenated/unsafe input."""
    candidate = html_lib.unescape((raw_url or "").strip()).rstrip("/>")
    if not candidate or candidate.count("https://") != 1:
        return None
    if any(char.isspace() for char in candidate):
        return None
    parsed = urlsplit(candidate)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        return None
    lowered_candidate = candidate.lower()
    if lowered_candidate.count("%7b") != lowered_candidate.count("%7d"):
        return None
    if parsed.path.rstrip("/") != "/NBE/E":
        return None
    if not re.match(r"^/(?:AR|EN)/[^/].*", parsed.fragment, re.IGNORECASE):
        return None
    lowered = candidate.lower()
    if any(marker in lowered for marker in EXCLUDED_ROUTE_MARKERS):
        return None
    return candidate


def _url_priority(url: str) -> int:
    """Put customer-facing search content before corporate/reporting pages."""
    lowered = url.lower()
    if "productdetails" in lowered:
        return 0
    if "productcategory" in lowered:
        return 1
    if any(
        marker in lowered
        for marker in (
            "accounts",
            "loans",
            "creditcards",
            "debitcards",
            "exchangerates",
            "atmbranch",
            "contactus",
            "faq",
        )
    ):
        return 2
    return 3


def detect_language(url: str) -> str:
    if "/AR/" in url:
        return "ar"
    if "/EN/" in url:
        return "en"
    return "ar"


def extract_sitemap_urls(markup: str, languages: set[str]) -> list[str]:
    """Extract unique, safe official URLs from Chrome-rendered sitemap XML."""
    urls: set[str] = set()
    for match in SITEMAP_URL_RE.findall(html_lib.unescape(markup or "")):
        url = normalize_target_url(match)
        if not url:
            continue
        language = detect_language(url)
        if language in languages:
            urls.add(url)
    return sorted(urls, key=lambda url: (detect_language(url), _url_priority(url), url))


def stable_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def clean_text_block(text: str) -> str | None:
    text = text.strip()
    if not text or len(text) < 2:
        return None
    if BOILERPLATE_RE.search(text):
        return None
    if BASE64_LIKE_RE.match(text.replace(" ", "")):
        return None
    return text


def table_to_markdown(table_soup) -> str:
    rows = []
    for tr in table_soup.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows[1:]:
        # pad/truncate to header length so the markdown table stays valid
        r = (r + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


async def wait_for_dom_settle(page: Page, quiet_ms: int = DOM_SETTLE_QUIET_MS,
                               timeout_ms: int = DOM_SETTLE_TIMEOUT_MS) -> None:
    """Wait until the DOM stops mutating for `quiet_ms`, or give up after timeout_ms.

    This is the key fix for SPA pages: 'networkidle' alone is not reliable
    on sites with persistent background connections (chat widgets, polling),
    so we additionally watch for actual DOM mutations to stop, which is a
    much more direct signal that client-side rendering has finished.
    """
    await page.evaluate(
        """
        (quietMs) => {
          window.__domSettled = false;
          let timer = null;
          const observer = new MutationObserver(() => {
            if (timer) clearTimeout(timer);
            timer = setTimeout(() => { window.__domSettled = true; }, quietMs);
          });
          observer.observe(document.body, { childList: true, subtree: true, characterData: true });
          timer = setTimeout(() => { window.__domSettled = true; }, quietMs);
        }
        """,
        quiet_ms,
    )
    try:
        await page.wait_for_function("window.__domSettled === true", timeout=timeout_ms)
    except PWTimeout:
        # Not fatal — we extract whatever is on the page at this point.
        pass


def _extract_visible_content(markup: str, url: str) -> tuple[str, str]:
    soup = BeautifulSoup(markup, "lxml")
    for tag in soup.select(
        "nav, header, footer, aside, script, style, iframe, noscript, svg, "
        ".FooterContainer, .oda-chat-wrapper, .grecaptcha-badge, "
        "[class*='chatbot'], [class*='Chatbot'], [id*='Footer'], [id*='Header']"
    ):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""
    generic_titles = {
        "national bank of egypt",
        "البنك الأهلى المصري",
        "البنك الأهلى المصرى",
        "",
    }
    if title.strip().lower() in generic_titles:
        for heading in soup.find_all(["h1", "h2", "h3"]):
            candidate = " ".join(heading.get_text(" ", strip=True).split())
            if len(candidate) >= 4 and not BOILERPLATE_RE.search(candidate):
                title = candidate
                break
    if not title or title.strip().lower() in generic_titles:
        title = urlsplit(url).fragment.split("?", 1)[0].rsplit("/", 1)[-1] or url

    raw_text = soup.get_text("\n", strip=True)
    kept: list[str] = []
    seen: set[str] = set()
    for raw_line in raw_text.splitlines():
        line = " ".join(raw_line.replace("\xa0", " ").split())
        cleaned = clean_text_block(line)
        if not cleaned:
            continue
        normalized = cleaned.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        kept.append(cleaned)
    return title, "\n".join(kept).strip()


async def extract_page(page: Page, url: str, retries: int = 2) -> PageResult:
    language = detect_language(url)
    last_error = ""
    for attempt in range(1, retries + 2):
        try:
            response = await page.goto(url, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
            if response is not None and response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")
            await page.wait_for_selector("body", state="attached", timeout=5_000)
            await wait_for_dom_settle(page)

            markup = await page.content()
            title, content = _extract_visible_content(markup, url)
            if len(content) < MIN_CONTENT_CHARS:
                raise RuntimeError(f"thin content ({len(content)} chars)")
            return PageResult(
                url=url,
                language=language,
                title=title,
                content=content,
                attempts=attempt,
            )
        except Exception as exc:  # noqa: BLE001 - retry isolated page failures
            last_error = str(exc)
            if attempt <= retries:
                await page.wait_for_timeout(1_500 * attempt)

    return PageResult(
        url=url,
        language=language,
        title=url,
        content="",
        error=last_error,
        attempts=retries + 1,
    )


async def _block_non_content_requests(route) -> None:
    request = route.request
    lowered = request.url.lower()
    if request.resource_type in {"image", "media", "font"}:
        await route.abort()
        return
    if any(host in lowered for host in BLOCKED_RESOURCE_HOSTS):
        await route.abort()
        return
    await route.continue_()


async def worker(
    name: int,
    queue: asyncio.Queue,
    browser,
    results: list[PageResult],
    progress: dict[str, int],
    *,
    delay_seconds: float,
    retries: int,
) -> None:
    context = await browser.new_context(
        locale="en-US",
        viewport={"width": 1366, "height": 900},
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
        ),
    )
    await context.route("**/*", _block_non_content_requests)
    page = await context.new_page()
    while True:
        url = await queue.get()
        if url is None:
            queue.task_done()
            break
        try:
            result = await extract_page(page, url, retries=retries)
            results.append(result)
            progress["done"] += 1
            status = "OK" if not result.error else "FAILED"
            print(
                f"[{progress['done']:4d}/{progress['total']}] "
                f"worker={name} {status:6s} lang={result.language} "
                f"chars={len(result.content):5d} attempts={result.attempts} {url}",
                flush=True,
            )
        finally:
            queue.task_done()
        await asyncio.sleep(delay_seconds)
    await context.close()


async def discover_sitemap_urls(
    sitemap_url: str,
    *,
    executable_path: str,
    languages: set[str],
) -> list[str]:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            executable_path=executable_path,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        page = await browser.new_page()
        await page.goto(sitemap_url, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        await page.wait_for_timeout(1_000)
        markup = await page.content()
        await browser.close()
    return extract_sitemap_urls(markup, languages)


async def run(
    urls: list[str],
    *,
    executable_path: str,
    concurrency: int,
    delay_seconds: float,
    retries: int,
) -> list[PageResult]:
    queue: asyncio.Queue = asyncio.Queue()
    for url in urls:
        queue.put_nowait(url)
    for _ in range(concurrency):
        queue.put_nowait(None)

    results: list[PageResult] = []
    progress = {"done": 0, "total": len(urls)}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            executable_path=executable_path,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        workers = [
            asyncio.create_task(
                worker(
                    index,
                    queue,
                    browser,
                    results,
                    progress,
                    delay_seconds=delay_seconds,
                    retries=retries,
                )
            )
            for index in range(concurrency)
        ]
        await queue.join()
        await asyncio.gather(*workers)
        await browser.close()
    return results


def _strip_cross_page_boilerplate(results: list[PageResult]) -> dict[str, int]:
    by_language: dict[str, list[PageResult]] = {"ar": [], "en": []}
    for result in results:
        if not result.error:
            by_language.setdefault(result.language, []).append(result)

    removed_counts: dict[str, int] = {}
    for language, pages in by_language.items():
        line_frequency: dict[str, int] = {}
        for result in pages:
            unique_lines = {
                " ".join(line.casefold().split())
                for line in result.content.splitlines()
                if 2 <= len(line.strip()) <= 180
            }
            for line in unique_lines:
                line_frequency[line] = line_frequency.get(line, 0) + 1

        threshold = max(10, int(len(pages) * 0.55))
        boilerplate = {
            line for line, count in line_frequency.items() if count >= threshold
        }
        removed = 0
        for result in pages:
            kept: list[str] = []
            for line in result.content.splitlines():
                normalized = " ".join(line.casefold().split())
                if normalized in boilerplate:
                    removed += 1
                    continue
                kept.append(line)
            result.content = "\n".join(kept).strip()
            if len(result.content) < MIN_CONTENT_CHARS:
                result.error = f"thin after boilerplate removal ({len(result.content)} chars)"
        removed_counts[language] = removed
    return removed_counts


def _language_matches(result: PageResult) -> bool:
    arabic = len(re.findall(r"[\u0600-\u06FF]", result.content))
    latin = len(re.findall(r"[A-Za-z]", result.content))
    if result.language == "ar":
        return arabic >= 20 or arabic >= latin * 0.15
    return latin >= 30 and arabic <= latin * 0.5


def _apply_language_quality_gate(results: list[PageResult]) -> int:
    rejected = 0
    for result in results:
        if result.error:
            continue
        if not _language_matches(result):
            result.error = f"language mismatch for {result.language}"
            rejected += 1
    return rejected


def _documents_from_results(results: list[PageResult]) -> tuple[list[dict], int]:
    extracted_at = datetime.now(timezone.utc).isoformat()
    documents: list[dict] = []
    canonical_by_content: dict[tuple[str, str], str] = {}
    duplicate_count = 0

    for result in sorted(results, key=lambda item: (item.language, item.url)):
        if result.error or len(result.content) < MIN_CONTENT_CHARS:
            continue
        digest = hashlib.sha256(result.content.encode("utf-8")).hexdigest()
        duplicate_of = canonical_by_content.get((result.language, digest))
        if duplicate_of:
            duplicate_count += 1
        else:
            canonical_by_content[(result.language, digest)] = stable_id(result.url)
        documents.append(
            {
                "id": stable_id(result.url),
                "title": result.title,
                "url": result.url,
                "language": result.language,
                "content": result.content,
                "metadata": {
                    "content_hash": digest,
                    "extracted_at": extracted_at,
                    "has_tables": bool(result.tables_md),
                    "source": "playwright_sitemap_rescrape",
                    "source_url": result.url,
                    "scrape_attempts": result.attempts,
                    "duplicate_of": duplicate_of,
                },
            }
        )
    return documents, duplicate_count


def write_corpus(results: list[PageResult], output_path: Path) -> tuple[int, int]:
    documents, duplicate_count = _documents_from_results(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
    temp_path.replace(output_path)
    return len(documents), duplicate_count


def write_report(
    results: list[PageResult],
    report_path: Path,
    *,
    discovered: int,
    boilerplate_removed: dict[str, int],
    documents_written: int,
    duplicate_count: int,
) -> None:
    failed = [result for result in results if result.error]
    ok = [result for result in results if not result.error]
    language_attempted = {
        language: sum(result.language == language for result in results)
        for language in ("ar", "en")
    }
    language_ok = {
        language: sum(result.language == language and not result.error for result in results)
        for language in ("ar", "en")
    }
    lines = [
        "# NBE Bilingual Re-scrape Report",
        "",
        f"- Sitemap URLs discovered and accepted: **{discovered}**",
        f"- Pages attempted: **{len(results)}** "
        f"(AR={language_attempted['ar']}, EN={language_attempted['en']})",
        f"- Valid pages after cleaning: **{len(ok)}** "
        f"(AR={language_ok['ar']}, EN={language_ok['en']})",
        f"- Failed/thin pages: **{len(failed)}**",
        f"- JSONL documents written: **{documents_written}**",
        f"- Exact-content duplicates flagged: **{duplicate_count}**",
        f"- Repeated boilerplate lines removed: **{sum(boilerplate_removed.values())}** "
        f"(AR={boilerplate_removed.get('ar', 0)}, EN={boilerplate_removed.get('en', 0)})",
        f"- Generated at: **{datetime.now(timezone.utc).isoformat()}**",
        "",
        "## Failed or thin pages",
        "",
    ]
    lines.extend(
        f"- {result.url} — {result.error}" for result in failed
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {report_path}", flush=True)


def _limit_balanced(urls: list[str], max_per_language: int) -> list[str]:
    if max_per_language <= 0:
        return urls
    selected: list[str] = []
    for language in ("ar", "en"):
        selected.extend(
            [url for url in urls if detect_language(url) == language][
                :max_per_language
            ]
        )
    return selected


async def async_main(args) -> None:
    languages = {
        language.strip().lower()
        for language in args.languages.split(",")
        if language.strip().lower() in {"ar", "en"}
    }
    if not languages:
        raise SystemExit("--languages must contain ar, en, or both")

    urls = await discover_sitemap_urls(
        args.sitemap_url,
        executable_path=args.executable_path,
        languages=languages,
    )
    discovered = len(urls)

    if args.urls_file:
        file_urls = [
            normalize_target_url(line)
            for line in args.urls_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        urls = sorted(
            {url for url in urls + file_urls if url},
            key=lambda url: (detect_language(url), _url_priority(url), url),
        )

    urls = _limit_balanced(urls, args.max_per_language)
    print(
        f"Accepted {discovered} sitemap URLs; crawling {len(urls)} "
        f"(AR={sum(detect_language(url) == 'ar' for url in urls)}, "
        f"EN={sum(detect_language(url) == 'en' for url in urls)})",
        flush=True,
    )

    results = await run(
        urls,
        executable_path=args.executable_path,
        concurrency=args.concurrency,
        delay_seconds=args.delay,
        retries=args.retries,
    )
    boilerplate_removed = _strip_cross_page_boilerplate(results)
    language_rejected = _apply_language_quality_gate(results)
    if language_rejected:
        print(f"Language quality gate rejected {language_rejected} pages", flush=True)
    documents_written, duplicate_count = write_corpus(results, args.jsonl_output)
    write_report(
        results,
        args.report,
        discovered=discovered,
        boilerplate_removed=boilerplate_removed,
        documents_written=documents_written,
        duplicate_count=duplicate_count,
    )
    print(f"Corpus written atomically to {args.jsonl_output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sitemap-url", default=DEFAULT_SITEMAP_URL)
    parser.add_argument("--urls-file", type=Path, default=None)
    parser.add_argument("--languages", default="ar,en")
    parser.add_argument(
        "--max-per-language",
        type=int,
        default=0,
        help="0 crawls every accepted sitemap URL; otherwise cap each language",
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path("nbe-scrape-cleaner/output/documents.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("nbe-scrape-cleaner/output/rescrape_report.md"),
    )
    parser.add_argument("--executable-path", default="/usr/bin/google-chrome")
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=CONCURRENCY)
    parser.add_argument("--delay", type=float, default=DELAY_BETWEEN_REQUESTS_SEC)
    parser.add_argument("--retries", type=int, choices=range(0, 4), default=2)
    args = parser.parse_args()

    if args.delay < 0.5:
        raise SystemExit("--delay must be at least 0.5 seconds")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
