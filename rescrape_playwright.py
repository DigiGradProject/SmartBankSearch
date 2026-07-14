"""
Re-scrape JS-rendered ("thin") NBE pages using Playwright.

WHY THIS SCRIPT EXISTS
-----------------------
The original scrape (`nbe_complete_scrape.zip`) used a scraper that captured
the page too early — before the Angular/SPA app finished fetching and
rendering the real product/rate content. As a result, 175 of 315 pages
(certificates, product details, rates, etc.) came back as empty or
near-empty ("SPA shells" — just nav/menu, no actual body content).

This script re-visits ONLY those 175 known-thin pages with a real headless
browser, waits for the JS-rendered content to actually settle on the page
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
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, TimeoutError as PWTimeout

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONCURRENCY = 2                # be gentle with NBE's production site
DELAY_BETWEEN_REQUESTS_SEC = 2.0
NAV_TIMEOUT_MS = 30_000
DOM_SETTLE_TIMEOUT_MS = 15_000
DOM_SETTLE_QUIET_MS = 1200      # how long the DOM must stop changing to be "settled"

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


def detect_language(url: str) -> str:
    if "/AR/" in url:
        return "ar"
    if "/EN/" in url:
        return "en"
    return "ar"


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


async def extract_page(page: Page, url: str) -> PageResult:
    language = detect_language(url)
    try:
        await page.goto(url, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        try:
            await page.wait_for_load_state("networkidle", timeout=8000)
        except PWTimeout:
            pass  # SPA background polling may prevent true networkidle — that's OK
        await wait_for_dom_settle(page)

        html = await page.content()
        soup = BeautifulSoup(html, "lxml")

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        # Prefer a main/content-ish container if one clearly stands out;
        # otherwise fall back to body. This site's templates vary, so we
        # don't hardcode a single selector — we just exclude obvious chrome.
        for tag in soup.select("nav, header, footer, script, style, iframe"):
            tag.decompose()

        text_blocks = []
        for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "span", "div", "td", "th"]):
            # only take elements with direct text (avoid re-capturing children's text repeatedly)
            direct_text = el.find(text=True, recursive=False)
            if direct_text:
                cleaned = clean_text_block(str(direct_text))
                if cleaned:
                    text_blocks.append(cleaned)

        # de-duplicate while preserving order
        seen = set()
        deduped = []
        for t in text_blocks:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        tables_md = [table_to_markdown(t) for t in soup.find_all("table")]
        tables_md = [t for t in tables_md if t]

        content = "\n\n".join(deduped)
        if tables_md:
            content += "\n\n" + "\n\n".join(tables_md)

        return PageResult(url=url, language=language, title=title, content=content, tables_md=tables_md)

    except Exception as exc:  # noqa: BLE001 - we want to log and continue, never crash the whole run
        return PageResult(url=url, language=language, title=url, content="", error=str(exc))


async def worker(name: int, queue: asyncio.Queue, browser, out_dir: Path, results: list[PageResult]):
    context = await browser.new_context(locale="ar-EG")
    page = await context.new_page()
    while True:
        url = await queue.get()
        if url is None:
            queue.task_done()
            break
        result = await extract_page(page, url)
        results.append(result)

        doc = {
            "id": stable_id(result.url),
            "title": result.title,
            "url": result.url,
            "language": result.language,
            "content": result.content,
            "metadata": {
                "content_hash": hashlib.sha256(result.content.encode("utf-8")).hexdigest(),
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "has_tables": bool(result.tables_md),
                "source": "playwright_rescrape",
                "error": result.error,
            },
        }
        out_path = out_dir / f"{doc['id']}.json"
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

        status = "OK" if not result.error and len(result.content.strip()) >= 120 else "STILL THIN"
        print(f"[worker {name}] {status:10s} ({len(result.content):5d} chars)  {url}")

        queue.task_done()
        await asyncio.sleep(DELAY_BETWEEN_REQUESTS_SEC)
    await context.close()


async def run(urls: list[str], out_dir: Path) -> list[PageResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue = asyncio.Queue()
    for u in urls:
        queue.put_nowait(u)
    for _ in range(CONCURRENCY):
        queue.put_nowait(None)  # sentinel per worker

    results: list[PageResult] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        workers = [
            asyncio.create_task(worker(i, queue, browser, out_dir, results))
            for i in range(CONCURRENCY)
        ]
        await queue.join()
        for w in workers:
            w.cancel()
        await browser.close()
    return results


def write_report(results: list[PageResult], out_dir: Path) -> None:
    still_thin = [r for r in results if len(r.content.strip()) < 120 and not r.error]
    failed = [r for r in results if r.error]
    ok = [r for r in results if r not in still_thin and r not in failed]

    lines = [
        "# Re-scrape Report (Playwright)",
        "",
        f"- Total pages attempted: **{len(results)}**",
        f"- Now have real content (>= 120 chars): **{len(ok)}**",
        f"- Still thin after JS wait (needs manual/page-specific fix): **{len(still_thin)}**",
        f"- Failed to load: **{len(failed)}**",
        "",
        "## Still thin — needs manual follow-up",
        "",
    ]
    for r in still_thin:
        lines.append(f"- {r.url} ({len(r.content)} chars)")

    if failed:
        lines += ["", "## Failed", ""]
        for r in failed:
            lines.append(f"- {r.url} — {r.error}")

    (out_dir / "rescrape_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {out_dir / 'rescrape_report.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls-file", type=Path, default=Path("priority_urls.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    urls = [line.strip() for line in args.urls_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(urls)} URLs to re-scrape from {args.urls_file}")

    results = asyncio.run(run(urls, args.out_dir))
    write_report(results, args.out_dir)


if __name__ == "__main__":
    main()
