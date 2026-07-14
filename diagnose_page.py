"""
Diagnostic script: find out WHY the real content isn't loading.

Run this against ONE known-broken page (e.g. the certificate rates page)
and read the console output carefully. It logs every network request the
SPA makes while the page loads, with status codes — this will show us
whether:
  (a) the data-fetch API call never happens at all (missing referer/session/
      navigation-state issue), or
  (b) it happens but gets blocked (401/403/429 = bot detection / rate
      limiting / missing auth), or
  (c) it succeeds (200) but the response body itself contains an error
      payload (a backend-side issue, not a scraping-detection issue).

Each of these needs a different fix, so don't skip this step.

USAGE
-----
python diagnose_page.py "https://www.nbe.com.eg/NBE/E/#/AR/CertificatesRatesForeignCurrency"
"""

import asyncio
import sys
from playwright.async_api import async_playwright


async def diagnose(url: str):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)

        # Use a realistic context: real Chrome UA, a normal viewport, and
        # (importantly) visit the homepage FIRST so the SPA establishes
        # whatever session/router state it expects before we deep-link.
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 900},
            locale="ar-EG",
        )
        page = await context.new_page()

        requests_log = []

        def on_request(req):
            requests_log.append({"url": req.url, "method": req.method, "type": "request"})

        def on_response(res):
            requests_log.append({"url": res.url, "status": res.status, "type": "response"})

        page.on("request", on_request)
        page.on("response", on_response)

        print(f"--- Step 1: visiting homepage first to establish session ---")
        await page.goto("https://www.nbe.com.eg/", timeout=30000, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)

        print(f"--- Step 2: navigating to target page ---\n{url}\n")
        await page.goto(url, timeout=30000, wait_until="domcontentloaded")
        await page.wait_for_timeout(6000)  # give the SPA time to fire its data calls

        print("\n=== Network activity during target page load ===\n")
        api_like = [
            r for r in requests_log
            if r["type"] == "response" and (
                "/api/" in r["url"].lower()
                or "service" in r["url"].lower()
                or r["url"].lower().endswith(".json")
            )
        ]
        if not api_like:
            print("!! No API-like responses detected at all. The SPA likely never "
                  "fired its data-fetch call for this route when deep-linked directly. "
                  "This points to a client-side routing/referrer issue, not a server block.")
        else:
            for r in api_like:
                flag = ""
                if r["status"] in (401, 403, 429):
                    flag = "  <<< LOOKS LIKE BOT/AUTH BLOCK"
                elif r["status"] >= 500:
                    flag = "  <<< SERVER ERROR"
                elif r["status"] != 200:
                    flag = "  <<< NON-200"
                print(f"{r['status']}  {r['url']}{flag}")

        # Also dump ALL requests so nothing is missed, in case the API
        # pattern doesn't match "/api/"/"service"/".json" for this site.
        print("\n=== ALL network requests (for manual inspection) ===\n")
        for r in requests_log:
            if r["type"] == "response":
                print(f"{r['status']}  {r['url']}")

        html_snippet = await page.content()
        if "2109372177" in html_snippet or "خطأ فني" in html_snippet:
            print("\n!! The known fallback error message IS present in the final DOM.")

        await browser.close()


if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 else (
        "https://www.nbe.com.eg/NBE/E/#/AR/CertificatesRatesForeignCurrency"
    )
    asyncio.run(diagnose(target_url))
