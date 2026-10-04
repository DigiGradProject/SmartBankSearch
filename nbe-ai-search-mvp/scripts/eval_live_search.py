"""Live end-to-end evaluation of /v1/search against the running API.

For each curated query we check:
  1. CITATION  — at least one citation URL matches `expected` (any-of substrings),
                 and NO citation URL matches `forbidden` (any-of substrings).
  2. ANSWER    — request was answered (unless expect_answered=False) and the
                 answer text contains at least one of `answer_keywords`
                 (case-insensitive; loose LLM sanity check).
  3. Intent    — recorded from the response (informational).

Edge/robustness cases are marked `strict=False`: they are reported but never
count as failures (e.g. nonsense or off-topic queries where abstention is
acceptable behaviour).

Usage:
    .venv/bin/python scripts/eval_live_search.py [--base http://localhost:7000]
        [--offset N] [--limit N] [--tag NAME] [--summary] [--mode ai|traditional]

Every finished case is appended immediately to
    data/reports/live_eval_<mode>_<tag>.jsonl
so partial runs are never lost. Run several chunks with --offset/--limit,
then print the final report with --summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXCH = "ExchangeRatesAndCurrencyConverter"

# (query, language, expected_any, forbidden_any, answer_keywords_any, expect_answered, strict, note)
CASES: list[dict] = [
    # ---------- Exchange rates (controls — must keep working) ----------
    dict(q="What is the exchange rate of USD to EGP today?", l="en", exp=[EXCH], forb=["CertificatesRates", "LocalCertificatesID"], kw=["usd", "dollar", "egp", "pound"]),
    dict(q="exchange rates", l="en", exp=[EXCH], forb=["CertificatesRates"]),
    dict(q="currency converter EUR to EGP", l="en", exp=[EXCH], forb=["CertificatesRates"], kw=["eur", "euro"]),
    dict(q="سعر الدولار اليوم", l="ar", exp=[EXCH], forb=["CertificatesRates"], kw=["دولار"]),
    dict(q="أسعار العملات", l="ar", exp=[EXCH]),
    dict(q="تحويل من الدولار إلى الجنيه", l="ar", exp=[EXCH], kw=["دولار", "تحويل"]),
    # ---------- Certificates ----------
    dict(q="كم فايده شهادة سنة بالجنيه؟", l="ar", exp=["LocalCertificatesID", "CertificatesID", "InvestmentCertificateCatID"], forb=[EXCH], kw=["شهاد", "عائد", "18", "١٨"]),
    dict(q="certificate of deposit interest rate in EGP", l="en", exp=["LocalCertificatesID", "CertificatesID", "InvestmentCertificateCatID", "BeladyCertificateID"], forb=[EXCH], kw=["certificate", "rate", "interest"]),
    dict(q="شراء شهادة ادخار", l="ar", exp=["CertificatesID", "LocalCertificatesID", "InvestmentCertificateCatID"], kw=["شهاد"]),
    dict(q="أسعار الشهادات بالعملة الأجنبية", l="ar", exp=["ForigenCertificatesID", "ForeignCurrency"], forb=[EXCH]),
    dict(q="Belady certificate 3 years", l="en", exp=["BeladyCertificateID", "beladythreeyears"], kw=["belady", "certificate"]),
    dict(q="شهادات بلادي", l="ar", exp=["BeladyCertificateID", "belady"], kw=["بلادي", "شهاد"]),
    # ---------- Cards ----------
    dict(q="credit cards NBE", l="en", exp=["CreditCards"], kw=["credit", "card"]),
    dict(q="بطاقات ائتمان", l="ar", exp=["CreditCards", "CreditCardsID"], kw=["ائتمان", "بطاقات"]),
    dict(q="What credit cards does NBE offer?", l="en", exp=["CreditCards", "CreditCardsID"], kw=["credit", "card"]),
    dict(q="فيزا كلاسيك", l="ar", exp=["CreditCardsID", "DepitCardsID"], kw=["فيزا", "كلاسيك"]),
    dict(q="debit cards", l="en", exp=["DebitCards", "DepitCardsID"], kw=["debit", "card"]),
    dict(q="بطاقات الخصم", l="ar", exp=["DebitCards", "DepitCardsID"], kw=["خصم", "بطاقات"]),
    # ---------- Loans ----------
    dict(q="personal loan NBE", l="en", exp=["Loans"], kw=["loan"]),
    dict(q="قرض شخصي", l="ar", exp=["Loans"], kw=["قرض"]),
    dict(q="car loan", l="en", exp=["NewAutoLoanID", "Loans"], kw=["auto", "car", "loan"]),
    dict(q="قرض سيارات", l="ar", exp=["NewAutoLoanID", "Loans", "carsandservices"], kw=["سيارات", "قرض"]),
    dict(q="مبادرات التمويل العقاري", l="ar", exp=["CBEsinitiativeID", "%22305%22", "%22306%22", "%22309%22", "%22310%22"], kw=["عقار", "تمويل", "مبادر"]),
    # ---------- Accounts ----------
    dict(q="فتح حساب بنكي", l="ar", exp=["Accounts", "CurrentAccountsID"], kw=["حساب"]),
    dict(q="how to open a bank account", l="en", exp=["Accounts", "CurrentAccountsID"], kw=["account", "open"]),
    dict(q="لو عايز افتح حساب بنكي اي الاوراق المطلوبة؟", l="ar", exp=["CurrentAccountsID", "AccountsID"], forb=["OpenYourBankAccountInEgypt"], kw=["حساب", "أوراق", "اوراق"]),
    dict(q="حساب التوفير", l="ar", exp=["PlatinumSaving", "AccountsID", "Accounts"], kw=["توفير", "حساب"]),
    dict(q="savings account interest", l="en", exp=["PlatinumSaving", "AccountsID", "Accounts"], kw=["saving", "account"]),
    # ---------- Digital banking / brand-title regression set ----------
    dict(q="What is National Bank of Egypt - Al Ahly Points?", l="en", exp=["ahlypoints"], forb=[EXCH], kw=["points", "loyalty"]),
    dict(q="What is National Bank of Egypt - Al Ahly Business?", l="en", exp=["AlAhlyBusiness"], forb=[EXCH], kw=["business", "corporate"]),
    dict(q="What is Al Ahly Net - Platinum?", l="en", exp=["DigitalBankingPlatinum"], forb=[EXCH], kw=["net", "digital", "banking", "internet"]),
    dict(q="Al Ahly Net online banking", l="en", exp=["%2250%22", "DigitalBankingPlatinum", "AhlyNet"], forb=[EXCH], kw=["net", "digital", "online", "banking"]),
    dict(q="الاهلي نت", l="ar", exp=["AhlyNet", "293", "%2250%22", "ElectronicServices"], forb=[EXCH], kw=["نت", "انترنت", "إنترنت", "الكترون"]),
    dict(q="الاهلي نقاط", l="ar", exp=["ahlypoints"], forb=[EXCH], kw=["نقاط"]),
    dict(q="NBE mobile app", l="en", exp=["%2250%22", "AlAhlyMobile", "Mobile"], kw=["mobile", "app"]),
    # ---------- Branches / ATM ----------
    dict(q="الفروع", l="ar", exp=["ATMBranch"], kw=["فرع", "فروع"]),
    dict(q="Where is the nearest NBE branch?", l="en", exp=["ATMBranch"], kw=["branch", "atm"]),
    dict(q="ATM locations", l="en", exp=["ATMBranch"], kw=["atm"]),
    # ---------- Corporate / SME / offers / FAQ ----------
    dict(q="عروض البنك", l="ar", exp=["DiscountOffers", "installmentoffers", "Offer"], kw=["عروض", "خصم", "تخفيض"]),
    dict(q="خدمات الشركات", l="ar", exp=["CorporateAll", "%2274%22", "Corporate"], kw=["شركات", "شركات"]),
    dict(q="المشروعات الصغيرة والمتوسطة", l="ar", exp=["SMEs"], kw=["مشروعات", "صغيرة", "متوسطة"]),
    dict(q="الأسئلة الشائعة", l="ar", exp=["FAQs"], kw=["أسئلة", "شائعة"]),
    dict(q="fees and charges", l="en", exp=["Tariff"], kw=["fees", "charges", "tariff"]),
    # ---------- General / about ----------
    dict(q="What is National Bank of Egypt?", l="en", exp=["AboutUS", "About"], strict=False, kw=["bank", "egypt", "national", "founded", "largest"]),
    # ---------- Mixed language ----------
    dict(q="سعر صرف الدولار USD", l="auto", exp=[EXCH], forb=["CertificatesRates"], kw=["دولار", "usd"]),
    # ---------- Robustness (informational — abstention acceptable) ----------
    dict(q="asdkjhqwe zzz qwerty", l="en", expect_answered=False, strict=False),
    dict(q="What is the weather in Cairo tomorrow?", l="en", strict=False),
]


def run_case(base: str, case: dict, timeout: float, mode: str = "ai") -> dict:
    body: dict = {"query": case["q"], "language": case["l"]}
    if mode == "traditional":
        body["mode"] = "traditional"
        body["limit"] = 10
    else:
        body["debug"] = True
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{base}/v1/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:  # server answered with an error code
        return dict(case, error=f"HTTP {exc.code}: {exc.read().decode()[:200]}", latency=round(time.time() - t0, 1))
    except Exception as exc:  # noqa: BLE001
        return dict(case, error=f"{type(exc).__name__}: {exc}", latency=round(time.time() - t0, 1))
    latency = round(time.time() - t0, 1)

    citations = body.get("citations") or []
    results = body.get("results") or []
    if mode == "traditional":
        # Keyword mode: results list replaces citations/answer.
        urls = [r.get("url") or "" for r in results]
        answer = " ".join(
            f"{r.get('title', '')} {r.get('snippet', '')}" for r in results
        )
        answered = bool(results)
    else:
        urls = [c.get("url") or "" for c in citations]
        answer = (body.get("answer") or "")
        answered = bool(body.get("answered"))
    strict = case.get("strict", True)

    expected = case.get("exp", [])
    forbidden = case.get("forb", [])
    keywords = case.get("kw", [])
    expect_answered = case.get("expect_answered", True)

    exp_hit = (not expected) or any(any(sub in u for sub in expected) for u in urls)
    forb_clean = (not forbidden) or not any(any(sub in u for sub in forbidden) for u in urls)
    answered_ok = answered == expect_answered
    low = answer.lower()
    kw_hit = (not keywords) or any(k.lower() in low for k in keywords)

    checks = {
        "citation_ok": exp_hit,
        "forbidden_ok": forb_clean,
        "answered_ok": answered_ok,
        "answer_kw_ok": kw_hit if expect_answered else True,
    }
    failed = [name for name, ok in checks.items() if not ok]
    status = "PASS" if (not failed or not strict) else "FAIL"
    if failed and not strict:
        status = "WARN"

    return dict(
        case,
        status=status,
        failed_checks=failed,
        intent=body.get("intent") or ("traditional_search" if mode == "traditional" else None),
        confidence=body.get("confidence"),
        answered=answered,
        abstention_reason=body.get("abstention_reason"),
        answer=answer[:400],
        citation_urls=urls,
        citation_titles=[r.get("title") or "" for r in results] or [c.get("title") or "" for c in citations],
        latency=latency,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live /v1/search evaluation")
    parser.add_argument("--base", default="http://localhost:7000")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--offset", type=int, default=0, help="start at case N (0-based)")
    parser.add_argument("--limit", type=int, default=0, help="only N cases (0 = all)")
    parser.add_argument("--tag", default=datetime.now().strftime("%H%M%S"))
    parser.add_argument("--summary", action="store_true", help="print report from existing JSONL and exit")
    parser.add_argument(
        "--mode",
        choices=["ai", "traditional"],
        default="ai",
        help="ai = full RAG answers (default); traditional = BM25 keyword results",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out_dir = ROOT / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"live_eval_{args.mode}_{args.tag}.jsonl"

    if args.summary:
        summarize(out)
        return

    cases = CASES[args.offset :]
    if args.limit:
        cases = cases[: args.limit]

    results = []
    with out.open("a", encoding="utf-8") as sink:
        for i, case in enumerate(cases, args.offset + 1):
            row = run_case(args.base, case, args.timeout, mode=args.mode)
            results.append(row)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            status = row.get("status", "ERROR")
            failed = ",".join(row.get("failed_checks", [])) or (row.get("error", "")[:60] if row.get("error") else "")
            print(
                f"[{i:>2}/{len(CASES)}] {status:<5} {row['q'][:58]:<58} "
                f"intent={row.get('intent') or '-':<22} "
                f"lat={row.get('latency', 0):>5}s {failed}",
                flush=True,
            )


def summarize(out: Path) -> None:
    results = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    strict_rows = [r for r in results if r.get("strict", True) and "error" not in r]
    errors = [r for r in results if "error" in r]
    passed = sum(1 for r in strict_rows if r.get("status") == "PASS")
    warns = [r for r in results if r.get("status") == "WARN"]

    print("==== SUMMARY ====")
    print(f"total={len(results)} strict={len(strict_rows)} passed={passed} "
          f"failed={len(strict_rows) - passed} warn={len(warns)} errors={len(errors)}")
    if strict_rows:
        print(f"avg latency (strict): {sum(r.get('latency', 0) for r in strict_rows) / len(strict_rows):.1f}s")

    if strict_rows:
        print("\n==== FAILURES ====")
        for r in strict_rows:
            if r.get("status") == "FAIL":
                print(f"- {r['q']}  -> {','.join(r.get('failed_checks', []))}")
                print(f"    citations: {r.get('citation_urls')}")
    if warns:
        print("\n==== INFORMATIONAL WARNINGS (not failures) ====")
        for r in warns:
            print(f"- {r['q']}  -> {','.join(r.get('failed_checks', []))}")
            print(f"    citations: {r.get('citation_urls')}")


if __name__ == "__main__":
    main()
