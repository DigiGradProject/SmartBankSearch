"""Generate validation_set.jsonl — paraphrased hold-out that avoids golden literals.

Queries are written so they do not literally equal any golden_set query and avoid
verbatim copies of intent-rule sample phrases. Intent labels remain for metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "retrieval" / "validation_set.jsonl"
GOLDEN = ROOT / "tests" / "retrieval" / "golden_set.jsonl"

# (query, language, intent, expected_url_contains, forbidden_url_contains|None)
TEMPLATES: list[tuple[str, str, str, str, str | None]] = [
    # exchange_rate — avoid exact "سعر الصرف" / "أسعار العملات" / "exchange rates"
    ("كام سعر الدولار النهاردة في الأهلي؟", "ar", "exchange_rate", "ExchangeRatesAndCurrencyConverter", "CertificatesRates"),
    ("ايه سعر اليورو مقابل الجنيه عندكم؟", "ar", "exchange_rate", "ExchangeRatesAndCurrencyConverter", "CertificatesRates"),
    ("محتاج جدول أسعار العملات الأجنبية", "ar", "exchange_rate", "ExchangeRatesAndCurrencyConverter", None),
    ("عايز اعرف قيمة الجنيه قصاد الدولار", "ar", "exchange_rate", "ExchangeRatesAndCurrencyConverter", None),
    ("قولي سعر شراء وبيع الدولار البنكي", "ar", "exchange_rate", "ExchangeRatesAndCurrencyConverter", None),
    ("what is the USD to EGP buying price today", "en", "exchange_rate", "ExchangeRatesAndCurrencyConverter", None),
    ("NBE forex board for EUR", "en", "exchange_rate", "ExchangeRatesAndCurrencyConverter", None),
    ("show me today's banknote FX table", "en", "exchange_rate", "ExchangeRatesAndCurrencyConverter", None),
    ("how many pounds for one dollar at NBE", "en", "exchange_rate", "ExchangeRatesAndCurrencyConverter", None),
    ("current USD sell rate please", "en", "exchange_rate", "ExchangeRatesAndCurrencyConverter", None),
    ("بكام الاسترليني النهاردة؟", "ar", "exchange_rate", "ExchangeRatesAndCurrencyConverter", None),
    ("عايز محول من دولار لجنيه", "ar", "exchange_rate", "ExchangeRatesAndCurrencyConverter", None),
    ("FX rates for tourists in EGP", "en", "exchange_rate", "ExchangeRatesAndCurrencyConverter", None),
    ("الأهلي بيعالجنيه كام دولار", "ar", "exchange_rate", "ExchangeRatesAndCurrencyConverter", None),
    ("need live currency board NBE", "en", "exchange_rate", "ExchangeRatesAndCurrencyConverter", None),
    # personal_loan
    ("عايز أقساط تمويل شخصي بدون ضامن", "ar", "personal_loan", "Loans", "NewsCat"),
    ("قد ايه تمويل فردي من البنك؟", "ar", "personal_loan", "Loans", None),
    ("شروط أخذ قرض لأغراض شخصية", "ar", "personal_loan", "Loans", None),
    ("محتاج تمويل للسفر أو الجواز", "ar", "personal_loan", "Loans", None),
    ("نسبة الفائدة على التمويل الشخصي", "ar", "personal_loan", "Loans", None),
    ("apply for an unsecured personal financing product", "en", "personal_loan", "Loans", None),
    ("NBE retail loan installments explained", "en", "personal_loan", "Loans", None),
    ("how to get consumer financing from NBE", "en", "personal_loan", "Loans", None),
    ("personal financing eligibility documents", "en", "personal_loan", "Loans", None),
    ("خواص القروض الفردية المتاحة", "ar", "personal_loan", "Loans", None),
    # credit_card / card_types
    ("عايز كارت ائتمان بمكافآت سفر", "ar", "credit_card", "CreditCards", None),
    ("ما هي بطاقات الشراء بالتقسيط؟", "ar", "credit_card", "CreditCards", None),
    ("قارن بين بطاقات الكريدت عندكم", "ar", "credit_card", "CreditCards", None),
    ("نسبة الفوائد على الكريديت كارد", "ar", "credit_card", "CreditCards", None),
    ("which revolving credit cards does NBE offer", "en", "credit_card", "CreditCards", None),
    ("visa platinum rewards card details", "en", "credit_card", "CreditCards", None),
    ("list of plastic payment products", "en", "card_types", "Cards", None),
    ("ايه أنواع الكروت البنكية عند الأهلي", "ar", "card_types", "Cards", None),
    ("بطاقة خصم مباشر ولا ائتمان؟", "ar", "card_types", "Cards", None),
    ("debit versus credit plastic options", "en", "card_types", "Cards", None),
    # certificate_rate / buy / types
    ("كام عائد ادخار لمدة 12 شهر بالجنيه", "ar", "certificate_rate", "LocalCertificatesID", "ExchangeRates"),
    ("فايدة شهادات الدولار السنوية", "ar", "certificate_rate", "ForigenCertificatesID", "ExchangeRates"),
    ("interest on USD saving instruments", "en", "certificate_rate", "ForigenCertificatesID", "ExchangeRates"),
    ("EGP one-year deposit yield table", "en", "certificate_rate", "LocalCertificatesID", "ExchangeRates"),
    ("عايز أشتري أداة ادخار جديدة", "ar", "certificate_buy", "CertificatesID", None),
    ("how do I purchase a savings certificate", "en", "certificate_buy", "CertificatesID", None),
    ("اعرض لي أصناف أدوات الادخار", "ar", "certificate_types", "CertificatesID", None),
    ("catalog of NBE savings certificates", "en", "certificate_types", "CertificatesID", None),
    ("شهادات لأجل ثلاث سنوات متاحة؟", "ar", "certificate_types", "CertificatesID", None),
    ("compare fixed-income deposit products", "en", "certificate_types", "CertificatesID", None),
    # account_open
    ("إزاي أبدأ حساب جديد في الفرع؟", "ar", "account_open", "Accounts", "OpenYourBankAccountInEgypt"),
    ("المستندات اللازمة لعمل حساب جاري", "ar", "account_open", "CurrentAccountsID", "OpenYourBankAccountInEgypt"),
    ("عايز حساب توفير عادي للمقيمين", "ar", "account_open", "Accounts", "OpenYourBankAccountInEgypt"),
    ("steps to open a resident current account", "en", "account_open", "CurrentAccountsID", "OpenYourBankAccountInEgypt"),
    ("documents required for a local savings account", "en", "account_open", "Accounts", None),
    ("فتح حساب للمصريين جوه مصر", "ar", "account_open", "CurrentAccountsID", "OpenYourBankAccountInEgypt"),
    # branch / atm
    ("أقرب فرع ليا فين؟", "ar", "branch_locator", "ATMBranch", None),
    ("فين ألقى صراف قريب مني", "ar", "atm_locator", "ATMBranch", None),
    ("locate nearest NBE branch on the map", "en", "branch_locator", "ATMBranch", None),
    ("where can I withdraw from an NBE cash machine", "en", "atm_locator", "ATMBranch", None),
    ("عناوين فروع الإسكندرية", "ar", "branch_locator", "ATMBranch", None),
    # offers / faq / corporate / sme / wallet
    ("في خصومات أو برومو حالية؟", "ar", "offers", "", None),
    ("current promotional campaigns", "en", "offers", "", None),
    ("اسألوني عن الحاجات المتكررة", "ar", "faq", "", None),
    ("common customer questions hub", "en", "faq", "", None),
    ("حلول بنكية للشركات الكبيرة", "ar", "corporate", "", None),
    ("banking solutions for large enterprises", "en", "corporate", "", None),
    ("تمويل أصحاب الورش الصغيرة", "ar", "sme", "", None),
    ("financing for small workshops", "en", "sme", "", None),
    ("محفظة الموبايل للدفع", "ar", "wallet", "", None),
    ("mobile wallet cash-out options", "en", "wallet", "", None),
]

# Paraphrase expanders to reach 100–300 without duplicating golden strings.
AR_FX_VARIANTS = [
    "محتاج أعرف تسعير العملة الأجنبية اليوم",
    "عايز شاشة أسعار البيع والشراء للنقد الأجنبي",
    "بكم يبيع البنك ورقة الدولار",
    "عرض أسعار الحوالات والعملات الورقية",
    "اللوحة اليومية لتقييم العملات",
]
EN_FX_VARIANTS = [
    "publisher of daily FX quotes for customers",
    "tell me the mid-market vs bank cash rate",
    "Egyptian pound versus dollar bank table",
    "NBE foreign exchange ticker for clients",
    "retail banknote conversion prices",
]
AR_LOAN_VARIANTS = [
    "تمويل أفراد بدون رهن عقاري",
    "قسط شهري مناسب لموظف حكومي",
    "حد الائتمان الشخصي المسموح",
    "عروض التقسيط الاستهلاكي",
]
EN_LOAN_VARIANTS = [
    "payroll-linked consumer financing",
    "unsecured retail loan brochure",
    "monthly installment loan calculator info",
]
AR_CARD_VARIANTS = [
    "كارت بمزايا تأمين سفر",
    "بطاقة تسوق أونلاين مؤمنة",
    "كريدت بحدود مرتفعة",
]
EN_CARD_VARIANTS = [
    "secure ecommerce revolving card",
    "travel benefits credit plastic",
]
AR_CERT_VARIANTS = [
    "عائد شهادة الادخار المحلية سنة",
    "فايدة أدوات الإيداع بالعملة الصعبة",
    "جدول أرباح شهادات الجنيه",
]
EN_CERT_VARIANTS = [
    "local currency deposit coupon rates",
    "foreign currency certificate APR sheet",
]
AR_ACCOUNT_VARIANTS = [
    "إجراءات بدء علاقة حسابية جديدة",
    "الأوراق لفتح حساب توفير للمقيم",
    "حساب جاري برواتب محولة",
]
EN_ACCOUNT_VARIANTS = [
    "resident retail account onboarding checklist",
    "open an EGP transactional account",
]


def _load_golden_queries() -> set[str]:
    if not GOLDEN.exists():
        return set()
    queries: set[str] = set()
    for line in GOLDEN.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        queries.add(json.loads(line)["query"].strip())
    return queries


def _row(
    query: str,
    language: str,
    intent: str,
    expected: str,
    forbidden: str | None,
) -> dict:
    row: dict = {
        "query": query,
        "language": language,
        "intent": intent,
        "expected_url_contains": expected,
    }
    if forbidden:
        row["forbidden_url_contains"] = forbidden
    return row


def build_rows() -> list[dict]:
    golden = _load_golden_queries()
    rows: list[dict] = []
    seen: set[str] = set()

    def add(query: str, language: str, intent: str, expected: str, forbidden: str | None) -> None:
        q = query.strip()
        if not q or q in seen or q in golden:
            return
        seen.add(q)
        rows.append(_row(q, language, intent, expected, forbidden))

    for item in TEMPLATES:
        add(*item)

    for q in AR_FX_VARIANTS:
        add(q, "ar", "exchange_rate", "ExchangeRatesAndCurrencyConverter", "CertificatesRates")
    for q in EN_FX_VARIANTS:
        add(q, "en", "exchange_rate", "ExchangeRatesAndCurrencyConverter", None)
    for q in AR_LOAN_VARIANTS:
        add(q, "ar", "personal_loan", "Loans", "NewsCat")
    for q in EN_LOAN_VARIANTS:
        add(q, "en", "personal_loan", "Loans", None)
    for q in AR_CARD_VARIANTS:
        add(q, "ar", "credit_card", "CreditCards", None)
    for q in EN_CARD_VARIANTS:
        add(q, "en", "credit_card", "CreditCards", None)
    for q in AR_CERT_VARIANTS:
        forbidden = "ExchangeRates"
        expected = "LocalCertificatesID" if "محلي" in q or "جنيه" in q else "CertificatesID"
        if "صعبة" in q or "أجنب" in q:
            expected = "ForigenCertificatesID"
        add(q, "ar", "certificate_rate", expected, forbidden)
    for q in EN_CERT_VARIANTS:
        expected = "LocalCertificatesID" if "local" in q.lower() else "ForigenCertificatesID"
        add(q, "en", "certificate_rate", expected, "ExchangeRates")
    for q in AR_ACCOUNT_VARIANTS:
        add(q, "ar", "account_open", "Accounts", "OpenYourBankAccountInEgypt")
    for q in EN_ACCOUNT_VARIANTS:
        add(q, "en", "account_open", "Accounts", "OpenYourBankAccountInEgypt")

    # Systematic expansions to land in 100–300 range.
    prefixes_ar = ["ممكن توضح", "لو سمحت", "محتاج معلومة عن", "أسئلة عن"]
    prefixes_en = ["please explain", "could you clarify", "looking for info on", "customer question about"]
    seeds = list(rows)
    for seed in seeds:
        if len(rows) >= 220:
            break
        q = seed["query"]
        lang = seed["language"]
        prefixes = prefixes_ar if lang == "ar" else prefixes_en
        for prefix in prefixes:
            if len(rows) >= 220:
                break
            add(
                f"{prefix} {q}",
                lang,
                seed["intent"],
                seed.get("expected_url_contains", ""),
                seed.get("forbidden_url_contains"),
            )

    return rows


def main() -> None:
    rows = build_rows()
    assert 100 <= len(rows) <= 300, f"expected 100–300 rows, got {len(rows)}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} cases -> {OUT}")


if __name__ == "__main__":
    main()
