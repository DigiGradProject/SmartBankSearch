"""BGE-M3 intent prototypes — paraphrase-rich, not tied to golden-set regex."""

from __future__ import annotations

# Multiple prototypes per intent improve semantic generalization.
INTENT_PROTOTYPES: dict[str, dict[str, tuple[str, ...]]] = {
    "exchange_rate": {
        "ar": (
            "أسعار العملات سعر الصرف تحويل العملات دولار يورو جنيه",
            "الدولار بكام النهاردة سعر شراء وبيع العملة الأجنبية",
            "عايز أعرف سعر بيع الدولار محول العملات",
            "جدول أسعار العملات الأجنبية مقابل الجنيه المصري",
            "اشتري دولار بكام سعر الصرف الرسمي",
        ),
        "en": (
            "exchange rates currency converter USD EGP banknote rate",
            "how much is one dollar in Egyptian pounds today",
            "USD buying and selling rate NBE forex board",
            "foreign currency conversion table live rates",
            "EUR GBP forex rates against EGP",
        ),
    },
    "certificate_rate": {
        "ar": (
            "عائد شهادة فايدة نسبة شهادات ادخار بالجنيه",
            "كم فايدة شهادة سنة بالعملة المحلية",
            "أسعار الشهادات بالعملة الأجنبية عائد الادخار",
        ),
        "en": (
            "certificate yield interest rate savings deposit",
            "EGP certificate APR one year yield",
            "foreign currency certificate interest rate",
        ),
    },
    "certificate_types": {
        "ar": (
            "أنواع الشهادات شهادات ادخار بلادي استثمار",
            "قائمة شهادات الادخار المتاحة في البنك",
            "ما هي أنواع الشهادات البنكية",
        ),
        "en": (
            "types of saving certificates deposit products",
            "catalog of NBE certificate offerings",
        ),
    },
    "certificate_buy": {
        "ar": (
            "شراء شهادة ادخار جديدة اشتراك في شهادة",
            "عايز أشتري شهادة من البنك",
        ),
        "en": (
            "buy a savings certificate open certificate subscription",
            "how to purchase NBE deposit certificate",
        ),
    },
    "personal_loan": {
        "ar": (
            "قرض شخصي تمويل أفراد قسط شهري",
            "شروط القرض الشخصي بدون ضامن",
            "تمويل استهلاكي للموظفين",
        ),
        "en": (
            "personal loan unsecured consumer financing",
            "retail loan installments eligibility NBE",
        ),
    },
    "credit_card": {
        "ar": (
            "بطاقات ائتمان كريدت كارد مزايا البطاقة",
            "بطاقة تسوق أونلاين بحد ائتماني",
            "كارت ائتمان بمكافآت سفر",
        ),
        "en": (
            "credit cards revolving NBE visa mastercard",
            "rewards credit card benefits and limits",
        ),
    },
    "card_types": {
        "ar": (
            "أنواع البطاقات ائتمان خصم مدفوعة مقدما",
            "قارن بين بطاقات الكريدت والخصم المباشر",
            "ايه أنواع الكروت البنكية",
        ),
        "en": (
            "types of bank cards debit credit prepaid",
            "compare NBE plastic payment products",
        ),
    },
    "debit_card": {
        "ar": ("بطاقة خصم مباشر debit card",),
        "en": ("debit card direct debit NBE",),
    },
    "account_open": {
        "ar": (
            "فتح حساب بنكي الأوراق المطلوبة حساب جاري",
            "إزاي أبدأ حساب جديد المستندات اللازمة",
            "فتح حساب توفير للمقيمين في مصر",
            "لو عايز افتح حساب بنكي اي الاوراق المطلوبة",
        ),
        "en": (
            "open bank account required documents current account",
            "resident account onboarding checklist EGP account",
            "steps to open a savings account NBE",
        ),
    },
    "branch_locator": {
        "ar": (
            "الفروع أقرب فرع موقع الفرع",
            "أقرب فرع ليا فين عناوين الفروع",
            "فين أقرب فرع للأهلي",
        ),
        "en": (
            "branch locator find nearest NBE branch",
            "branch addresses and locations map",
        ),
    },
    "atm_locator": {
        "ar": (
            "الصراف الآلي ماكينات الصرف أقرب ATM",
            "فين ألقى صراف قريب مني",
        ),
        "en": (
            "ATM locator cash machine near me",
            "where can I withdraw from NBE ATM",
        ),
    },
    "wallet": {
        "ar": ("فون كاش محفظة الموبايل للدفع",),
        "en": ("phone cash mobile wallet payment",),
    },
    "digital_banking": {
        "ar": (
            "الأهلي نت موبايل بانكنج انترنت بنكي",
            "خدمات رقمية تطبيق البنك",
        ),
        "en": (
            "internet banking mobile banking digital services",
            "NBE online banking app login",
        ),
    },
    "offers": {
        "ar": ("عروض البنك خصومات برومو",),
        "en": ("bank offers promotions discounts NBE",),
    },
    "news": {
        "ar": ("أخبار البنك بيان صحفي",),
        "en": ("NBE news press release",),
    },
    "reports": {
        "ar": ("تقارير البنك القوائم المالية تقرير سنوي",),
        "en": ("annual report financial statements NBE",),
    },
    "corporate": {
        "ar": ("خدمات الشركات قطاع الشركات",),
        "en": ("corporate banking business banking",),
    },
    "sme": {
        "ar": ("المشروعات الصغيرة والمتوسطة تمويل المنشآت",),
        "en": ("SME small medium enterprises financing",),
    },
    "faq": {
        "ar": ("أسئلة شائعة FAQ",),
        "en": ("frequently asked questions help center",),
    },
}

INTENT_CATEGORIES: dict[str, str] = {
    "exchange_rate": "exchange_rates",
    "certificate_rate": "certificates",
    "certificate_types": "certificates",
    "certificate_buy": "certificates",
    "personal_loan": "loans",
    "credit_card": "cards",
    "card_types": "cards",
    "debit_card": "cards",
    "account_open": "accounts",
    "branch_locator": "branches",
    "atm_locator": "branches",
    "wallet": "wallet",
    "digital_banking": "digital_banking",
    "offers": "offers",
    "news": "news",
    "reports": "reports",
    "corporate": "corporate",
    "sme": "sme",
    "faq": "general",
    "customer_service": "contact",
    "password_reset": "digital_banking",
}
