"""Clean text extracted by Playwright re-scrape."""

from __future__ import annotations

import re

from shared.document_quality import strip_site_chrome

PLAYWRIGHT_NOISE_LINES = re.compile(
    r"^(?:"
    r"Nab Bar Menu|Change Language|Search Bar|NBE logo|Mega Menu|"
    r"Catagory landing div|Floating Banner|SiteMap|Copyrights|Confirm|"
    r"أفراد|الشركات|المشروعات الصغيرة|بلاتينم|الوظائف|"
    r"هذه الرسالة تحتوي على بيانات شخصية|برجاء عدم الإفصاح|"
    r"This message contains personal or financial|Please do not disclose|"
    r"برجاء العلم أن الخدمة للاستعلام فقط|مرحبا،|دعنا نتحدث|"
    r"#(?:Category|Product)[A-Za-z]*Container#|"
    r"1187 كورنيش النيل|محليا: 19623|اتصل بنا على الفور"
    r")$",
    re.IGNORECASE,
)

FOOTER_BLOCK = re.compile(
    r"SiteMap\s*\n.*?Copyrights\s*\n(?:Floating Banner.*?Confirm\s*\n)?",
    re.DOTALL | re.IGNORECASE,
)

CHAT_BLOCK = re.compile(
    r"برجاء العلم أن الخدمة للاستعلام فقط.*?دعنا نتحدث.*?(?:\n\n|\Z)",
    re.DOTALL,
)


def clean_playwright_content(content: str) -> str:
    content = FOOTER_BLOCK.sub("\n", content)
    content = CHAT_BLOCK.sub("\n", content)
    content = strip_site_chrome(content)

    kept: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if PLAYWRIGHT_NOISE_LINES.match(line):
            continue
        if len(line) < 2:
            continue
        kept.append(line)

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return cleaned
