"""Enterprise document cleaner — strip chrome, menus, widgets, placeholders."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from shared.arabic_normalize import normalize_text
from shared.document_quality import is_menu_heavy_text, strip_site_chrome

FOOTER_MARKERS = (
    "جميع الحقوق محفوظة",
    "All Rights Reserved",
    "© National Bank of Egypt",
    "بنك الأهلي المصري ©",
    "Follow us",
    "تابعنا على",
)

NAV_LINE = re.compile(
    r"^(أهلا بك|Welcome|تسجيل الخروج|Logout|القوائم الرئيسية|Main Menu|"
    r"عرض الصفحة الشخصية|Notifications|Favorite List|Follow Up|"
    r"حماية حقوق العملاء|Consumer Protection)\b",
    re.I,
)

WIDGET_LINE = re.compile(
    r"(swiper|cookie|subscribe|newsletter|chatbot|whatsapp widget|"
    r"share this|أرسل الصفحة|print this)",
    re.I,
)

TEMPLATE_PLACEHOLDER = re.compile(
    r"(\{\{[^}]+\}\}|%\{[^}]+\}%|\[\[[^\]]+\]\]|null\s*undefined|TODO_CONTENT|lorem ipsum)",
    re.I,
)

HTML_ENTITY = re.compile(r"&(?:nbsp|amp|lt|gt|quot|#\d+);", re.I)
MULTI_BLANK = re.compile(r"\n{3,}")
MULTI_SPACE = re.compile(r"[ \t]{2,}")
PIPE_MENU = re.compile(r"(?:\s*\|\s*[^\n|]{2,40}){4,}")


def _strip_html(content: str) -> str:
    if "<" in content and ">" in content:
        soup = BeautifulSoup(content, "lxml")
        for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
            tag.decompose()
        content = soup.get_text("\n", strip=True)
    return HTML_ENTITY.sub(" ", content)


def _drop_footer(content: str) -> str:
    lower = content.lower()
    cut = len(content)
    for marker in FOOTER_MARKERS:
        idx = lower.find(marker.lower())
        if idx > int(len(content) * 0.55):
            cut = min(cut, idx)
    return content[:cut].rstrip()


def _filter_lines(content: str) -> str:
    kept: list[str] = []
    prev = ""
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if NAV_LINE.match(line) or WIDGET_LINE.search(line):
            continue
        if PIPE_MENU.search(line):
            continue
        if line == prev:
            continue
        kept.append(line)
        prev = line
    return "\n".join(kept)


def clean_document_text(content: str, language: str = "ar") -> str:
    """Full cleaning pipeline for scraped NBE pages."""
    content = _strip_html(content)
    content = strip_site_chrome(content)
    content = TEMPLATE_PLACEHOLDER.sub(" ", content)
    content = _drop_footer(content)
    content = _filter_lines(content)
    content = MULTI_SPACE.sub(" ", content)
    content = MULTI_BLANK.sub("\n\n", content)
    content = normalize_text(content, language)
    return content.strip()


def cleaning_quality_penalty(content: str) -> float:
    """Return 0..0.4 penalty for residual menu/boilerplate density."""
    if not content:
        return 0.4
    penalty = 0.0
    if is_menu_heavy_text(content):
        penalty += 0.25
    if content.count(" | ") >= 6:
        penalty += 0.1
    if len(content) < 120:
        penalty += 0.1
    return min(0.4, penalty)
