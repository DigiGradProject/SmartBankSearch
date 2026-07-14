import re
import unicodedata

DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ELONGATION = re.compile(r"\u0640+")
WHITESPACE = re.compile(r"\s+")

ALEF_VARIANTS = str.maketrans("أإآٱ", "اااا")
YEH_VARIANTS = str.maketrans("ى", "ي")
TA_MARBUTA = str.maketrans("ة", "ه")


def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = DIACRITICS.sub("", text)
    text = text.translate(ALEF_VARIANTS)
    text = text.translate(YEH_VARIANTS)
    text = text.translate(TA_MARBUTA)
    text = ELONGATION.sub("", text)
    text = WHITESPACE.sub(" ", text).strip()
    return text


def normalize_text(text: str, language: str) -> str:
    cleaned = WHITESPACE.sub(" ", text).strip()
    if language == "ar":
        return normalize_arabic(cleaned)
    return cleaned
