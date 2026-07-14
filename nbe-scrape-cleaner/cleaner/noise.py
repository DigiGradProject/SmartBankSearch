"""Strip embedded base64 / high-entropy binary noise from text."""

from __future__ import annotations

import re

# Long base64-looking tokens (PNG/JPEG dumps, data-URI payloads, etc.)
BASE64_TOKEN = re.compile(
    r"(?:data:image\/[a-zA-Z0-9.+-]+;base64,)?[A-Za-z0-9+/]{100,}={0,2}"
)
PNG_MARKERS = re.compile(r"iVBORw0KGgo[\s\S]{0,20}|AAAAAElFTkSuQmCC")


def is_base64ish_token(token: str, min_length: int = 100) -> bool:
    if len(token) < min_length:
        return False
    if any(ch.isspace() for ch in token):
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    if not set(token) <= allowed:
        return False
    # High alphabet diversity is typical of base64 payloads.
    return len(set(token)) >= 16


def strip_base64_noise(text: str) -> str:
    if not text:
        return ""
    cleaned = BASE64_TOKEN.sub(" ", text)
    cleaned = PNG_MARKERS.sub(" ", cleaned)
    # Also drop remaining space-free ultra-long tokens.
    parts: list[str] = []
    for token in re.split(r"(\s+)", cleaned):
        if token and not token.isspace() and is_base64ish_token(token):
            continue
        parts.append(token)
    cleaned = "".join(parts)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
