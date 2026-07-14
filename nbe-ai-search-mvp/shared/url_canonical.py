"""Canonical URL keys for deduplication and merge."""

from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit, urlunsplit


def canonicalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    decoded = unquote(raw)
    parts = urlsplit(decoded)
    path = re.sub(r"/{2,}", "/", parts.path or "")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, parts.fragment))


def canonical_url_key(url: str) -> str:
    return canonicalize_url(url).lower()
