"""Near-duplicate detection with human-review flagging (no silent deletes)."""

from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass
class DuplicatePair:
    left_id: str
    right_id: str
    left_url: str
    right_url: str
    similarity: float
    duplicate_id: str
    canonical_id: str


def url_canonicity_score(url: str) -> float:
    """Higher is more canonical / preferable as the primary source."""
    score = 0.0
    lower = (url or "").lower()
    score -= min(len(url), 500) / 500.0
    score -= lower.count("%") * 0.15
    score -= lower.count(" ") * 0.2
    if "productdetails" in lower:
        score -= 0.4
    if "productcategory" in lower:
        score += 0.3
    if "newcontent" in lower or "undefined" in lower:
        score -= 1.0
    if "typo" in lower or "old" in lower or "legacy" in lower:
        score -= 0.5
    # Prefer shorter fragments after #/
    if "#/" in url:
        fragment = url.split("#/", 1)[1]
        score -= fragment.count("/") * 0.05
        score -= fragment.count("{") * 0.2
    return score


def content_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    # Use a bounded prefix for speed while remaining stable.
    left = a[:12000]
    right = b[:12000]
    return difflib.SequenceMatcher(None, left, right).ratio()


def lengths_comparable(a: str, b: str, max_ratio: float = 1.35) -> bool:
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return False
    ratio = max(la, lb) / max(1, min(la, lb))
    return ratio <= max_ratio


def find_near_duplicates(
    documents: list[dict],
    threshold: float = 0.9,
    min_content_chars: int = 120,
) -> tuple[dict[str, str | None], list[DuplicatePair]]:
    """
    Returns:
      - mapping doc_id -> duplicate_of (or None)
      - list of flagged pairs for human review

    Very short/empty SPA shells are excluded from pairwise matching so they
    do not create a combinatorial explosion of trivial 1.0 matches.
    """
    duplicate_of: dict[str, str | None] = {doc["id"]: None for doc in documents}
    pairs: list[DuplicatePair] = []

    by_lang: dict[str, list[dict]] = {"ar": [], "en": []}
    for doc in documents:
        if len(doc.get("content") or "") < min_content_chars:
            continue
        by_lang.setdefault(doc["language"], []).append(doc)

    for _language, docs in by_lang.items():
        n = len(docs)
        for i in range(n):
            for j in range(i + 1, n):
                left, right = docs[i], docs[j]
                if not lengths_comparable(left["content"], right["content"]):
                    continue
                similarity = content_similarity(left["content"], right["content"])
                if similarity < threshold:
                    continue

                left_score = url_canonicity_score(left["url"])
                right_score = url_canonicity_score(right["url"])
                if left_score >= right_score:
                    canonical, duplicate = left, right
                else:
                    canonical, duplicate = right, left

                if duplicate_of[duplicate["id"]] is None:
                    duplicate_of[duplicate["id"]] = canonical["id"]

                pairs.append(
                    DuplicatePair(
                        left_id=left["id"],
                        right_id=right["id"],
                        left_url=left["url"],
                        right_url=right["url"],
                        similarity=round(similarity, 4),
                        duplicate_id=duplicate["id"],
                        canonical_id=canonical["id"],
                    )
                )

    return duplicate_of, pairs


def pair_urls(pair: DuplicatePair) -> tuple[str, str]:
    if pair.canonical_id == pair.left_id:
        return pair.left_url, pair.right_url
    return pair.right_url, pair.left_url
