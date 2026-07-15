"""Retrieval evaluation metrics: Recall@k, MRR, nDCG, Top-k hit rates."""

from __future__ import annotations

import math
from typing import Sequence


def url_is_relevant(url: str | None, expected_contains: str) -> bool:
    if not expected_contains:
        return True
    if not url:
        return False
    return expected_contains in url


def relevance_labels(
    ranked_urls: Sequence[str | None],
    expected_contains: str,
    *,
    k: int | None = None,
) -> list[int]:
    urls = list(ranked_urls[:k] if k is not None else ranked_urls)
    return [1 if url_is_relevant(url, expected_contains) else 0 for url in urls]


def recall_at_k(labels: Sequence[int], *, k: int = 5) -> float:
    if not labels or not any(labels):
        # If gold has an expected URL and none retrieved in top-k → 0;
        # if no positive labels because expected blank, treat as 1.0 upstream.
        return 0.0
    window = labels[:k]
    return 1.0 if any(window) else 0.0


def mrr(labels: Sequence[int]) -> float:
    for idx, label in enumerate(labels, start=1):
        if label:
            return 1.0 / idx
    return 0.0


def dcg_at_k(labels: Sequence[int], *, k: int = 5) -> float:
    total = 0.0
    for i, rel in enumerate(labels[:k], start=1):
        if rel:
            total += rel / math.log2(i + 1)
    return total


def ndcg_at_k(labels: Sequence[int], *, k: int = 5) -> float:
    actual = dcg_at_k(labels, k=k)
    if not any(labels):
        return 0.0
    # Ideal DCG: all relevant docs packed at the front (binary or graded).
    ideal_rels = sorted(labels[:k], reverse=True)
    if not any(ideal_rels):
        ideal_rels = [1] + [0] * max(0, k - 1)
    ideal = dcg_at_k(ideal_rels, k=k)
    return min(1.0, actual / ideal) if ideal > 0 else 0.0


def top_k_hit(labels: Sequence[int], *, k: int) -> float:
    return 1.0 if any(labels[:k]) else 0.0


def aggregate_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {
            "recall@5": 0.0,
            "mrr": 0.0,
            "ndcg@5": 0.0,
            "intent_accuracy": 0.0,
            "top1": 0.0,
            "top3": 0.0,
            "confidence_accuracy": 0.0,
            "latency_ms_mean": 0.0,
            "latency_ms_p95": 0.0,
        }

    n = len(rows)
    latencies = sorted(float(r.get("latency_ms", 0.0)) for r in rows)
    p95_idx = min(n - 1, max(0, int(math.ceil(0.95 * n) - 1)))

    return {
        "recall@5": sum(r["recall@5"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
        "ndcg@5": sum(r["ndcg@5"] for r in rows) / n,
        "intent_accuracy": sum(1.0 for r in rows if r.get("intent_ok")) / n,
        "top1": sum(r["top1"] for r in rows) / n,
        "top3": sum(r["top3"] for r in rows) / n,
        "confidence_accuracy": sum(1.0 for r in rows if r.get("confidence_ok")) / n,
        "latency_ms_mean": sum(latencies) / n,
        "latency_ms_p95": latencies[p95_idx],
    }
