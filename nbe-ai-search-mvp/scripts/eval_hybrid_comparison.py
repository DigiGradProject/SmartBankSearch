"""Three-mode retrieval comparison: KEYWORD vs HYBRID vs PURE_SEMANTIC.

Reuses the metrics from scripts/eval_metrics.py and the same JSONL datasets
as eval_retrieval_modes.py. The HYBRID column runs the retrieval-only hybrid
page-search service (dense + BM25 RRF + rerank + title boost, no LLM).

Usage:
    python scripts/eval_hybrid_comparison.py --mode HYBRID
    python scripts/eval_hybrid_comparison.py --mode all --dataset tests/retrieval/validation_set.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_metrics import (  # noqa: E402
    aggregate_metrics,
    mrr,
    ndcg_at_k,
    recall_at_k,
    relevance_labels,
    top_k_hit,
)
from services.rag.intent_fallback import classify_with_fallback  # noqa: E402
from scripts.eval_retrieval_modes import evaluate_cases, load_jsonl, print_summary  # noqa: E402
from services.search_service.hybrid_pages import HybridPageSearchService  # noqa: E402
from shared.retrieval_mode import RetrievalMode  # noqa: E402


class _HybridRetrievalAdapter:
    """Duck-typed retrieval result so shared metric helpers work unchanged."""

    def __init__(self, outcome):
        self.chunks = outcome.results
        self.confidence = 1.0 if outcome.results else 0.0
        self.should_answer = bool(outcome.results)
        self.retrieval_mode = "HYBRID"
        self.business_rules_applied = False


def evaluate_hybrid_cases(cases: list[dict], *, top_k: int = 5) -> dict:
    service = HybridPageSearchService()
    rows: list[dict] = []
    for case in cases:
        query = case["query"]
        language = case.get("language", "auto")
        expected_intent = case.get("intent", "general_faq")
        expected = case.get("expected_url_contains", "")
        forbidden = case.get("forbidden_url_contains", "")

        intent = classify_with_fallback(query, language if language != "auto" else "ar")
        intent_ok = intent.intent == expected_intent

        t0 = time.perf_counter()
        outcome = service.search(query, language)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        retrieval = _HybridRetrievalAdapter(outcome)

        urls = [chunk.url for chunk in retrieval.chunks[: max(top_k, 5)]]
        labels = relevance_labels(urls, expected) if expected else [1] * min(len(urls), 1)

        if expected:
            r5 = recall_at_k(labels, k=5)
            mrr_v = mrr(labels)
            ndcg_v = ndcg_at_k(labels, k=5)
            t1 = top_k_hit(labels, k=1)
            t3 = top_k_hit(labels, k=3)
        else:
            r5 = mrr_v = ndcg_v = t1 = t3 = 1.0

        forbidden_ok = (
            not any(forbidden in (url or "") for url in urls[:3]) if forbidden else True
        )
        rows.append(
            {
                "query": query,
                "expected_intent": expected_intent,
                "actual_intent": intent.intent,
                "intent_ok": intent_ok,
                "recall@5": r5,
                "mrr": mrr_v,
                "ndcg@5": ndcg_v,
                "top1": t1,
                "top3": t3,
                "confidence_ok": forbidden_ok,
                "latency_ms": latency_ms,
                "top_url": urls[0] if urls else None,
                "confidence": retrieval.confidence,
                "should_answer": retrieval.should_answer,
                "business_rules_applied": False,
                "forbidden_ok": forbidden_ok,
            }
        )

    summary = aggregate_metrics(rows)
    summary["total"] = len(cases)
    summary["mode"] = "HYBRID"
    summary["results"] = rows
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="KEYWORD vs HYBRID vs PURE_SEMANTIC comparison")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "tests" / "retrieval" / "validation_set.jsonl",
    )
    parser.add_argument("--golden", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["KEYWORD", "HYBRID", "PURE_SEMANTIC", "all"],
        default="all",
    )
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N cases")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dataset = (
        ROOT / "tests" / "retrieval" / "golden_set.jsonl" if args.golden else args.dataset
    )
    cases = load_jsonl(dataset)
    if args.limit:
        cases = cases[: args.limit]
    print(f"dataset={dataset} cases={len(cases)}")

    from services.search_service.search import SearchService
    from services.search_service.traditional import TraditionalSearchService

    if args.mode in {"KEYWORD", "all"}:
        print_summary(evaluate_cases(cases, mode=RetrievalMode.KEYWORD))
    if args.mode in {"HYBRID", "all"}:
        print_summary(evaluate_hybrid_cases(cases))
    if args.mode in {"PURE_SEMANTIC", "all"}:
        # PURE_SEMANTIC retrieval without LLM generation is expensive per case
        # (embedding + rerank); run it through the same evaluator for parity.
        summary = evaluate_cases(cases, mode=RetrievalMode.PURE_SEMANTIC)
        summary["mode"] = "PURE_SEMANTIC(ret)"
        print_summary(summary)


if __name__ == "__main__":
    main()
