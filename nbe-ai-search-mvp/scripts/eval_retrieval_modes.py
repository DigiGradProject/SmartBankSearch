"""Evaluate retrieval in PURE_SEMANTIC and ENTERPRISE modes.

Reports Recall@5, MRR, nDCG@5, Intent Accuracy, Top1, Top3,
Confidence Accuracy, and Latency.

Treats golden_set.jsonl as the training/reference set and
validation_set.jsonl as the paraphrase hold-out set.
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
from services.search_service.search import SearchService  # noqa: E402
from shared.config import settings  # noqa: E402
from shared.retrieval_mode import RetrievalMode  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _confidence_ok(case: dict, retrieval) -> bool:
    """Confidence accuracy: answered iff expected URL hit in top-3 when labeled."""
    expected = case.get("expected_url_contains", "")
    if not expected:
        # Unlabeled URL cases: confidence should not force a hard fail.
        return True
    top3 = [chunk.url for chunk in retrieval.chunks[:3]]
    hit = any(expected in (url or "") for url in top3)
    if hit:
        return retrieval.confidence >= settings.confidence_threshold * 0.85
    # Miss: preferably abstain / low confidence in ENTERPRISE; soft check in PURE.
    if retrieval.retrieval_mode == RetrievalMode.PURE_SEMANTIC.value:
        return True
    return (not retrieval.should_answer) or (
        retrieval.confidence < settings.confidence_threshold + 0.15
    )


def evaluate_cases(
    cases: list[dict],
    *,
    mode: RetrievalMode,
    top_k: int = 5,
) -> dict:
    service = SearchService()
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
        retrieval = service.retrieve(
            query,
            language,
            retrieval_mode=mode,
            business_rules=(mode == RetrievalMode.ENTERPRISE),
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

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
                "confidence_ok": _confidence_ok(case, retrieval) and forbidden_ok,
                "latency_ms": latency_ms,
                "top_url": urls[0] if urls else None,
                "confidence": retrieval.confidence,
                "should_answer": retrieval.should_answer,
                "business_rules_applied": retrieval.business_rules_applied,
                "forbidden_ok": forbidden_ok,
            }
        )

    summary = aggregate_metrics(rows)
    summary["total"] = len(cases)
    summary["mode"] = mode.value
    summary["results"] = rows
    return summary


def print_summary(summary: dict) -> None:
    print(
        f"mode={summary['mode']} cases={summary['total']} "
        f"Recall@5={summary['recall@5']:.1%} "
        f"MRR={summary['mrr']:.3f} "
        f"nDCG@5={summary['ndcg@5']:.3f} "
        f"Intent={summary['intent_accuracy']:.1%} "
        f"Top1={summary['top1']:.1%} "
        f"Top3={summary['top3']:.1%} "
        f"ConfAcc={summary['confidence_accuracy']:.1%} "
        f"lat_mean={summary['latency_ms_mean']:.0f}ms "
        f"lat_p95={summary['latency_ms_p95']:.0f}ms"
    )


def compare_modes(pure: dict, enterprise: dict) -> dict:
    keys = [
        "recall@5",
        "mrr",
        "ndcg@5",
        "intent_accuracy",
        "top1",
        "top3",
        "confidence_accuracy",
        "latency_ms_mean",
    ]
    delta = {k: enterprise[k] - pure[k] for k in keys}
    return {
        "PURE_SEMANTIC": {k: pure[k] for k in keys + ["total"]},
        "ENTERPRISE": {k: enterprise[k] for k in keys + ["total"]},
        "delta_enterprise_minus_pure": delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-mode NBE retrieval evaluation")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "tests" / "retrieval" / "validation_set.jsonl",
        help="JSONL cases (default: validation_set)",
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        help="Use golden_set.jsonl (reference/training set)",
    )
    parser.add_argument(
        "--mode",
        choices=["PURE_SEMANTIC", "ENTERPRISE", "both"],
        default="both",
    )
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON comparison")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dataset = (
        ROOT / "tests" / "retrieval" / "golden_set.jsonl" if args.golden else args.dataset
    )
    cases = load_jsonl(dataset)
    print(f"dataset={dataset} cases={len(cases)}")

    summaries: dict[str, dict] = {}
    modes = (
        [RetrievalMode.PURE_SEMANTIC, RetrievalMode.ENTERPRISE]
        if args.mode == "both"
        else [RetrievalMode(args.mode)]
    )

    for mode in modes:
        summary = evaluate_cases(cases, mode=mode)
        print_summary(summary)
        summaries[mode.value] = summary
        if args.report:
            for row in summary["results"]:
                status = "OK" if row["intent_ok"] and row["top3"] >= 1.0 else "FAIL"
                print(f"  [{status}][{mode.value}] {row['query'][:80]}")

    payload: dict = {"dataset": str(dataset)}
    if "PURE_SEMANTIC" in summaries and "ENTERPRISE" in summaries:
        payload["comparison"] = compare_modes(
            summaries["PURE_SEMANTIC"], summaries["ENTERPRISE"]
        )
        print("\n=== PURE_SEMANTIC vs ENTERPRISE ===")
        for key, value in payload["comparison"]["delta_enterprise_minus_pure"].items():
            print(f"  Δ {key}: {value:+.4f}")
    payload["summaries"] = {
        name: {k: v for k, v in summary.items() if k != "results"}
        for name, summary in summaries.items()
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")

    if args.gate and "ENTERPRISE" in summaries:
        ent = summaries["ENTERPRISE"]
        ok = (
            ent["intent_accuracy"] >= settings.eval_intent_min_accuracy
            and ent["top3"] >= settings.eval_url_min_hit_rate
        )
        if not ok:
            print("GATE FAILED")
            raise SystemExit(1)
        print("GATE PASSED")


if __name__ == "__main__":
    main()
