"""Evaluate retrieval quality against golden query set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.search_service.intent_classifier import classify_query  # noqa: E402
from services.search_service.search import SearchService  # noqa: E402


def load_golden(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(golden_path: Path, top_k: int = 3) -> dict:
    service = SearchService()
    cases = load_golden(golden_path)
    intent_hits = 0
    url_hits = 0
    forbidden_misses = 0
    results: list[dict] = []

    for case in cases:
        query = case["query"]
        language = case["language"]
        intent = classify_query(query, language)
        intent_ok = intent.intent == case["intent"]
        if intent_ok:
            intent_hits += 1

        retrieval = service.retrieve(query, language)
        top_urls = [chunk.url for chunk in retrieval.chunks[:top_k]]
        expected = case.get("expected_url_contains", "")
        forbidden = case.get("forbidden_url_contains", "")

        url_ok = any(expected in url for url in top_urls) if expected else True
        if url_ok:
            url_hits += 1

        forbidden_ok = not any(forbidden in url for url in top_urls) if forbidden else True
        if forbidden_ok:
            forbidden_misses += 1

        results.append(
            {
                "query": query,
                "expected_intent": case["intent"],
                "actual_intent": intent.intent,
                "intent_ok": intent_ok,
                "url_ok": url_ok,
                "forbidden_ok": forbidden_ok,
                "top_url": top_urls[0] if top_urls else None,
                "filter_applied": retrieval.filter_applied,
            }
        )

    total = len(cases)
    return {
        "total": total,
        "intent_accuracy": intent_hits / total if total else 0.0,
        "top3_url_hit_rate": url_hits / total if total else 0.0,
        "forbidden_avoid_rate": forbidden_misses / total if total else 0.0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NBE retrieval golden set")
    parser.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "tests" / "retrieval" / "golden_set.jsonl",
    )
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    summary = evaluate(args.golden)
    print(
        f"cases={summary['total']} "
        f"intent_acc={summary['intent_accuracy']:.1%} "
        f"top3_url={summary['top3_url_hit_rate']:.1%} "
        f"forbidden_avoid={summary['forbidden_avoid_rate']:.1%}"
    )
    if args.report:
        for row in summary["results"]:
            status = "OK" if row["intent_ok"] and row["url_ok"] and row["forbidden_ok"] else "FAIL"
            print(f"[{status}] {row['query']}")
            if status == "FAIL":
                print(f"       intent: {row['actual_intent']} (expected {row['expected_intent']})")
                print(f"       top: {row['top_url']}")


if __name__ == "__main__":
    main()
