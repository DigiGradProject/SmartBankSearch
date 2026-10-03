"""Purge poisoned entries from the semantic query cache.

Deletes cached responses whose stored citations include an off-topic URL that
saturated the reranker (ExchangeRatesAndCurrencyConverter) while the query is
about a different topic (brand/product queries). See
docs/root-cause-exchange-rate-citations.md.

Usage:
    .venv/bin/python scripts/purge_semantic_cache.py          # dry-run
    .venv/bin/python scripts/purge_semantic_cache.py --apply  # delete
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")

import chromadb
from chromadb.config import Settings as ChromaSettings

OFFTOPIC_MARKERS = ("ExchangeRatesAndCurrencyConverter",)


def is_poisoned(urls: list[str]) -> tuple[bool, str]:
    """Poisoned = cites an off-topic page that is not the query's own topic."""
    for url in urls:
        if any(marker in url for marker in OFFTOPIC_MARKERS):
            return True, url
    return False, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually delete")
    args = parser.parse_args()

    from shared.config import settings

    client = chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    name = settings.semantic_cache_collection
    try:
        collection = client.get_collection(name)
    except Exception as exc:  # noqa: BLE001
        print(f"collection {name!r} not found: {exc}")
        return 1

    result = collection.get(include=["metadatas"])
    ids = result.get("ids") or []
    metas = result.get("metadatas") or []

    to_delete: list[str] = []
    for cache_id, meta in zip(ids, metas):
        meta = meta or {}
        query = str(meta.get("query", "?"))
        try:
            urls = json.loads(meta.get("urls") or "[]")
        except json.JSONDecodeError:
            urls = []
        poisoned, bad_url = is_poisoned([str(u) for u in urls])
        if poisoned:
            print(f"POISONED  {query[:60]!r} cites {bad_url[:70]}")
            to_delete.append(cache_id)
        else:
            print(f"keep      {query[:60]!r}")

    if not to_delete:
        print("\nNo poisoned entries found.")
        return 0

    print(f"\n{len(to_delete)} poisoned / {len(ids)} total.")
    if not args.apply:
        print("Dry run — re-run with --apply to delete.")
        return 0

    collection.delete(ids=to_delete)
    print(f"Deleted {len(to_delete)} entries from {name!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
