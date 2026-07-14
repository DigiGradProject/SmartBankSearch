#!/usr/bin/env python3
"""
Clean NBE website scrape into RAG-ready JSONL documents.

Reads pages directly from nbe_complete_scrape.zip (no filesystem extraction)
and writes one JSON object per line to documents.jsonl.

JSONL is preferred over one-file-per-page because:
  - avoids Windows path-length issues from long ProductDetails folder names
  - is stream-friendly for downstream ingestion
  - remains a single portable artifact
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from cleaner.archive import load_pages
from cleaner.boilerplate import build_boilerplate_set
from cleaner.build import build_content
from cleaner.doc_classifier import classify_document
from cleaner.duplicates import find_near_duplicates
from cleaner.ids import canonicalize_url, content_hash, document_id
from cleaner.report import write_report
from cleaner.title import extract_title

ROOT = Path(__file__).resolve().parent


def _raw_size(page) -> int:
    blocks = page.data.get("text_blocks") or []
    text_size = sum(len(block.get("text") or "") for block in blocks)
    content_size = len(page.content_txt or "")
    return max(text_size, content_size)


def clean_corpus(
    zip_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    similarity_threshold: float = 0.9,
    boilerplate_fraction: float = 0.55,
) -> dict:
    print(f"[1/5] Loading pages from zip: {zip_path}")
    pages, load_failures = load_pages(zip_path)
    print(f"       loaded {len(pages)} pages")

    print("[2/5] Building empirical boilerplate set (per language)")
    boilerplate_by_lang: dict[str, set[str]] = {}
    for language in ("ar", "en"):
        lang_blocks = [
            [block.get("text") or "" for block in (page.data.get("text_blocks") or [])]
            for page in pages
            if page.language == language
        ]
        boilerplate_by_lang[language] = build_boilerplate_set(
            lang_blocks,
            min_page_fraction=boilerplate_fraction,
            min_absolute=max(20, int(len(lang_blocks) * 0.35)),
        )
        print(f"       {language}: {len(boilerplate_by_lang[language])} boilerplate lines")

    print("[3/5] Cleaning pages")
    documents: list[dict] = []
    failures: list[tuple[str, str]] = list(load_failures)
    raw_sizes: list[int] = []
    clean_sizes: list[int] = []
    tables_count = 0
    # Stable timestamp from zip mtime => identical reruns on same archive.
    zip_mtime = datetime.fromtimestamp(zip_path.stat().st_mtime, tz=timezone.utc)
    extracted_at = zip_mtime.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    for index, page in enumerate(pages, start=1):
        source = page.source_path
        try:
            url = canonicalize_url(page.data.get("url") or "")
            if not url:
                raise ValueError("missing url")

            html = page.data.get("html") or page.snapshot_html or ""
            text_blocks = page.data.get("text_blocks") or []
            tables = page.data.get("tables") or []
            title = extract_title(html, text_blocks, tables)
            content, has_tables = build_content(
                text_blocks,
                tables,
                boilerplate_by_lang.get(page.language, set()),
            )
            if has_tables:
                tables_count += 1

            classification = classify_document(url, title, content)
            doc = {
                "id": document_id(url),
                "title": title,
                "url": url,
                "language": page.language,
                "content": content,
                "metadata": {
                    "source_path": source,
                    "content_hash": content_hash(content),
                    "extracted_at": extracted_at,
                    "has_tables": has_tables,
                    "duplicate_of": None,
                    "doc_type": classification.doc_type,
                    "category": classification.category,
                    "canonical_url_slug": classification.canonical_url_slug,
                },
            }
            documents.append(doc)
            raw_sizes.append(_raw_size(page))
            clean_sizes.append(len(content))
        except Exception as exc:  # noqa: BLE001
            failures.append((source, str(exc)))
            print(f"       ! failed {source}: {exc}", file=sys.stderr)

        if index % 50 == 0 or index == len(pages):
            print(f"       progress {index}/{len(pages)}")

    print("[4/5] Detecting near-duplicates")
    duplicate_map, pairs = find_near_duplicates(documents, threshold=similarity_threshold)
    for doc in documents:
        doc["metadata"]["duplicate_of"] = duplicate_map.get(doc["id"])

    # Stable ordering for idempotent output
    documents.sort(key=lambda item: (item["language"], item["url"], item["id"]))

    print(f"[5/5] Writing corpus -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for doc in documents:
            handle.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")

    total_ar = sum(1 for doc in documents if doc["language"] == "ar")
    total_en = sum(1 for doc in documents if doc["language"] == "en")
    avg_raw = sum(raw_sizes) / len(raw_sizes) if raw_sizes else 0.0
    avg_clean = sum(clean_sizes) / len(clean_sizes) if clean_sizes else 0.0

    report_text = write_report(
        report_path,
        total_ar=total_ar,
        total_en=total_en,
        avg_raw=avg_raw,
        avg_clean=avg_clean,
        tables_count=tables_count,
        pairs=pairs,
        failures=failures,
        output_path=output_path,
        thin_pages=sum(1 for doc in documents if len(doc["content"]) < 120),
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(report_text)
    return {
        "documents": len(documents),
        "failures": len(failures),
        "pairs": len(pairs),
        "tables": tables_count,
        "avg_raw": avg_raw,
        "avg_clean": avg_clean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean NBE scrape zip into RAG-ready JSONL")
    parser.add_argument(
        "--zip",
        type=Path,
        default=ROOT.parent / "nbe_complete_scrape.zip",
        help="Path to nbe_complete_scrape.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "documents.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "report.md",
        help="Markdown report path",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument("--boilerplate-fraction", type=float, default=0.55)
    args = parser.parse_args()

    if not args.zip.exists():
        raise SystemExit(f"Zip not found: {args.zip}")

    clean_corpus(
        args.zip,
        args.output,
        args.report,
        similarity_threshold=args.similarity_threshold,
        boilerplate_fraction=args.boilerplate_fraction,
    )


if __name__ == "__main__":
    main()
