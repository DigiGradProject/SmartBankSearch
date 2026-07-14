"""Write cleaning run report."""

from __future__ import annotations

from pathlib import Path

from cleaner.duplicates import DuplicatePair, pair_urls


def write_report(
    path: Path,
    *,
    total_ar: int,
    total_en: int,
    avg_raw: float,
    avg_clean: float,
    tables_count: int,
    pairs: list[DuplicatePair],
    failures: list[tuple[str, str]],
    output_path: Path,
    thin_pages: int = 0,
) -> str:
    reduction = 0.0 if avg_raw <= 0 else (1.0 - (avg_clean / avg_raw)) * 100.0
    lines = [
        "# NBE Scrape Cleaning Report",
        "",
        f"- Output corpus: `{output_path.as_posix()}`",
        f"- Pages processed: **{total_ar + total_en}** (AR={total_ar}, EN={total_en})",
        f"- Average raw text size: **{avg_raw:,.0f}** chars",
        f"- Average cleaned content size: **{avg_clean:,.0f}** chars",
        f"- Size reduction: **{reduction:.1f}%**",
        f"- Pages with tables inlined: **{tables_count}**",
        f"- Thin/SPA shell pages (<120 cleaned chars): **{thin_pages}**",
        f"- Near-duplicate pairs flagged (contentful pages only): **{len(pairs)}**",
        f"- Failed pages: **{len(failures)}**",
        "",
        "## Notes",
        "",
        "- Boilerplate was derived empirically per language from cross-page frequency.",
        "- Base64/image dumps were stripped from all text fields.",
        "- Near-duplicates are flagged via `metadata.duplicate_of`; nothing was deleted.",
        "- Thin pages usually mean the scrape captured chrome only (JS-rendered body missing).",
        "",
        "## Near-duplicate pairs (human review)",
        "",
    ]
    if not pairs:
        lines.append("_None flagged._")
    else:
        lines.append("| Similarity | Canonical URL | Duplicate URL | duplicate_of |")
        lines.append("|---|---|---|---|")
        for pair in sorted(pairs, key=lambda p: p.similarity, reverse=True):
            canonical_url, duplicate_url = pair_urls(pair)
            lines.append(
                f"| {pair.similarity:.3f} | {canonical_url} | {duplicate_url} "
                f"| `{pair.duplicate_id}` -> `{pair.canonical_id}` |"
            )

    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("_None._")
    else:
        for source, reason in failures:
            lines.append(f"- `{source}`: {reason}")

    lines.append("")
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text
