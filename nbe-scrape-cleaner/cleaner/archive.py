"""Read scrape pages directly from a zip archive (no filesystem extraction)."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class RawPage:
    source_path: str
    language: str
    data: dict
    content_txt: str | None = None
    snapshot_html: str | None = None


def _language_from_member(member: str) -> str | None:
    normalized = member.replace("\\", "/")
    if "/AR/pages/" in normalized:
        return "ar"
    if "/EN/pages/" in normalized:
        return "en"
    return None


def iter_page_groups(zip_path: Path) -> Iterator[tuple[str, dict[str, str]]]:
    """Yield (page_folder, {filename: member_path}) for each scraped page."""
    with zipfile.ZipFile(zip_path) as archive:
        groups: dict[str, dict[str, str]] = {}
        for member in archive.namelist():
            normalized = member.replace("\\", "/")
            if "/pages/" not in normalized:
                continue
            parts = normalized.split("/")
            try:
                pages_idx = parts.index("pages")
            except ValueError:
                continue
            if pages_idx + 1 >= len(parts):
                continue
            page_folder = "/".join(parts[: pages_idx + 2])
            filename = parts[-1]
            if not filename or filename.endswith("/"):
                continue
            groups.setdefault(page_folder, {})[filename] = member

        for page_folder in sorted(groups):
            yield page_folder, groups[page_folder]


def load_pages(zip_path: Path) -> tuple[list[RawPage], list[tuple[str, str]]]:
    pages: list[RawPage] = []
    failures: list[tuple[str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        groups: dict[str, dict[str, str]] = {}
        for member in archive.namelist():
            normalized = member.replace("\\", "/")
            if "/pages/" not in normalized:
                continue
            parts = normalized.split("/")
            try:
                pages_idx = parts.index("pages")
            except ValueError:
                continue
            if pages_idx + 1 >= len(parts):
                continue
            page_folder = "/".join(parts[: pages_idx + 2])
            filename = parts[-1]
            if not filename or filename.endswith("/"):
                continue
            groups.setdefault(page_folder, {})[filename] = member

        for page_folder in sorted(groups):
            language = _language_from_member(page_folder + "/")
            if language is None:
                continue
            files = groups[page_folder]
            data_member = files.get("data.json")
            if not data_member:
                failures.append((page_folder, "missing data.json"))
                continue
            try:
                data = json.loads(archive.read(data_member).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                failures.append((page_folder, f"failed to parse data.json: {exc}"))
                continue

            content_txt = None
            if "content.txt" in files:
                content_txt = archive.read(files["content.txt"]).decode("utf-8", errors="ignore")

            snapshot_html = None
            if "snapshot.html" in files:
                snapshot_html = archive.read(files["snapshot.html"]).decode("utf-8", errors="ignore")

            pages.append(
                RawPage(
                    source_path=page_folder,
                    language=language,
                    data=data,
                    content_txt=content_txt,
                    snapshot_html=snapshot_html,
                )
            )
    return pages, failures
