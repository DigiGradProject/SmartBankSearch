"""Metadata helpers for filtering and enrichment (Phase 1 scaffold)."""

from __future__ import annotations

from shared.document_quality import is_official_product_category_url


def build_doc_type_filter(doc_types: list[str] | None) -> dict | None:
    if not doc_types:
        return None
    return {"doc_type": {"$in": list(doc_types)}}


def is_category_landing(url: str) -> bool:
    return is_official_product_category_url(url)
