"""Validate banking synonym ontology JSON files."""

from __future__ import annotations

import json
from pathlib import Path


def validate_synonym_file(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing file: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid json in {path.name}: {exc}"]

    if not isinstance(payload, dict) or not payload:
        errors.append(f"{path.name}: root must be a non-empty object")
        return errors

    for key, values in payload.items():
        if not isinstance(key, str) or not key.strip():
            errors.append(f"{path.name}: empty key")
            continue
        if not isinstance(values, list) or not values:
            errors.append(f"{path.name}: '{key}' must be a non-empty list")
            continue
        seen: set[str] = set()
        for item in values:
            if not isinstance(item, str) or not item.strip():
                errors.append(f"{path.name}: '{key}' has empty synonym")
            elif item in seen:
                errors.append(f"{path.name}: '{key}' duplicate synonym '{item}'")
            else:
                seen.add(item)
    return errors


def validate_vocabulary(project_root: Path) -> list[str]:
    vocab = project_root / "data" / "vocabulary"
    errors: list[str] = []
    for name in ("banking_synonyms.ar.json", "banking_synonyms.en.json"):
        errors.extend(validate_synonym_file(vocab / name))
    return errors
