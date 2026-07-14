"""Post-answer faithfulness self-evaluation via Ollama."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import httpx

from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)


class FaithfulnessLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class SelfEvalResult:
    label: FaithfulnessLabel
    raw: str
    should_regenerate: bool


_LABEL_RE = re.compile(
    r"\b(SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED)\b",
    re.I,
)


def parse_faithfulness_label(text: str) -> FaithfulnessLabel:
    match = _LABEL_RE.search(text or "")
    if not match:
        return FaithfulnessLabel.PARTIALLY_SUPPORTED
    return FaithfulnessLabel(match.group(1).upper())


async def evaluate_answer(
    *,
    query: str,
    answer: str,
    context: str,
    language: str,
) -> SelfEvalResult:
    if not settings.self_eval_enabled:
        return SelfEvalResult(FaithfulnessLabel.SUPPORTED, "disabled", False)

    prompt = (
        "You are a faithfulness checker for a bank AI search engine.\n"
        "Decide if EVERY factual statement in the ANSWER is fully supported by CONTEXT.\n"
        "Reply with ONLY one token: SUPPORTED or PARTIALLY_SUPPORTED or UNSUPPORTED.\n\n"
        f"Language: {language}\n"
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT:\n{context[:4000]}\n\n"
        f"ANSWER:\n{answer[:3000]}\n"
    )
    try:
        async with httpx.AsyncClient(timeout=min(45.0, settings.llm_timeout_seconds)) as client:
            response = await client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
            )
            response.raise_for_status()
            raw = (response.json().get("response") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("self_eval_failed", error=str(exc))
        return SelfEvalResult(FaithfulnessLabel.PARTIALLY_SUPPORTED, str(exc), False)

    label = parse_faithfulness_label(raw)
    logger.info("self_eval_result", label=label.value)
    return SelfEvalResult(
        label=label,
        raw=raw[:200],
        should_regenerate=label == FaithfulnessLabel.UNSUPPORTED,
    )


STRICT_REGEN_HINT = (
    "CRITICAL: Use ONLY facts explicitly present in the retrieved documents. "
    "If a claim is not supported, omit it. Prefer short factual bullets."
)
