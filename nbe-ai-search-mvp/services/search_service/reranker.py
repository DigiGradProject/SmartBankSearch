"""Cross-encoder reranker: BAAI/bge-reranker-v2-m3."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ingestion.embedding.vector_store import RetrievedChunk
from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)


class BGEM3Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model: Any | None = None
        self._backend: str | None = None

    def _load(self) -> Any:
        if self._model is None:
            logger.info("loading_reranker_model", model=self.model_name)
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, max_length=512)
                self._backend = "ce"
            except Exception as exc:  # noqa: BLE001
                logger.warning("cross_encoder_unavailable_fallback_flag", error=str(exc))
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(self.model_name, use_fp16=False)
                self._backend = "flag"
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        limit = top_k or len(chunks)
        model = self._load()
        pairs = [[query, chunk.text[:1800]] for chunk in chunks]

        if self._backend == "flag":
            raw_scores = model.compute_score(pairs, normalize=True)
        else:
            raw_scores = model.predict(pairs)

        if isinstance(raw_scores, (float, int)):
            scores = [float(raw_scores)]
        else:
            scores = [float(score) for score in raw_scores]

        ranked = sorted(
            (replace(chunk, score=max(0.0, min(1.0, score))) for chunk, score in zip(chunks, scores)),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[:limit]


_reranker: BGEM3Reranker | None = None


def get_reranker() -> BGEM3Reranker:
    global _reranker
    if _reranker is None:
        _reranker = BGEM3Reranker()
    return _reranker
