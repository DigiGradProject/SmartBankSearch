"""BGE-M3 dense + sparse (hybrid) embeddings."""

from __future__ import annotations

from typing import Any

from shared.config import settings
from shared.logging import get_logger

logger = get_logger(__name__)


class BGEM3Embedder:
    """Dense vectors for Chroma + sparse lexical weights for hybrid scoring."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            logger.info("loading_embedding_model", model=self.model_name)
            try:
                from FlagEmbedding import BGEM3FlagModel

                self._model = BGEM3FlagModel(self.model_name, use_fp16=False)
                self._backend = "flag"
            except Exception as exc:  # noqa: BLE001
                logger.warning("flagembedding_unavailable_fallback_st", error=str(exc))
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                self._backend = "st"
        return self._model

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        if getattr(self, "_backend", "st") == "flag":
            output = model.encode(
                texts,
                batch_size=8,
                max_length=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vectors = output["dense_vecs"]
            return [vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in vectors]

        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_hybrid(self, texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
        model = self._load()
        if getattr(self, "_backend", "st") == "flag":
            output = model.encode(
                texts,
                batch_size=8,
                max_length=512,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense = [
                vector.tolist() if hasattr(vector, "tolist") else list(vector)
                for vector in output["dense_vecs"]
            ]
            sparse = [_normalize_lexical(weights) for weights in output["lexical_weights"]]
            return dense, sparse

        dense = self.embed_dense(texts)
        sparse = [{} for _ in texts]
        return dense, sparse


def _normalize_lexical(weights: dict) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in (weights or {}).items():
        token = str(key)
        try:
            normalized[token] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def sparse_score(query_weights: dict[str, float], doc_weights: dict[str, float]) -> float:
    if not query_weights or not doc_weights:
        return 0.0
    score = 0.0
    for token, weight in query_weights.items():
        if token in doc_weights:
            score += weight * doc_weights[token]
    return float(score)


def sparse_score_against_text(query_weights: dict[str, float], text: str) -> float:
    """Fallback sparse score when document lexical weights are unavailable."""
    if not query_weights or not text:
        return 0.0
    haystack = text.lower()
    score = 0.0
    for token, weight in query_weights.items():
        if token.lower() in haystack:
            score += float(weight)
    return float(score)


_embedder: BGEM3Embedder | None = None


def get_embedder() -> BGEM3Embedder:
    global _embedder
    if _embedder is None:
        _embedder = BGEM3Embedder()
    return _embedder
