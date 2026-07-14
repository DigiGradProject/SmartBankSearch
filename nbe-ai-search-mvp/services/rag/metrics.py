"""Prometheus metric handles shared by API and orchestrator."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

RETRIEVE_LATENCY = Histogram("nbe_retrieve_latency_seconds", "Hybrid retrieval latency")
RERANK_LATENCY = Histogram("nbe_rerank_latency_seconds", "Reranker latency")
LLM_LATENCY = Histogram("nbe_llm_latency_seconds", "LLM generation latency")
SEMANTIC_CACHE_HITS = Counter("nbe_semantic_cache_hits_total", "Semantic cache hits")
FEEDBACK_TOTAL = Counter("nbe_feedback_total", "User feedback votes", ["vote"])
