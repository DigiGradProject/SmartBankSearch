# Enterprise AI Search — Production Hardening

Companion to Phase A control-plane fixes in [`analysis.md`](analysis.md).

This document covers the ten production layers added on top of the existing
hybrid retrieval stack (**BM25 + BGE-M3 + bge-reranker-v2-m3 + Ollama**). Nothing
in that stack was replaced.

## Architecture delta

```
User Query (unchanged)
  → QueryUnderstanding (language → intent → entities → expand → rewrite)
  → L1 exact cache / Semantic cache (Chroma, cosine ≥ 0.95)
  → QueryPlanner (≤2 intents on conjunctions)
  → Hybrid retrieve + rerank (existing)
  → Context compression
  → Ollama answer
  → Self-eval (SUPPORTED / PARTIAL / UNSUPPORTED → optional regen)
  → Calibrated confidence + ranked citations
  → Analytics JSONL + Prometheus
```

## Folder map

| Path | Role |
|------|------|
| `services/rag/query_understanding.py` | Unified QU façade |
| `services/rag/query_planner.py` | Multi-intent split/merge |
| `services/rag/semantic_cache.py` | Chroma similarity cache |
| `services/rag/self_eval.py` | Faithfulness gate |
| `services/rag/confidence.py` | Calibrated score + reason |
| `services/rag/analytics.py` | `data/logs/retrieval_analytics.jsonl` |
| `services/rag/explainability.py` | Debug payload |
| `services/rag/metrics.py` | Prometheus stage metrics |
| `services/context_builder/compressor.py` | Context compression |
| `services/feedback/` | Thumbs feedback JSONL |

## Additive API (backward compatible)

### `POST /v1/search`

Request additions:

- `debug: bool = false`

Response additions (all optional for old clients):

- `confidence_reason`
- `cache_hit`
- `rewritten_query`
- `entities`
- `faithfulness`
- `explain` (only when `debug=true`)
- Citation: `category`, `relevance_score`, `reranker_score`

### `POST /v1/feedback`

```json
{
  "query_hash": "…",
  "vote": "helpful|not_helpful",
  "reason": null,
  "question": "…",
  "answer": "…",
  "docs": [],
  "confidence": 0.9
}
```

Stores `data/logs/feedback.jsonl` for future reranker fine-tuning.

## Feature flags (`shared/config.py` / `.env`)

| Flag | Default | Purpose |
|------|---------|---------|
| `semantic_cache_enabled` | true | Chroma query cache |
| `semantic_cache_threshold` | 0.95 | Cosine similarity hit |
| `self_eval_enabled` | true | Post-answer faithfulness |
| `context_compression_enabled` | true | Compress LLM context |
| `query_planner_enabled` | true | Multi-intent fan-out |
| `analytics_log_enabled` | true | JSONL analytics |

## Per-feature notes

### 1. Query Understanding

Original query is never mutated. Retrieval embeds `search_query` only. Entities cover product, certificate, loan, currency, card, branch, location (+ tenors).

### 2. Context compression

Drops near-duplicates, nav/boilerplate, extracts query-overlapping paragraphs, merges same-URL overlaps, respects `context_max_tokens`.

### 3. Confidence calibration

```
C ≈ 0.15·intent + 0.25·top1 + 0.15·mean(top5)
  + 0.20·category_consistency + 0.15·chunk_agreement + 0.10·source_diversity
```

Human-readable `confidence_reason` returned to clients.

### 4. Semantic cache

Collection `nbe_semantic_cache` (separate from document index). Never caches abstentions / low confidence. TTL via metadata.

### 5. Query planner

Splits on `و` / `and` / `ثم` when sub-parts resolve to **distinct** intents (max 2). Contexts merged by URL score before LLM.

### 6. Self evaluation

Ollama checks faithfulness. `UNSUPPORTED` → one strict regenerate. On failure/timeout → `PARTIALLY_SUPPORTED` passthrough.

### 7. Citation ranking

Citations ranked by reranker/relevance; include category + scores (e.g. display as percentages in UI).

### 8. Retrieval analytics

- JSONL: `data/logs/retrieval_analytics.jsonl`
- Prometheus: `/metrics` (`nbe_retrieve_*`, `nbe_rerank_*`, `nbe_llm_*`, `nbe_semantic_cache_hits_total`, `nbe_feedback_total`)
- Grafana: import `observability/dashboards/search-overview.json`

### 9. Feedback loop

UI thumbs → `/v1/feedback` → JSONL dataset.

### 10. Explainability

`debug=true` returns intent, entities, rewrite, confidence reason, retrieved docs + why selected. **Do not enable in public prod UI.**

## Performance impact

| Layer | Typical cost |
|-------|----------------|
| QU (rules) | +5–20 ms |
| Compression | +10–30 ms, −30–60% tokens |
| Semantic cache hit | skips retrieve/LLM |
| Planner (2 intents) | ~1.6–2× retrieve |
| Self-eval | +1 LLM call when enabled |

Disable `self_eval_enabled` / `semantic_cache_enabled` under load if needed.

## Security

- `debug` explain leaks ranking internals — keep default false; ACL at edge.
- Feedback/ingest APIs have no app-level auth (same as MVP admin ingest) — protect at network edge.
- Semantic cache isolated from document collection; TTL eviction.

## Tests

```bash
.\.venv\Scripts\python.exe -m pytest tests/unit tests/integration -q
.\.venv\Scripts\python.exe scripts/eval_retrieval.py --gate --report
```

## Ops smoke

```bash
uvicorn services.api.main:app --reload --host 127.0.0.1 --port 7000
curl -X POST http://127.0.0.1:7000/v1/search -H "Content-Type: application/json" -d "{\"query\":\"سعر الصرف\",\"language\":\"ar\",\"debug\":true}"
```
