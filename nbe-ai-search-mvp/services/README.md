# `services/` — Module Map

Single source of truth. **Do not add re-export façades** — import from the canonical module directly.

## `services/search_service/` — Retrieval engine

| Module | Role | Import from |
|--------|------|-------------|
| `search.py` | `SearchService.retrieve()` — hybrid + rerank + modes | `services.search_service.search` |
| `hybrid_retriever.py` | BM25 + BGE-M3 + RRF | `services.search_service.hybrid_retriever` |
| `reranker.py` | BGE cross-encoder reranker | `services.search_service.reranker` |
| `intent_classifier.py` | Regex rules + `QueryIntent` type | `services.search_service.intent_classifier` |
| `intent_boost.py` | Additive score boosts | `services.search_service.intent_boost` |
| `keyword_rank.py` | Lexical rerank | `services.search_service.keyword_rank` |
| `card_catalog.py` / `certificate_catalog.py` / … | Product-family shortcuts | respective module |

## `services/rag/` — Query understanding & control plane

| Module | Role | Import from |
|--------|------|-------------|
| `intent_fallback.py` | **Main intent API** — `classify_with_fallback()` | `services.rag.intent_fallback` |
| `semantic_intent.py` | BGE-M3 semantic intent | `services.rag.semantic_intent` |
| `critical_rules.py` | Deterministic banking routes (19623, password) | `services.rag.critical_rules` |
| `query_understanding.py` | Full QU pipeline | `services.rag.query_understanding` |
| `query_planner.py` | Multi-intent split | `services.rag.query_planner` |
| `business_rules.py` | Enterprise soft boost (not inject) | `services.rag.business_rules` |
| `decision_engine.py` | ANSWER / RETRY / NO_ANSWER | `services.rag.decision_engine` |
| `confidence.py` | Calibrated confidence gate | `services.rag.confidence` |
| `metadata_filter.py` | Staged L0–L4 filter policy | `services.rag.metadata_filter` |
| `hybrid_rank.py` | Metadata-aware ranking | `services.rag.hybrid_rank` |
| `language.py` / `query_rewrite.py` / `entity_extractor.py` | QU helpers | respective module |
| `orchestrator-facing` | `analytics`, `audit`, `semantic_cache`, `self_eval`, `explainability`, `response_formatter`, `metrics` | respective module |

## `services/api/` — HTTP layer

| Module | Role |
|--------|------|
| `orchestrator.py` | Production entry: `plan_query` → `SearchService` → LLM |
| `main.py` | FastAPI routes |

## Intent — which import?

```python
# Production / eval — layered semantic + regex fallback
from services.rag.intent_fallback import classify_with_fallback

# Regex-only (tests, legacy debug)
from services.search_service.intent_classifier import classify_query

# Types + filter helper
from services.search_service.intent_classifier import QueryIntent, should_apply_metadata_filter
```

## Removed (dead façades — do not recreate)

- ~~`services/rag/intent_classifier.py`~~ → use `intent_fallback` + `search_service.intent_classifier`
- ~~`services/rag/retriever.py`~~ → use `search_service.hybrid_retriever`
- ~~`services/rag/reranker.py`~~ → use `search_service.reranker`
- ~~`services/rag/pipeline.py`~~ → use `api.orchestrator` + `search_service.search`
- ~~`services/rag/llm.py`~~ → use `services.llm_service.llm`
- ~~`services/rag/prompt_builder.py`~~ → use `services.llm_service.llm`
- ~~`services/rag/metadata.py`~~ → use `services.rag.metadata_filter`
