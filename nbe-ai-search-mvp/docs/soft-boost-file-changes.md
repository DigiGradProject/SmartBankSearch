# File-by-File Modifications (Soft-Boost Refactor)

Incremental refactor — stack unchanged (FastAPI, React, BGE-M3, BM25, RRF, BGE-Reranker-v2-m3, Ollama, query rewrite, intent, confidence gate).

## Removed dead façades (architecture cleanup)

Deleted unused re-export stubs — import canonical modules directly (see `services/README.md`):

- ~~`services/rag/intent_classifier.py`~~
- ~~`services/rag/retriever.py`~~
- ~~`services/rag/reranker.py`~~
- ~~`services/rag/pipeline.py`~~
- ~~`services/rag/llm.py`~~
- ~~`services/rag/prompt_builder.py`~~
- ~~`services/rag/metadata.py`~~

| File | Role |
|------|------|
| `shared/retrieval_mode.py` | `RetrievalMode` enum + helpers |
| `services/rag/business_rules.py` | Soft multiplicative enterprise scoring |
| `scripts/eval_metrics.py` | Recall@5, MRR, nDCG, aggregation |
| `scripts/eval_retrieval_modes.py` | Dual-mode evaluation harness |
| `scripts/generate_validation_set.py` | Build paraphrase hold-out |
| `tests/retrieval/validation_set.jsonl` | 220 paraphrased AR/EN cases |
| `tests/unit/test_soft_boost_retrieval.py` | Soft boost + mode unit tests |
| `tests/unit/test_eval_metrics.py` | Metric helpers |
| `tests/integration/test_retrieval_modes.py` | Mode integration tests |
| `docs/soft-boost-architecture.md` | Architecture + principle |
| `docs/migration-soft-boost.md` | Migration / rollback |
| `docs/mode-performance-comparison.md` | How to compare modes |
| `scripts/__init__.py` | Package marker for eval imports |

## Modified files

| File | Change |
|------|--------|
| `shared/config.py` | `retrieval_mode`, `business_rules_enabled`; deprecate `force_canonical_inject` |
| `services/rag/metadata_filter.py` | `PREFERRED_CATEGORIES`, soft URL markers API |
| `services/rag/hybrid_rank.py` | Preferred-category soft weights |
| `services/rag/decision_engine.py` | No inject/pin; `FORCE_CANONICAL` deprecated unused |
| `services/search_service/search.py` | Soft boost path; mode wiring; remove `_force_inject_canonical` |
| `scripts/eval_retrieval.py` | `--mode` flag; mode-aware retrieve |
| `tests/unit/test_phase_a_control_plane.py` | Expect `RETRY_RELATED` instead of force pin |

## Unchanged (preserved)

- Intent classifier (`intent_classifier.py`)
- Hybrid RRF retriever, embedder, BM25, reranker
- Confidence calibration formula
- FastAPI routes / `SearchRequest` / `SearchResponse`
- Frontend React app
- `golden_set.jsonl` (reference / training)
