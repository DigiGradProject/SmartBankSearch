# NBE Enterprise AI Search — Implementation Plan (Phases 1–4)

**Status:** **Phases 1–4 implemented in code** (2026-07-14)  
**Remaining runtime dependency:** finish/verify `nbe_chunks_bge_m3_v4` ingest + golden `--gate` on GPU host.

**Companion docs:**  
- Model decisions: [`enterprise-model-decisions.md`](./enterprise-model-decisions.md)  
- Retrieval redesign: [`plan-retrieval-redesign.md`](./plan-retrieval-redesign.md)  
- CPU profile: [`cpu-profile.md`](./cpu-profile.md)  
- Licenses: [`licenses.md`](./licenses.md)  
- Security: [`security-notes.md`](./security-notes.md)

---

## Phase 1 — Deterministic query understanding ✅

| Deliverable | Status |
|-------------|--------|
| Lingua language detection | ✅ `services/rag/language.py` |
| Rule intent taxonomy | ✅ expanded intents |
| Entity dictionary extractor | ✅ + optional GLiNER flag |
| Synonym ontology validation | ✅ `synonym_ontology.py` + schema |
| Rule query rewrite | ✅ wired into `SearchService` |
| Latency smoke tests | ✅ |

---

## Phase 2 — Retrieval quality ✅

| Deliverable | Status |
|-------------|--------|
| Cleaner + parent/child chunks | ✅ |
| Metadata enrichment | ✅ |
| Hybrid Top20 → rerank Top5 | ✅ config `rerank_pool_size/keep` |
| Account/card/cert prioritizers | ✅ |
| Golden eval gate | ✅ `scripts/eval_retrieval.py --gate` |

---

## Phase 3 — Grounded generation & structure ✅

| Deliverable | Status |
|-------------|--------|
| Enterprise RAG prompt + NO_ANSWER | ✅ |
| Pydantic structured formatter | ✅ `structured` field on response |
| Composite confidence | ✅ |
| Extractive bypass catalogs | ✅ |
| MiniLM intent fallback | ✅ feature-flagged (default off) |

---

## Phase 4 — Hardening ✅

| Deliverable | Status |
|-------------|--------|
| Audit logging | ✅ `services/rag/audit.py` |
| Optional GLiNER | ✅ `GLINER_ENABLED` |
| Optional LLM rewrite | ✅ `LLM_REWRITE_ENABLED` |
| CPU profile docs | ✅ |
| License pack | ✅ |
| Security notes | ✅ |
| Prometheus metrics | ✅ existing `/metrics` |

---

## Go-live checklist

- [ ] v4 ingest completed  
- [ ] `python scripts/eval_retrieval.py --gate` green  
- [ ] Confidence abstention tested  
- [ ] Citations present when answered  
- [ ] Legal license memo signed  
- [ ] No outbound network from inference hosts  

---

## Feature flags

```
MINILM_INTENT_FALLBACK_ENABLED=false
GLINER_ENABLED=false
LLM_REWRITE_ENABLED=false
AUDIT_LOG_ENABLED=true
RERANK_POOL_SIZE=20
RERANK_KEEP_SIZE=5
```

---

*Last updated: 2026-07-14*
