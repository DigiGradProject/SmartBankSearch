# NBE Enterprise AI Search — Model & Architecture Decision Report

**Classification:** Internal / Architecture Board  
**Author role:** Principal AI Engineer  
**Environment:** Fully on-premise banking (no cloud LLM APIs)  
**Product type:** Enterprise AI Search (NOT a chatbot)  
**Date:** 2026-07-14  

---

## Executive principle

Prefer **deterministic, lightweight, explainable** components.  
Use an LLM **only** for grounded answer synthesis (and optionally rare colloquial rewrite misses).  
Never use an LLM for language ID, intent (primary), synonyms, or ranking.

---

## TASK 1 — Language Detection

### Winner: **lingua-language-detector (Lingua)**

| Criterion | Decision |
|-----------|----------|
| Best choice | **Lingua** (`lingua-language-detector`) |
| Why | Highest short-query accuracy for AR/EN; banking queries are often 3–12 tokens where `langdetect` fails |
| Arabic | Excellent MSA + dialect-tolerant for script detection |
| Latency | ~1–5 ms after warm load |
| Memory | ~50–150 MB (low languages subset: AR+EN only) |
| GPU | **Not required** |
| CPU-only | **Yes** |
| License | Apache-2.0 |

### Alternatives

| Library | Pros | Cons | Verdict |
|---------|------|------|---------|
| **fastText lid.176** | Fastest (~1 ms), tiny | Needs model file ops; overkill for 2 languages | Runner-up if extreme QPS |
| **CLD3** | Solid | Heavier native dep; ops friction | Skip for bank ops simplicity |
| **langdetect** | Simple | Weak on short text | Reject |

### Policy

1. If UI passes `ar`/`en` → trust it.  
2. Else Lingua with `AR`/`EN` only.  
3. Fallback: Arabic script regex heuristic (already in codebase).

---

## TASK 2 — Intent Classification

### Winner: **Rule-based primary + optional Multilingual MiniLM fallback**

| Criterion | Decision |
|-----------|----------|
| Best choice | **Rules/regex + banking lexicon (primary)** |
| Optional ML | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` **only as soft fallback** when rules confidence &lt; threshold |
| Why ML is not primary | Banks need **explainability**, auditability, zero surprise drift; intents are a closed taxonomy |
| Arabic | Rules + Egyptian→MSA expand already strong; MiniLM adequate as soft backup |
| Latency | Rules: **&lt;1–3 ms**; MiniLM: **5–15 ms CPU** |
| Memory | Rules: negligible; MiniLM: ~400–500 MB |
| GPU | Not required |
| License | MiniLM: Apache-2.0 |

### Alternatives rejected as primary

| Option | Why not primary |
|--------|-----------------|
| ModernBERT / e5 / bge classifiers | Overkill; harder to audit; still need labels |
| Qwen embeddings for intent | Too heavy for a gate |
| FastText supervised | Needs labeled corpus + retrain ops |
| Pure LLM intent | Non-deterministic, slow, costly |

**Should this be ML or rule-based?**  
**Rule-based first.** ML only as fallback for paraphrase coverage.

---

## TASK 3 — Entity Extraction

### Winner: **Dictionary + regex NER (primary) + GLiNER optional**

| Criterion | Decision |
|-----------|----------|
| Best choice for v1 | **Banking entity dictionary + patterns** (product, currency, tenor, branch) |
| Best ML add-on | **GLiNER (multilingual)** for zero-shot labels: `PRODUCT`, `CURRENCY`, `TENOR`, `CITY` |
| Why not AraBERT/CAMeL first | Excellent MSA NER, but entity set is **banking-domain**, not news persons/orgs; needs fine-tune |
| Arabic | Dictionary covers NBE product names exactly; GLiNER OK for open phrases |
| Latency | Dict: **&lt;2 ms**; GLiNER-small: **20–60 ms CPU** |
| Memory | Dict: tiny; GLiNER-small: ~0.5–1 GB |
| GPU | Optional for GLiNER; dict CPU-only |
| License | GLiNER: Apache-2.0 |

### Alternatives

| Model | Role |
|-------|------|
| CAMeL Tools NER | Best linguistic Arabic NER — use later if labeling persons/orgs in news |
| AraBERT NER | Needs fine-tune for NBE products |
| spaCy | Weak Arabic OOTB |
| ModernBERT NER | Not Arabic-first |

---

## TASK 4 — Query Rewriting

### Winner: **Rule engine + synonym dictionary (primary)**

| Criterion | Decision |
|-----------|----------|
| Best choice | **Deterministic rewrite**: colloquial map → formal banking terms → CategoryID hints |
| Tiny LLM? | **Only Tier-2**, if rewrite confidence low AND retrieval returns &lt;3 hits |
| Recommended tiny LLM | **Qwen2.5-3B-Instruct** or **Qwen3-4B** (Arabic strength) |
| Reject | Llama 3.2 3B / Phi-4 Mini / Gemma as primary Arabic rewrite (weaker Arabic) |
| When to use LLM rewrite | Egyptian slang / multi-intent long questions after rule miss |
| Fallback | Original normalized query unchanged |

**Should rewriting use an LLM?**  
**Rarely.** Default path must work without LLM.

---

## TASK 5 — Synonym Expansion

### Winner: **Versioned JSON ontology (human-owned)**

| Criterion | Decision |
|-----------|----------|
| Source of truth | `data/vocabulary/banking_synonyms.{ar,en}.json` + product alias tables |
| Not primary | Live embedding similarity (non-deterministic, can expand wrongly) |
| Optional | Offline embedding clustering to **propose** synonyms for analyst approval |
| Knowledge graph | Phase 4+ if product taxonomy grows (Neo4j/RDF) — not needed for MVP |

**Maintenance:** Product/content owners PR synonym changes; CI validates JSON schema; no silent model drift.

---

## TASK 6 — Embeddings

### Winner: **BAAI/bge-m3** (keep)

| Criterion | Decision |
|-----------|----------|
| Best choice | **bge-m3** |
| Why | Best open multilingual + Arabic among production-proven models; dense + sparse in one; already integrated |
| Cross-language | Strong (AR↔EN) |
| On-prem | Excellent; Apache-2.0 |
| Memory | ~2–4 GB RAM (FP16/CPU); GPU VRAM ~2–4 GB preferred |
| Latency | ~10–40 ms/query embed (batch helps) |
| GPU | Preferred; **CPU possible** (slower) |

### Alternatives

| Model | Why not replace bge-m3 now |
|-------|----------------------------|
| Qwen3-Embedding | Promising; less bank production track record; migration cost |
| multilingual-e5-large | Strong; no sparse multi-vector like m3 |
| gte-large / nomic / jina | Weaker Arabic or less hybrid capability |

---

## TASK 7 — Hybrid Retrieval

### Winner: **BM25 ⊕ Dense (bge-m3) ⊕ Metadata filter → RRF**

| Channel | Role |
|---------|------|
| BM25 | Exact rates, product IDs, Arabic morphology tokens |
| Dense | Semantic paraphrase |
| Metadata filter | Intent → doc_type / category / page_type |
| RRF | Stable fusion without score calibration pain |
| SPLADE | Optional Phase 4 learned sparse — not required now |

**Do not** use vector-only. Banking numbers and product names need lexical.

---

## TASK 8 — Reranker

### Winner: **BAAI/bge-reranker-v2-m3**

| Criterion | Decision |
|-----------|----------|
| Best choice | **bge-reranker-v2-m3** |
| Arabic | Best open multilingual balance for AR/EN |
| Contract | Retrieve **Top 20** → rerank → **Top 5** |
| Latency | ~30–80 ms for 20 pairs (GPU); higher on CPU |
| Memory | ~1–2 GB |
| GPU | Strongly preferred |
| License | MIT/Apache (BAAI) |

### Alternatives

| Model | Note |
|-------|------|
| Jina reranker v2 multilingual | Good alt; evaluate if latency better on your hardware |
| MiniLM cross-encoder | Fast but weaker Arabic |
| Qwen reranker | Heavier; not needed while bge-m3 reranker works |

---

## TASK 9 — LLM (answer synthesis only)

### Winner: **Qwen3-8B (latency tier) + Qwen3-14B (quality tier)**

| Criterion | Decision |
|-----------|----------|
| Best primary | **Qwen3-8B-Instruct** via Ollama |
| Best quality escalate | **Qwen3-14B-Instruct** for compare/explain queries |
| Why Qwen | Strongest open Arabic+English among practical on-prem sizes |
| Reject as primary | Llama 3.1/3.2 (Arabic weaker), Mistral Small, Phi-4, Gemma 3, DeepSeek (ops/license/Arabic tradeoffs) |
| GPU | 8B ≈ 8–12 GB VRAM; 14B ≈ 14–20 GB |
| CPU-only | Possible with llama.cpp/Ollama quantized — **p95 will miss &lt;500 ms target** |
| License | Tongyi Qianwen / Apache-style — confirm compliance with Legal |

**LLM is forbidden for:** language ID, primary intent, synonym expansion, ranking.

---

## TASK 10 — Structured Response

### Winner: **Pydantic models + prompt contract + post-parse**

| Layer | Tool |
|-------|------|
| API schema | **Pydantic** `SearchResponse` |
| Answer sections | Prompt `###` contract + `response_formatter.py` |
| Optional later | JSON grammar / outlines in Ollama when stable |
| Function calling | Not primary (Ollama variance) |

Deterministic extractive paths (cards/certs/accounts) bypass LLM when possible.

---

## TASK 11 — Target architecture

```
User Query
  → Language Detection (Lingua + heuristic)
  → Intent Classification (Rules → optional MiniLM)
  → Entity Extraction (Dict/regex → optional GLiNER)
  → Synonym Expansion (JSON ontology)
  → Query Rewriting (Rules; LLM Tier-2 rare)
  → Metadata Filter
  → Hybrid Retrieval (BM25 + Dense + RRF) → Top 20
  → Cross-Encoder Rerank → Top 5
  → Context Packer (parent expand)
  → LLM grounded generate OR extractive template
  → Structured Formatter (Pydantic sections)
  → Answer + Sources + Confidence
```

---

## TASK 12 — Folder structure

```
nbe-ai-search-mvp/
  services/
    api/                 # FastAPI routes, DI, orchestrator
    rag/
      pipeline.py        # end-to-end async pipeline
      language.py
      intent_classifier.py
      entity_extractor.py
      query_rewrite.py
      synonyms.py
      retriever.py
      reranker.py
      confidence.py
      prompt_builder.py
      response_formatter.py
      llm.py
    search_service/      # legacy implementations (gradual cutover)
  ingestion/
    cleaning/
    chunking/
    classification/
    embedding/
    lexical/
  data/
    vocabulary/          # synonym ontology
    curated_documents.json
  shared/                # config, schemas, logging
  tests/unit/
  tests/retrieval/
  docs/
    enterprise-model-decisions.md   # this file
    plan-enterprise-ai-search.md
```

---

## TASK 14 — Latency budget (GPU path, p50 targets)

| Stage | Target |
|-------|--------|
| Language detection | 2 ms |
| Intent (rules) | 3 ms |
| Entity extraction (dict) | 2 ms |
| Synonym + rewrite (rules) | 3 ms |
| Hybrid retrieve Top20 | 40–80 ms |
| Rerank Top20→5 | 40–80 ms |
| LLM generate (8B, short) | 250–400 ms |
| Formatter | 1–2 ms |
| **Total** | **&lt;500–600 ms p50** (GPU) |

CPU-only total: expect **1.5–4 s** depending on quantization — acceptable for internal MVP, not for public &lt;500 ms SLA.

---

## Security / banking constraints

- No external model calls at runtime  
- Log query hash, intent, chunk IDs, scores, model versions  
- Abstain / `NO_ANSWER` below confidence threshold  
- Citations required for answered responses  
- Model licenses reviewed by Legal before production

---

## Summary — one best choice per component

| Component | Best choice |
|-----------|-------------|
| Language | Lingua (AR/EN) |
| Intent | Rules (+ MiniLM fallback) |
| NER | Dict/regex (+ GLiNER optional) |
| Rewrite | Rules (+ Qwen3-4B rare) |
| Synonyms | Versioned JSON ontology |
| Embeddings | bge-m3 |
| Retrieval | Hybrid BM25+Dense+RRF |
| Reranker | bge-reranker-v2-m3 |
| LLM | Qwen3-8B / 14B |
| Structure | Pydantic + formatter |

---

*End of decision report*
