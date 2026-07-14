# NBE AI Search — Enterprise Retrieval Redesign Plan

**Audience:** Principal / Senior AI Engineering  
**Scope:** Retrieval quality (not infra rewrite)  
**Constraint:** Incremental delivery — do **not** rewrite the stack in one pass  
**Companion:** Existing enterprise `plan.md` remains the platform roadmap; this document owns retrieval quality only.

---

## Diagnosis Summary (Task 1)

### Current pipeline (as implemented)

```
Query
 → language + normalize
 → rule intent + synonym/colloquial expand
 → Hybrid (dense Chroma + BM25) + optional doc_type filter
 → junk / menu filters
 → keyword rerank + intent boosts
 → cross-encoder rerank (bge-reranker-v2-m3) → top 8
 → orchestrator short-circuits (certs/cards) OR context → LLM
```

### Root causes of poor retrieval

| # | Weakness | Why it hurts |
|---|----------|--------------|
| 1 | **Coarse `doc_type`** | All accounts share `account` — Initiative + Current + Savings compete equally |
| 2 | **Query expansion bias** | `"افتح حسابك"` boosts diaspora Initiative title over Current Account |
| 3 | **No product-aware ranking for accounts** | Cards/certs have prioritizers; accounts do not |
| 4 | **Canonical slug mismatch** | Boost targets `#/AR/Accounts`; real pages use `CategoryID=CurrentAccountsID` |
| 5 | **Boilerplate / menu leakage** | Menu-heavy pages still enter dense/BM25 before filters |
| 6 | **Flat chunking** | 400-token windows mix eligibility, fees, and nav; no parent-child |
| 7 | **Thin metadata** | Missing `product_name`, `page_type`, `intent`, `keywords`, `subcategory` |
| 8 | **Intent taxonomy incomplete** | Missing Offers, News, Reports, Corporate, SME as first-class filters |
| 9 | **Rerank pool too small / final k too high for LLM** | Rerank→8; enterprise pattern is retrieve 20 → rerank → **top 5** |
| 10 | **LLM sees wrong top docs** | Correct — this is retrieval, not prompt failure |

### Failure case

**Query:** `لو عايز افتح حساب بنكي اي الاوراق المطلوبة؟`

1. Intent → `account_open` → filter `doc_type=account`
2. Expansion adds phrases that match **مبادرة افتح حسابك في مصر**
3. Initiative page has long “الأوراق المطلوبة” sections → wins BM25 + dense + reranker
4. Current Account page is short / partially unavailable → loses
5. LLM correctly answers from retrieved Initiative page

**Fix direction:** finer account subtypes + Initiative penalty unless diaspora intent + prefer `CurrentAccountsID` / ProductDetails with document sections.

---

## Target architecture (Tasks 5–7, 11)

```
User Query
    ↓
Intent Detection (+ language)
    ↓
Query Rewrite (synonyms / banking formal)
    ↓
Metadata Filter (category + page_type + product family)
    ↓
Hybrid Search (BM25 ⊕ Vector) → Top 20
    ↓
Cross-Encoder Reranker → Top 5
    ↓
Context Packer (parent expand optional)
    ↓
LLM (grounded prompt)
    ↓
Structured Formatter
    ↓
Answer + Sources + Confidence
```

---

## Phase 1 — Ranking & Intent Surgery (1–2 weeks)

**Goal:** Stop wrong-page wins for accounts / docs queries without re-ingest.

### Deliverables

1. **Account disambiguation ranking**
   - Prefer `CurrentAccountsID`, `SavingLocalAccountsID`, ProductDetails
   - Penalize `OpenYourBankAccountInEgypt` unless query mentions مبادرة / خارج مصر / سفارة
2. **Fix canonical boosts** for accounts → CategoryIDs, not SPA shell `#/AR/Accounts`
3. **Rerank contract:** candidates 20 → return 5 for LLM context
4. **Prompt already updated** — keep structured NO_ANSWER prompt; add response formatter module
5. **Module façades** (thin wrappers, no big move yet):
   - `services/rag/pipeline.py`
   - `services/rag/retriever.py` (wrap HybridRetriever)
   - `services/rag/reranker.py` (wrap existing)
   - `services/rag/intent_classifier.py` (re-export)
   - `services/rag/prompt_builder.py`
   - `services/rag/response_formatter.py`
   - `services/rag/llm.py` (re-export)
6. Golden queries: account open + required documents

### Why this improves retrieval

Fixes score ties between diaspora Initiative and retail Current Account; reduces LLM context noise (5 vs 8).

| Metric | Expectation |
|--------|-------------|
| Accuracy (account docs queries) | **+15–25 pts** hit@1 on account-open golden set |
| Latency | **−50 to +100 ms** (smaller LLM context; same reranker) |
| Difficulty | **Low–Medium** |

---

## Phase 2 — Corpus Quality (2–3 weeks)

**Goal:** Cleaner docs + semantic / parent-child chunking + richer metadata → re-ingest.

### Task 2 — Preprocessing cleaner

Pipeline stages:

1. HTML artifact strip  
2. Nav / footer / widget / chrome removal  
3. Duplicate header collapse  
4. Template placeholder removal (`{{...}}`, empty CMS shells)  
5. Menu-list detection → drop or demote to `page_type=navigation`  
6. Quality score → skip embed if below threshold  

### Task 3 — Chunking

| Type | Role |
|------|------|
| **Parent** | Full product section (eligibility + docs + fees) |
| **Child** | 150–250 token semantic units for retrieval |
| **Overlap** | 40–60 tokens on child boundaries |
| **Product-aware** | Split on headings: مميزات / الأوراق المطلوبة / الرسوم / العائد |

Retrieve child → expand parent into LLM context.

### Task 4 — Metadata enrichment

Auto fields per document/chunk:

| Field | Extraction logic |
|-------|------------------|
| `title` | Cleaned page title |
| `page_type` | product / category / faq / initiative / nav / news |
| `category` | From CategoryID map |
| `subcategory` | Current / Savings / Diaspora Initiative |
| `language` | ar / en detector |
| `product_name` | ProductID decode or H1 |
| `service_name` | Digital / PhoneCash / etc. |
| `document_type` | web_page / pdf / stub |
| `intent` | Primary bank intent tag |
| `keywords` | YAKE/KeyBERT or banking lexicon hits |
| `last_updated` | Scrape `extracted_at` / page date if present |

Bump Chroma collection → `nbe_chunks_bge_m3_v4`.

| Metric | Expectation |
|--------|-------------|
| Accuracy | **+10–20 pts** overall nDCG@5 |
| Latency | Ingest **+20–40%**; query **neutral to −5%** (cleaner top-k) |
| Difficulty | **Medium–High** |

---

## Phase 3 — Enterprise Search Parity (3–4 weeks)

**Goal:** Azure/Vertex-like behavior on local stack.

### Task 5 — Intent taxonomy (full)

Exchange Rates · Certificates · Loans · Accounts · Cards · Branches · Offers · News · Reports · Corporate · SMEs · FAQ · Digital Banking · Other

Map each → metadata filter + optional boost/penalty profiles.

### Task 6 — Hybrid ranking

```
score = α · dense_norm + β · bm25_norm + γ · metadata_match + δ · freshness
```

Then RRF fusion as today; keep α≈0.6–0.7 for Arabic banking.

### Task 7 — Reranker

- Retrieve **Top 20**  
- Cross-encoder → **Top 5**  
- Recommended models (Arabic-capable, open-source):
  1. **BAAI/bge-reranker-v2-m3** (current — keep)
  2. **BAAI/bge-reranker-v2-gemma** (stronger, heavier)
  3. **jina-reranker-v2-base-multilingual** (good AR/EN, production-friendly)

### Tasks 8–10 — Prompt, formatter, confidence

```
confidence =
  0.30 · embedding_sim +
  0.25 · hybrid_rrf +
  0.30 · reranker_score +
  0.15 · llm_grounding_check
```

Abstain / NO_ANSWER if confidence < threshold or formatter cannot fill required fields.

### Task 12 — Full module cutover

Move logic into `services/rag/*`; orchestrator calls `pipeline.run()` only.

| Metric | Expectation |
|--------|-------------|
| Accuracy | **+5–15 pts** on hard multi-intent queries |
| Latency | Target **p95 &lt; 2.5s** local GPU/CPU mix |
| Difficulty | **Medium** (refactor) + **High** (eval harness) |

---

## Module target layout (Task 12)

```
services/rag/
  pipeline.py              # end-to-end
  retriever.py             # hybrid + filters
  reranker.py              # cross-encoder
  intent_classifier.py
  metadata.py              # filter builders + enrichment helpers
  prompt_builder.py
  response_formatter.py
  llm.py
  confidence.py

ingestion/
  cleaning/cleaner.py      # Phase 2
  chunking/semantic.py     # Phase 2
  classification/metadata_enricher.py
```

---

## Eval gate (every phase)

Before merge:

1. Golden set hit@1 / hit@5 / nDCG@5  
2. Account-open docs query must prefer Current/Savings ProductDetails over Initiative  
3. Cards / certificates regression suite green  
4. Latency budget logged  

---

## Non-goals (this redesign)

- Replacing Chroma with Azure AI Search / OpenSearch (later ADR)  
- Cloud LLM APIs  
- Full rewrite of FastAPI surface  
- Training a custom embedding model  

---

## Immediate next actions

1. ~~Land Phase 1 account ranking + Top20→Top5 rerank~~ **DONE**
2. ~~Scaffold `services/rag/` façades~~ **DONE**
3. ~~Phase 2 cleaner + parent/child chunking + metadata enricher~~ **DONE** (needs re-ingest → `nbe_chunks_bge_m3_v4`)
4. ~~Phase 3 intent taxonomy + confidence + formatter wiring~~ **DONE** (code); evaluate golden set after re-ingest
5. Run: `python scripts/run_ingest.py --source merged`

---

*Owner: AI Search engineering · Last updated: 2026-07-14 · Status: Phases 1–3 implemented in code; corpus re-ingest required for v4 metadata/chunks*
