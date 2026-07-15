# NBE AI Search — Root Cause Analysis & Technical Design

**Production hardening:** see [`enterprise-production.md`](enterprise-production.md) (QU, compression, calibrated confidence, semantic cache, planner, self-eval, citations, analytics, feedback, explain).

**Document type:** Design proposal (no implementation until approval)  
**Audience:** Principal / AI Engineering  
**Date:** 2026-07-14  
**Corpus under test:** `nbe_chunks_bge_m3_v4` (181 docs / **496 chunks**)  
**Eval snapshot:** Intent 100% · Top-3 86.4% · Forbidden 100% · Gate PASS  

---

## 0. Executive verdict

The failed cases are **not an LLM problem** and **not primarily a reranker problem**.

They are caused by a **retrieval control-plane failure**:

1. High-confidence intents correctly classified (`exchange_rate` / `personal_loan` at 0.95).  
2. Metadata filter correctly applied to sparse inventory.  
3. Filter finds **&lt; 3** hits → code immediately calls `intent_filter_fallback_broad`.  
4. Broad search floods the pool with accounts / news / generic products.  
5. Post-rerank **account prioritizer runs on every query**, pushing Current/Savings pages above the real FX page even when FX has reranker score **1.0**.  
6. Confidence mixes overlapping signals and still emits a Top-5 of wrong docs with low confidence / abstention inconsistency.

This must be redesigned as a **staged metadata filter + decision engine**, not threshold tweaking alone.

---

## 1. Failed cases — exact reproduction

| Query | Intent (OK) | Filter docs | Dense@filter | BM25@filter | Fallback? | Top result (bad) | Correct page in Top-5? |
|-------|-------------|-------------|--------------|-------------|-----------|------------------|------------------------|
| `سعر الصرف` | `exchange_rate` 0.95 | `exchange_rate`, `currency_converter` | **1** | **1** | **YES** | حساب فايدة بلس (`account`) | Yes @ rank 4, score **1.0** |
| `تحويل العملات` | `exchange_rate` 0.95 | same | **1** | **1** | **YES** | حساب التوفير (`account`) | Yes @ rank 4, score **1.0** |
| `قرض شخصي` | `personal_loan` 0.95 | `loan` | **1** | **0** | **YES** | `NewsCat1ID` product | AutoLoan category @ rank 2; **no Personal Loan AR page** |

Log signature (all three):

```text
intent_filter_fallback_broad  dense_count=0|1  bm25_count=0|1
filter_applied=False
```

---

## 2. Corpus facts (v4) — primary structural root cause

BM25 inventory by `doc_type` (496 chunks):

| doc_type | count | Notes |
|----------|------:|-------|
| general | 123 | Includes news-like / home / register |
| product | 97 | **includes NewsCat* and some loan category pages** |
| branch | 89 | |
| credit_card | 67 | |
| account | 37 | |
| wallet | 33 | |
| certificate | 20 | |
| debit_card | 17 | |
| **loan** | **4** | Mostly AR Loans hub + EN loan cats |
| **exchange_rate** | **2** | AR + EN stubs/pages only |
| certificate_rate | 3 | |
| **currency_converter** | **0** | Intent allows it → never helps filter count |

### Why the filter always collapses for FX / loans

Hybrid fallback rule (`hybrid_retriever.py`):

```python
if apply_filter and len(dense_chunks) < 3 and len(bm25_chunks) < 3:
    # BROAD SEARCH — no doc_type filter
```

With only **2** FX chunks (AR+EN) and language=`ar`, AR filter can see at most **1** language-matched FX page → always `&lt; 3` → **forced broad**.

Same for loans: AR `loan` inventory ≈ 1–2 pages → fallback.

**Conclusion:** Even perfect intent + perfect embeddings cannot satisfy a filter that demands ≥3 hits when inventory &lt; 3 for that (language ∩ doc_type).

This was worsened by v3→v4 rechunk (≈9953 → 496 chunks): aggressive child-only indexing + cleaner reduced alternative FX/loan fragments that previously padded the filter.

---

## 3. Component-by-component root causes

### 3.1 Query Understanding / Intent

| Finding | Detail |
|---------|--------|
| Severity | Low for these three failures |
| Evidence | Intent 100% on golden; all fail cases correctly `exchange_rate` / `personal_loan` at 0.95 |
| Gap | Intent taxonomy allows `currency_converter` but nothing is labeled that way |
| Gap | No soft category ladder (exact → family → related) |

**Not the primary cause of wrong Top-1.**

### 3.2 Query Rewrite

| Finding | Detail |
|---------|--------|
| Severity | Medium (amplifier) |
| Evidence | `سعر الصرف` expands to long multi-term bag including English slug + شراء/بيع |
| Problem | No structured fields (CategoryID / ProductID / intent tag) — only bag-of-words |
| Problem | Duplicated phrases inflate BM25 on common banking words also present in accounts |
| Problem | For loans: expands `تمويل` / `تسهيلات` which can match non-loan product text |

**Fix direction:** intent-conditioned structured rewrite, not longer free text.

### 3.3 Metadata / Classification

| Finding | Detail |
|---------|--------|
| Severity | **Critical** |
| FX | Only 2 chunks tagged `exchange_rate`; live rate table pages may be excluded/stubbed |
| Loans | AR Personal Loans category poorly represented; `NewAutoLoanID` often `product` not `loan` |
| News | `NewsCat1ID` remains `product` → survives loan filter family only after broad fallback |
| `currency_converter` | Dead doc_type in filter allow-list |

### 3.4 BM25

| Finding | Detail |
|---------|--------|
| Severity | High after fallback |
| Under filter | Correctly finds ~1 FX hit when available |
| After broad | Common Arabic tokens (`سعر`, `صرف`, `قرض`, `تمويل`) appear in many product/news pages |
| Index size | 496 docs makes rare types even rarer |

### 3.5 Dense retrieval (bge-m3)

| Finding | Detail |
|---------|--------|
| Severity | High after fallback |
| Under filter | Finds the one FX page |
| After broad | Semantic neighborhood of “سعر” / “تحويل” overlaps accounts mentioning rates/yield and transfer products (`ExpatriatesTransfers`) |

### 3.6 Hybrid fusion (unweighted RRF)

| Finding | Detail |
|---------|--------|
| Severity | High after fallback |
| Algorithm | Equal RRF of dense + BM25 ranks (`rrf_k=60`), score `* 30` |
| Problem | No intent/metadata channel weight |
| Problem | After broad pool, accounts/news enter both legs and fuse upward |

### 3.7 Reranker (bge-reranker-v2-m3)

| Finding | Detail |
|---------|--------|
| Severity | Medium (victim more than culprit) |
| Evidence | For FX queries the canonical page often receives **reranker score 1.0** |
| Failure mode | **Post-rerank account prioritizer reorders** Top-5 so accounts with score ~0.0 appear above FX |
| Secondary | On loans, news/product text can outrank thin loan hub pages when filter already dropped |

### 3.8 Account prioritizer bug (definite code defect)

`prioritize_account_chunks()` is invoked **unconditionally** in `SearchService.retrieve` (before and after rerank).

It sorts by account URL markers with **no intent gate**. Result for FX queries:

- `CurrentAccountsID` / savings / ProductDetails accounts get priority 2–5  
- FX page gets priority 0  
- Sort key `(-priority, -score)` puts accounts first even when FX score = 1.0  

This alone explains **سعر الصرف → Current/Plus Account** and **تحويل العملات → Savings**.

### 3.9 Confidence

| Finding | Detail |
|---------|--------|
| Severity | High (governance) |
| Current formula | Mixes “embedding”, “retriever”, “reranker” but after rerank **all three proxy the same score field** |
| Missing | BM25 raw score, metadata match, intent confidence, Top-1−Top-2 gap |
| Symptom | Confidence ~0.18 with wrong Top-1; does not encode “canonical FX exists @ rank 4 with score 1.0” |

### 3.10 Decision layer

| Finding | Detail |
|---------|--------|
| Severity | Critical (missing) |
| Current | `should_answer = confidence ≥ threshold ∧ len(chunks) > 0` |
| Missing | Retry with related categories, force-include canonical URL, or NO_ANSWER when Top-1 doc_type ∉ intent allow-list |

---

## 4. Causal chain (enterprise view)

```
Correct Intent (0.95)
        ↓
Metadata Filter on under-provisioned doc_type (1–2 chunks)
        ↓
dense_count < 3 AND bm25_count < 3
        ↓
intent_filter_fallback_broad  ← CONTROL PLANE BUG
        ↓
Broad hybrid pool (accounts, news, transfers, products)
        ↓
Reranker may still boost canonical FX (score 1.0)
        ↓
Account prioritizer (always on)  ← RANKING BUG
        ↓
Wrong Top-1 → LLM/eval failure
```

---

## 5. Design recommendations (approve before coding)

### TASK 1 — Staged Metadata Filter Policy

Replace binary fallback with configurable ladder when `intent.confidence ≥ 0.9`:

| Stage | Action | Stop condition |
|-------|--------|----------------|
| L0 Exact | `doc_types = intent.allowed_doc_types` | ≥ `min_hits` (default **1** when conf≥0.9, else 3) |
| L1 Parent | Expand to parent category types (e.g. cards family) | ≥ min_hits |
| L2 Related | Related types map (FX → none for accounts; loans → product **only if** URL contains Loan) | ≥ min_hits |
| L3 Soft lexical | Keep language filter; post-filter by canonical slug / CategoryID | ≥ 1 canonical |
| L4 Broad | Last resort; **penalize** out-of-intent doc_types heavily | always |

**Policy for banking production:**  
If L0 finds the canonical slug/page (even 1 hit), **never** broaden.

Config sketch:

```yaml
metadata_filter:
  high_confidence: 0.9
  min_hits_high_conf: 1
  min_hits_low_conf: 3
  allow_broad_fallback: true   # but only at L4
  forbid_broad_for: [exchange_rate, personal_loan]  # optional hard lock
```

### TASK 2 — Retrieval Decision Engine

Insert after rerank / before LLM:

| Decision | Condition |
|----------|-----------|
| `ANSWER` | Top-1 doc_type ∈ allow-list OR URL matches canonical slug; confidence ≥ threshold |
| `RETRY_RELATED` | High intent conf but Top-1 out-of-family; broaden one stage and re-rerank once |
| `FORCE_CANONICAL` | **Deprecated.** Hard inject/pin removed. Decision engine emits `RETRY_RELATED` / `NO_ANSWER` only. Soft boosts live in `business_rules.py`. |
| `NO_ANSWER` | No canonical; Top-1–Top-2 gap tiny & all out-of-family; or confidence &lt; threshold |

### TASK 3 — Confidence redesign

Normalized [0,1] breakdown (log every term):

```
C =
  0.15 · dense_sim(top1)
+ 0.15 · bm25_norm(top1)
+ 0.20 · metadata_match(top1, intent)   # 1/0.5/0
+ 0.15 · intent_confidence
+ 0.20 · reranker_score(top1)
+ 0.15 · margin(top1 − top2)
```

Hard multiplicative penalty `× 0.4` if `metadata_match == 0` and intent conf ≥ 0.9 (prevents answering from accounts on FX intent).

### TASK 4 — Exchange rates

1. **Corpus:** Ensure AR+EN FX page + rich stub chunks for شراء/بيع/دولار/يورو/بنكنوت/تحويل; ingest enough FX child chunks that L0 has ≥1 strong hit.  
2. **Synonyms:** Map `سعر الصرف|تحويل العملات|الدولار|اليورو|buying|selling|currency converter` → expand to `ExchangeRatesAndCurrencyConverter` + currencies.  
3. **Hard boost:** Canonical slug boost **before** any account heuristics; **disable account prioritizer** unless intent ∈ accounts.  
4. **Negative filter:** After FX intent, drop `doc_type=account` unless query also contains حساب.

### TASK 5 — Loans

1. Classify `PersonalLoansCatID`, `NewAutoLoanID`, `*Loan*` CategoryIDs as `loan` (not bare `product`).  
2. Synonyms: `قرض|تمويل|تسهيلات|personal loan|retail loan|consumer loan`.  
3. Penalty: `NewsCat*`, MultiArticle, Home never outrank `loan` when intent=`personal_loan` (URL deny-list).  
4. Stub AR Personal Cash Loan page if scrape incomplete.

### TASK 6 — Structured Query Rewrite

Output object (not only string):

```json
{
  "surface": "سعر صرف الدولار شراء بيع",
  "intent": "exchange_rate",
  "entities": ["USD"],
  "synonyms": ["تحويل العملات", "محول العملات"],
  "canonical_slugs": ["ExchangeRatesAndCurrencyConverter"],
  "category_ids": [],
  "negative_doc_types": ["account", "loan", "certificate"]
}
```

Embedding query uses `surface`; filters/boosts use structured fields.

### TASK 7 — Weighted RRF + channels

```
score = w_bm25 · RRF_bm25
      + w_dense · RRF_dense
      + w_meta · metadata_channel   # binary or soft
      + w_intent · intent_boost
```

Suggested defaults (FX/loan tuned):

| Intent family | w_bm25 | w_dense | w_meta | w_intent |
|---------------|-------:|--------:|-------:|---------:|
| exchange_rate | 0.35 | 0.25 | 0.25 | 0.15 |
| personal_loan | 0.30 | 0.25 | 0.30 | 0.15 |
| general | 0.40 | 0.40 | 0.10 | 0.10 |

### TASK 8 — Enterprise retrieval tracing

Per query structured log (JSONL):

- language, intent, intent_conf, entities  
- rewrite object  
- filter stage used (L0–L4), doc_types  
- BM25 Top10 ids/urls/scores  
- Dense Top10  
- Fusion Top20  
- Reranker Top5 + scores  
- decision + confidence breakdown  
- stage latencies (ms)

### TASK 9 — Eval upgrades

Extend golden + metrics: Top1, Top3, MRR, nDCG@10, Recall@10, Precision@5, intent acc, avg confidence, p50/p95 latency.  
Add regression cases exactly for the three failures with `forbidden_doc_types` field.

### TASK 10–11 — Target module layout

```
services/rag/
  pipeline.py              # orchestration
  query_understanding.py   # lang + intent + entities
  query_rewrite.py         # structured rewrite
  metadata_filter.py       # staged L0–L4
  fusion.py                # weighted RRF
  retriever.py             # hybrid invoke
  reranker.py              # façade
  decision_engine.py       # ANSWER/RETRY/NO_ANSWER
  confidence.py            # new breakdown
  tracing.py               # enterprise logs
  evaluation.py            # metrics helpers
```

Gate prioritizers (`account_rank`, `card_catalog`) **must be intent-scoped**.

---

## Status (2026-07-14)

**Phase A implemented** in code: intent-gated prioritizers, staged filter (`min_hits=1` @ conf≥0.9 / no broad when any exact hit), canonical force-inject, loan/NewsCat classification, decision engine, confidence redesign. Smoke: `سعر الصرف` / `تحويل العملات` → ExchangeRates; `قرض شخصي` → Loans.

## 6. Implementation phases (after approval only)

### Phase A — Hotfixes (1–2 days, highest ROI)

1. Intent-gate `prioritize_account_chunks` / card prioritizers.  
2. Staged filter: if L0 hits ≥1 at intent≥0.9, **skip broad**.  
3. Canonical force-inject for `ExchangeRatesAndCurrencyConverter` / `Loans` / `PersonalLoansCatID`.  
4. Loan CategoryID → `doc_type=loan`; NewsCat penalty.  

**Expected:** Fix the three golden failures without full redesign. Latency ≈ same.

### Phase B — Decision engine + confidence + weighted fusion (3–5 days)

### Phase C — Structured rewrite + tracing + eval metrics (3–5 days)

### Phase D — Corpus repair (reclassify + richer FX/loan stubs + controlled re-ingest)

**Important:** Do **not** only re-ingest; fix control plane first or failures will recur whenever rare types stay sparse.

---

## 7. What we will NOT do

- Use LLM for ranking or primary intent.  
- Disable metadata filters globally.  
- Blindly raise/lower confidence threshold as the “fix”.  
- Treat Azure/Vertex as cloud APIs — patterns only, on-prem stack remains.

---

## 8. Approval checklist

Please approve / amend:

- [ ] Phase A hotfixes as immediate patch  
- [ ] Staged metadata filter L0–L4 design  
- [ ] Decision engine states  
- [ ] New confidence formula  
- [ ] Intent-scoped prioritizers  
- [ ] Module refactor layout  
- [ ] Corpus repairs for FX/loan  

**No code changes will be made until you approve this analysis.**

---

## 9. Appendix — key code references

| Area | Path |
|------|------|
| Broad fallback | `services/search_service/hybrid_retriever.py` L85–96 |
| Unconditional account sort | `services/search_service/search.py` L213–229 + `account_rank.py` |
| Confidence | `services/rag/confidence.py` |
| Intent allow-lists | `services/search_service/intent_classifier.py` |
| Doc typing | `ingestion/classification/doc_classifier.py` |
| FX stubs | `data/product_stubs.json` |

---

*End of analysis.md*
