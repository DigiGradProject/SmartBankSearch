# Root-Cause Report: Exchange-Rate Pages Cited as Sources for Al Ahly Product Queries

**Date:** 2026-10-02
**Scope:** `nbe-ai-search-mvp`
**Symptom:** Queries like *"What is National Bank of Egypt - Al Ahly Points?"*, *"… - Al Ahly Business?"*, *"… - Al Ahly Net - Platinum?"* return a **correct answer**, but the **cited sources are Exchange Rates & Currency Converter pages**, not the product page the answer actually came from.

---

## 1. TL;DR

A **semantic intent misfire** (these queries classify as `exchange_rate` at cosine ≈ 0.52) triggers an **intent expansion** that appends *"exchange rate currency converter banknote transfer rate"* to the embedded query. The re-written query then matches the ExchangeRates page almost exactly (lexical/rerank score **1.0**), which **outranks the true source** (the actual Al Ahly product page at 0.14–0.40). Because citations are selected purely by top score with no query-source relevance check, the wrong page gets cited — even though the LLM answers from the correct product chunk.

---

## 2. Evidence (from production logs and a live reproduction)

From `data/logs/retrieval_analytics.jsonl` (2026-10-02):

| Query | Intent (wrong) | Top source (score) | True source (score) |
|---|---|---|---|
| What is National Bank of Egypt - Al Ahly Points? | `exchange_rate` (0.522) | Exchange Rates And Currency Converter (**1.0**) | Al Ahly Points (0.142) |
| What is National Bank of Egypt - Al Ahly Business? | `exchange_rate` (0.522) | Exchange Rates And Currency Converter (**1.0**) | Al Ahly Business (0.404) |
| What is National Bank of Egypt - Al Ahly Net - Platinum? | `exchange_rate` (0.522) | Exchange Rates And Currency Converter (**1.0**) | Al Ahly Net - Platinum (0.0) |

Rewritten query observed for all three:

```
What is National Bank of Egypt - Al Ahly Business? exchange rate currency converter banknote transfer rate
```

The bank's site literally has an `ExchangeRatesAndCurrencyConverter` page whose page title contains all these tokens — so when the rewritten query is embedded and reranked, it matches that page near-perfectly (score 1.0).

Live reproduction (`scripts/diagnose_intent.py`, BGE-M3 vs. the 76 intent prototypes):

```
QUERY: What is National Bank of Egypt - Al Ahly Points?
  exchange_rate        0.532   <-- wins
  digital_banking      0.412
  credit_card          0.405
```

Same result for the Business (0.522) and Net-Platinum (0.520) variants.

---

## 3. Root causes, ranked

### RC1 (primary): Brand-product queries classify as `exchange_rate` — layered classifier picks the wrong winner

**File:** `services/rag/intent_fallback.py` (layering), `services/rag/semantic_intent.py` (scoring)

The pipeline classifies intent by cosine similarity between the query embedding and prototype embeddings (`INTENT_PROTOTYPES`). The user's queries are of the form "What is National Bank of Egypt - Al Ahly **X**?" — near-duplicates of the scraped page **titles**. These have no lexical overlap with any banking-intent prototype, so **all intents score low and close together** (0.40–0.53). In that flat band, `exchange_rate` edges out the rest, likely because its prototypes are numerically/entity dense ("USD EGP", "how much is one dollar...") and BGE-M3 maps title-like, entity-heavy strings nearer that region.

Layering then works *against* safety here:

```python
# intent_fallback.py — classify_layered()
if semantic.confidence >= high (0.90):   return semantic      # not hit
if semantic.confidence >= mid (0.60):    regex confirm/route  # not hit (0.52)
if rule.intent != general_faq:           return rule          # regex says general_faq -> skipped
if semantic.confidence >= min (0.42):    return semantic      # <-- 0.52 accepted here
```

The `semantic_intent_min_score = 0.42` gate was intended to reject junk similarities, but **BGE-M3 cosine similarities for unrelated banking text routinely sit at 0.42–0.55**, so the floor lets the wrong winner through. Below 0.42, queries correctly fall to `general_faq` (broad hybrid, no intent filter) — the logs confirm the short query "alahly points" (best score 0.426 → `branch_locator`... actually below all route thresholds in one run) sometimes abstains with "No retrieved documents" instead of misrouting.

**Consequence:** intent = `exchange_rate`, category = `exchange_rates`, and — critically — the intent's `expand_en` metadata is attached to the query.

### RC2 (amplifier): The intent expansion string is injected into the search query

**Files:** `services/rag/query_rewrite.py`, `services/rag/query_understanding.py`, `services/search_service/intent_classifier.py`

```python
# query_rewrite.py — _rule_rewrite()
expanded = expand_with_synonyms(expanded, language)
if intent_expand and intent_expand not in expanded:
    expanded = f"{expanded} {intent_expand}".strip()
```

For `exchange_rate`, `expand_en = "exchange rate currency converter banknote transfer rate"`. The **original user query is never retrieved**; only this rewritten string is embedded and reranked (`search.py`: `embed_query = expanded_query`). The appended tokens *drown out* the actual topic ("Al Ahly Business") and point retrieval at the exchange-rates page — whose on-site title is *"Exchange Rates And Currency Converter"*.

### RC3 (decisive for citation ordering): ExchangeRates page outranks the true source, and citations are chosen purely by score

**Files:** `services/search_service/search.py` (rerank + pool), `services/context_builder/builder.py` (`_select_citations`)

Observed scores after rerank: ExchangeRates = **1.0** vs. AlAhlyBusiness = **0.404**, AlAhlyPoints = **0.142**, AlAhlyNetPlatinum = **0.0**. With BGE-reranker-v2-m3 outputs clipped to [0,1] and the query containing "exchange rate currency converter...", the exchange page saturates at ~1.0.

`ContextBuilder._select_citations` then ranks by score and applies a **relative cutoff of 0.12 from the top score**:

```python
if top_score - chunk.score > 0.12:
    continue
```

Only the ExchangeRates page survives (0.404 is 0.596 below the top). Even when the true product page is in the pool, it is mathematically excluded from citations. The LLM still answers correctly because the **context** (packed separately, `context_max_tokens=2000`) contains the product chunk text — hence "correct info, wrong sources."

Also relevant: `citation_min_score = 0.45` — for AlAhly Points (0.142) and Net Platinum (0.0), the true source would have been dropped by the floor anyway.

### RC4 (propagation): Semantic cache stores and replays the bad citation set

**Files:** `services/api/orchestrator.py` (`_get_semantic_cache`, `_cached_response_is_supported`), `services/rag/semantic_cache.py`

`semantic_cache_threshold = 0.95` with a 24h TTL. Similar brand queries replay a stored response whose citations were already poisoned. The guard `_cached_response_is_supported` only requires **one** overlapping URL between cached citations and current retrieval — and since ExchangeRates keeps getting retrieved, the poisoned answer keeps being accepted. This is why the pattern repeats across many queries ("...and others").

### RC5 (contributing): No brand/entity awareness anywhere in the pipeline

There is no document, alias table, or intent for NBE brand product names ("Al Ahly Points", "Al Ahly Net", "Al Ahly Business"). The prototypes have `digital_banking` (closest semantic neighbor for "Net" products) and `corporate` (for "Business"), but nothing teaches the classifier that "Al Ahly X" = the NBE product called X. Note the corpus **does** contain these pages with correct titles/URLs, so the raw material exists.

### RC6 (minor, UX layer): Frontend silently shows wrong sources

`frontend/ai-search-toggle/src/App.tsx` renders whatever citations the API returns. There's no UI-level sanity check (e.g., hiding citations whose relevance_score ≪ the answer's basis), so the mismatch is directly exposed to users.

---

## 4. Why exchange rates *specifically*

1. The exchange-rate intent has the most "entity-dense" prototypes (currency names, numbers, "how much is X"), and BGE-M3 maps short title-like product queries closest to that cluster among all prototypes (measured: 0.520–0.532, vs 0.40–0.45 for everything else).
2. Its `expand_en` string ("exchange rate currency converter banknote transfer rate") happens to be an almost verbatim match for the literal page title *"Exchange Rates And Currency Converter"* on the bank's site — guaranteeing a near-perfect rerank score (1.0) for that page whenever the expansion fires.
3. Every other intent's expansion strings either don't correspond to a single dominant page title, or are longer/more varied, so no other intent produces such a dominant wrong source.

---

## 5. Recommended fixes (ordered by impact/effort)

1. **Gate misfires at the source (RC1):** raise `semantic_intent_min_score` from 0.42 → ~0.58, **and/or** require a margin (best − second-best ≥ 0.05) before accepting a low-band semantic intent. These title-like queries scored 0.52 with runner-up 0.41–0.45 — a margin rule kills exactly this failure while keeping legit mid-band hits. Consider also: if the query contains a known product-page title/entity from the corpus, skip intent expansion entirely.
2. **Make expansion additive, not overriding (RC2):** embed a *combined* query — original + expansion — with the original weighted/kept intact (e.g., two-channel retrieval: embed original and expanded separately, fuse with RRF, which the codebase already does for dense+BM25), instead of concatenating expansion tokens onto the only embedded string.
3. **Category-aware citation guard (RC3):** in `_select_citations`, never let a citation from a *different category* than the dominant context category outrank it when its score gap > ~0.3 — or simply require the cited page to share the answer's dominant category (`exchange_rates` vs `products` here). Cheap, surgical, would have prevented this exact symptom.
4. **Tighten cache acceptance (RC4):** require the **top** current chunk's URL to appear among cached citations (not "any overlap"), and lower the semantic-cache threshold or shorten TTL so poisoned entries expire faster; purge existing cache entries citing ExchangeRates for product queries.
5. **Add brand prototypes (RC5):** add "Al Ahly Points / Al Ahly Net / Al Ahly Business" prototypes under `digital_banking`/`corporate` (and a dedicated `brand_product` intent if brand queries grow), plus alias terms to the synonyms vocab.
6. **Regression test:** add a unit test asserting that "What is National Bank of Egypt - Al Ahly …?" queries do **not** resolve to `exchange_rate` and do not cite `ExchangeRatesAndCurrencyConverter` — encode the three observed queries as fixtures.

---

## 6. Files implicated

| File | Role in the bug |
|---|---|
| `services/rag/semantic_intent.py` | picks `exchange_rate` at 0.52 (floor 0.42) |
| `services/rag/intent_fallback.py` | layered routing accepts low-band semantic winner |
| `services/rag/query_rewrite.py` | injects exchange-rate expansion into the embedded query |
| `services/search_service/search.py` | embeds/reranks only the rewritten query; builds citation pool |
| `services/search_service/reranker.py` | saturates ExchangeRates at 1.0 |
| `services/context_builder/builder.py` | citation = top-score-only + 0.12 relative cutoff → only wrong page survives |
| `services/api/orchestrator.py` | semantic cache replays poisoned responses |
| `services/rag/intent_prototypes.py` | no brand-product prototypes |
| `shared/config.py` | `semantic_intent_min_score=0.42`, `citation_min_score=0.45` |

---

## 7. Diagnostic artifacts

- Repro script: `scripts/diagnose_intent.py` (prints per-intent prototype scores for the failing queries).
- Log queries used: `grep -i "alahly" data/logs/retrieval_analytics.jsonl`, `data/logs/search_audit.jsonl`.

---

## 8. Fixes applied (2026-10-02)

1. **RC1 — intent gate hardened** (`services/rag/semantic_intent.py`, `shared/config.py`):
   `semantic_intent_min_score` 0.42 → **0.58** and a new **margin rule**
   (`semantic_intent_margin = 0.05`): the semantic winner must beat the
   runner-up intent, else the query routes to `general_faq` broad retrieval
   (log: `semantic_intent_rejected`).
2. **RC2 — expansion gating** (`services/rag/query_rewrite.py`): intent
   expansions are appended only for deterministic (`critical`/`regex`)
   classifications or high-band semantic wins (≥ 0.90); mid-band semantic
   winners can no longer inject "exchange rate currency converter …" tokens.
   Synonym expansion ("alahly points → loyalty points") still applies.
3. **RC3 — off-topic citation guard** (`services/context_builder/builder.py`,
   `citation_category_gap = 0.30` in config): when the top-scored chunk beats
   the runner-up by > 0.30 **and** its body shares no discriminative vocabulary
   with the query (site-generic tokens like "bank" ignored, word-boundary
   matching in EN, substring in AR), it is excluded from citations and the
   anchor re-bases on the remaining evidence, whose score then bypasses the
   global `citation_min_score` floor (log: `citation_offtopic_head_dropped`).
4. **RC4 — cache hardened** (`services/api/orchestrator.py`):
   `_cached_response_is_supported` now requires the **top current chunk's URL**
   to appear among the cached citations (was: any overlap). Poisoned entries
   purged via `scripts/purge_semantic_cache.py --apply` (5 deleted / 10).
5. **RC5 — brand prototypes** (`services/rag/intent_prototypes.py`,
   `data/vocabulary/banking_synonyms.en.json`): Al Ahly Net/Points →
   `digital_banking`, Al Ahly Business → `corporate` prototypes; brand alias
   synonyms. Also `digital_banking` intent now admits `doc_type=product` so
   Net pages satisfy the decision engine.
6. **Planner guard** (`services/rag/query_planner.py`): fragments that fall
   through to the broad `general_faq` fallback no longer open their own
   retrieval branch (the whole-query classification already covers them).

### Post-fix verification (live retrieval + citations)

| Query | Intent | Answered | Top citation |
|---|---|---|---|
| …Al Ahly Points? | general_faq (broad) | ✅ | **Al Ahly Points** |
| …Al Ahly Business? | corporate (0.669) | ✅ | **Al Ahly Business** |
| …Al Ahly Net - Platinum? | digital_banking (0.615) | ✅ | **Al Ahly Net - Platinum** |
| exchange rates USD to EGP (control) | exchange_rate (regex) | ✅ | Exchange Rates & Currency Converter ✅ correct |
| سعر الدولار اليوم (control) | exchange_rate | ✅ | سعر الصرف و تحويل العملات ✅ correct |

Tests: 137 unit tests pass, including new regression suite
`tests/unit/test_exchange_rate_citation_regression.py`.

### Known separate issue (not fixed here)

`services/context_builder/compressor.py::_char_jaccard` compares **character
sets**, so any two fluent English paragraphs score ≥ 0.82 (both cover the full
alphabet) and the compressor collapses most English chunks into one. This
shrinks LLM context for English queries and was worked around in the citation
tests via `compress=False`; it deserves its own fix (e.g., token n-gram
shingles).
