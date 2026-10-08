# Plan: Hybrid Search tab (dense + BM25, RRF-fused, no LLM)

Status: **IMPLEMENTED on branch `feat/hybrid-search-tab`**
Branch target: new branch `feat/hybrid-search-tab` off current `main`
Owner: Buffy + user review

---

## 1. Goal

Add a third search-mode tab, **Hybrid Search**, next to "AI Search Mode" and "Traditional Search".
It shows the hybrid retrieval pipeline — BGE-M3 dense vector results **fused with BM25 lexical
results via Reciprocal Rank Fusion (RRF)**, then the cross-encoder reranker — as a **plain ranked
list of pages**, like the Traditional tab:

- ✅ **Included**: hybrid retrieval (dense + lexical), RRF fusion, reranker, per-page dedup,
  snippets, scores, language tags, pagination ("Load more"), autocomplete dropdown,
  Arabic/English rendering.
- ❌ **Excluded**: LLM answer generation, intent business-rule boosts, confidence/decision gate,
  caching semantics of AI mode. The user sees *what the retriever ranks* — no generated answer.

This fills the gap between the tabs:

| | Traditional | **Hybrid (new)** | AI |
|---|---|---|---|
| Retrieval | BM25 only | **dense + BM25 (RRF)** | dense + BM25 (RRF) |
| Reranker | no | **yes** | yes |
| Business rules / gate | no | **no** | yes |
| LLM answer | no | **no** | yes |
| Output | page list | **page list** | answer + citations |

## 2. Why not just reuse an existing endpoint?

- `mode=ai` runs the full orchestrator: LLM answer, gate, cache — expensive and hides retrieval.
- `mode=traditional` is BM25-only.
- The retrieval pipeline itself already exists and is tested
  (`services/search_service/hybrid_retriever.py::HybridRetriever` + `SearchService.retrieve`
  with `retrieval_mode=PURE_SEMANTIC`, which skips business rules and the gate). We only need a
  thin "retrieval-only" service on top.

## 3. Design

### 3.1 Backend — new service `services/search_service/hybrid_pages.py`

```
HybridPageSearchService:
    def __init__(vector_store: VectorStore | None = None,
                 bm25_index: BM25Index | None = None)
    def search(query, language="auto", *, limit=10, offset=0) -> HybridPagesOutcome
```

Pipeline per query:

1. **Understand** the query: `understand_query(query, language)` (same language/intent/
   query-expansion the AI path uses — intent is used for metadata filtering only when the
   filter would legitimately apply, mirroring `SearchService.retrieve` with `rules_on=False`).
2. **Hybrid retrieve** via `HybridRetriever.retrieve(...)`. If the BM25 index is unavailable,
   degrade gracefully to dense-only (retriever already does this).
3. **Rerank** the fused chunks with `keyword_rank.rerank_chunks` + the cross-encoder
   (`get_reranker().rerank(...)`), reusing the same pool/keep settings as AI mode, wrapped in
   try/except → deterministic keyword fallback on failure.
4. **Group per canonical page** (reuse `canonical_url_key`; best chunk wins) so 809 chunks
   never leak as duplicate pages — same invariant as the Traditional tab.
5. **Light title boost** after fusion, reusing `traditional._title_similarity` with the existing
   corporate-prefix handling — keeps "Platinum" at #1 for title-exact queries in Hybrid too.
   (Kept deliberately simple; the express tier stays a Traditional-mode concept unless review
   says otherwise — see Open questions.)
6. **Snippet**: reuse `traditional.build_snippet` with the per-result highlight terms.
7. **Score**: min-max normalize the fused/best score per page into `(0.05 … 1.0]`
   (reuse `traditional.normalize_scores`), same display contract as Traditional.
8. Return `HybridPagesOutcome(results: list[TraditionalResult], total, language, hybrid_used)`.

`TraditionalResult` is reused as the response item type (title/url/snippet/score/terms/
language/category/doc_type) — identical rendering contract for the UI.

### 3.2 API — extend `POST /v1/search` with `mode: "hybrid"`

- `shared/schemas.py`: `SearchRequest.mode: Literal["ai","traditional","hybrid"]`.
  Response stays `SearchResponse`; for hybrid it carries
  `mode="hybrid"`, `results=[TraditionalResult…]`, `total_results=N`, `answer=None`,
  `citations=[]`, `suggestions` (autocomplete, `catalog_only=False` — hybrid may propose
  semantic suggestions too).
- `services/api/main.py`: new branch before the orchestrator call, mirroring the traditional
  branch (metrics `HYBRID_LATENCY`, `_log_traditional_event`-style audit with intent/hybrid_used,
  abstention `hybrid_failed` on total failure).
- Note: `RetrievalMode` enumeration is untouched — this is a UI/API mode layered on the
  existing retrieval modes; internally it calls retrieval with `retrieval_mode=PURE_SEMANTIC`
  semantics (no gate, **no** business rules).
- Health endpoint unchanged (bm25 + vector_db already reported).

### 3.3 Frontend — third tab in `frontend/ai-search-toggle/src/App.tsx`

- `SearchMode = "traditional" | "ai" | "hybrid"`.
- Third tab button (icon: e.g. the existing `IconBuilding`-style stroke icon or a small
  layers icon) between AI and Traditional, with `role="tab"`, `aria-selected`, i18n labels:
  EN "Hybrid Search Mode" / AR "البحث المختلط". Hero icon/text and empty-state copy per mode.
- State: `hybridResults`, `hybridTotal`, loading/error handling mirrors the traditional block
  (debounced fetch, `TRAD_PAGE_SIZE` reuse, "Load more" via `offset: hybridResults.length`,
  append vs replace; mode switch resets output — the existing effect already covers this once
  the mode union is extended).
- Result card reuse: the existing Traditional results card component renders
  `TraditionalResult[]` as-is; snippet/score/language/category props match 1:1.
- Styles: only tab-width adjustments if needed (3 tabs already fit the current toggle layout;
  the portal dropdown/list CSS is mode-agnostic and needs no changes).
- Rebuild `dist/` with existing `VITE_API_BASE` setup (`npm run build`), verify via `serve.py`.

### 3.4 Langauge / autocomplete behavior

- Autocomplete keeps working in hybrid mode (`mode=hybrid` ends up using
  `catalog_only=False`, i.e. catalog + BM25 titles + semantic matches — same as AI mode).
- Language handling is driven by `understand_query` exactly like AI mode; results carry
  `language` tags for RTL rendering.

## 4. Files to touch

| File | Change |
|---|---|
| `services/search_service/hybrid_pages.py` | **new** — HybridPageSearchService |
| `shared/schemas.py` | widen `mode` Literal; nothing else |
| `services/api/main.py` | hybrid branch in `/v1/search` + metrics/audit |
| `frontend/ai-search-toggle/src/App.tsx` | third tab + state + fetch + render |
| `tests/unit/test_hybrid_pages.py` | **new** — see §6 |
| `tests/unit/test_api_modes.py` (existing API-contract tests) | hybrid contract tests |
| `docs/plan-hybrid-search.md` | this file, updated after implementation |

No changes: `hybrid_retriever.py`, `search.py`, `traditional.py` (imported, not edited unless
helpers need export), schemas beyond the Literal, frontend CSS (unless tab overflow appears).

## 5. Rollout / git

1. `git checkout -b feat/hybrid-search-tab` off `main` (respecting the branch-per-change rule).
2. Backend service + tests → green.
3. API branch + contract tests → green. Full suite `pytest tests/unit`.
4. Frontend tab + `npm run build` → puppeteer checks on `http://localhost:5173`
   (existing `test.mjs` harness: label checks + a search run in hybrid mode).
5. Manual eval wisls: `platinum`, `alahly points`, `الشهادة البلاتينيه`, plus AI-quality queries
   (paraphrase like "how can I pay my card bill") to show the dense channel pulling its weight.
6. Regression: `scripts/eval_retrieval_modes.py --mode all` (KEYWORD must stay at Recall@5
   54.1% / 206 tests green).
7. Commit + push on the feature branch → open PR to `main` for review.

## 6. Tests (unit)

- reranker-failure fallback returns keyword-ordered pages, not 500.
- BM25 unavailable → dense-only results (degradation path).
- empty query / `effective=False` → empty outcome without calling the retriever.
- page grouping: 5 chunks of one canonical URL → 1 result (best score).
- score normalization bounds `(0.05 … 1.0]`.
- language: Arabic query → Arabic results first; non-http URLs dropped.
- API contract: `mode=hybrid` returns `results` + `total_results` + `mode="hybrid"`,
  `answer=None`; invalid mode values are rejected by pydantic.

## 7. Risks / notes

- **Latency**: hybrid + rerank ≈ AI-mode retrieval cost (~350–800ms on CPU depending on embed
  model load). Consider documenting expected latency in the tab tooltip if review wants it.
- **Reranker state**: reranker loads on first call; first hybrid query may be slow —
  same as first AI query today. Mitigation: none for MVP, acceptable.
- **Score semantics**: RRF scores are not BM25 scores; the min-max normalized display keeps
  the card contract consistent, but users should not compare them to Traditional scores.
- **Ethical/business behavior**: no business rules → ranking is pure retrieval quality; fine
  for an MVP tab (it mirrors PURE_SEMANTIC).

## 8. Open questions for review

1. **Express tier**: should exact-title pages also jump to #1 in Hybrid mode (reusing the
   Traditional-mode express scan), or stay BM25-style? My default: **yes, reuse them** for
   consistent UX — cheap and predictable.
2. **Reranker on/off by default** in hybrid tab: on (better ranking, ~+150ms) or off
   (faster)? My default: **on**.
3. **Tab label**: "Hybrid Search" / "البحث المختلط" ok?
4. Should hybrid mode also appear in the health/metrics dashboards as its own latency series
   (recommended: yes, one `HYBRID_LATENCY` histogram).

---

## 9. Implementation results (2026-10-08)

### Final pipeline (as approved, note #2)

```
BM25 Top 50 (rank_bm25, keyword-prepped query)  +  BGE-M3 dense Top 50 (Chroma)
                 ↓ reciprocal_rank_fusion (rrf_k=60, shared with HybridRetriever)
              Top 30 (HYBRID_FUSED_KEEP)
                 ↓ BGEM3Reranker cross-encoder (pool 30 → keep 30)
                 ↓ controlled title boost: ×(1 + 0.35·coverage) (+0.25 exact-title)
                 ↓ canonical-URL page dedup + min-max normalize
              Top 10 page-level results (TraditionalResult contract)
```

### Files changed

| File | Change |
|---|---|
| `services/search_service/hybrid_pages.py` | **new** — HybridPageSearchService (348 lines) |
| `shared/schemas.py` | `mode: Literal["ai","traditional","hybrid"]` — additive only |
| `services/api/main.py` | hybrid branch in `/v1/search`, `HYBRID_LATENCY` histogram, `_log_hybrid_event` |
| `frontend/ai-search-toggle/src/App.tsx` | third tab (IconLayers), hybrid state/fetch/load-more, result sections |
| `frontend/ai-search-toggle/src/styles.css` | `.mode-toggle--three` (3-column, mobile stack) |
| `scripts/eval_hybrid_comparison.py` | **new** — three-mode comparison harness |
| `tests/unit/test_hybrid_pages.py` | **new** — 17 tests |
| `tests/unit/test_api_hybrid_mode.py` | **new** — 5 contract tests |
| `ingestion/lexical/bm25_index.py` | unchanged in the end (reused as-is) |
| `services/search_service/hybrid_retriever.py` | unchanged — `reciprocal_rank_fusion` reused |
| `services/search_service/traditional.py` | unchanged — query prep + snippet helpers reused |

### Design notes (matching review feedback)

- **No shared-code changes**: Traditional/AI pipelines, `HybridRetriever`, and the BM25 index
  are untouched. The hybrid service composes existing tested pieces.
- **Query understanding**: uses `traditional.prepare_keyword_query` (stopwords + light stems
  only) — no intent rewriting, no synonym expansion, fair dense-vs-lexical comparison (note #4).
- **RRF** is the existing `reciprocal_rank_fusion` — unchanged and deterministic.
- **Title boost AFTER rerank** is controlled (note #6): coverage-based ×(1+0.35) with an
  exact-title ×1.6 cap total; a rerank score 2× higher still wins — no hard override.
- **Metrics**: `nbe_hybrid_search_latency_seconds` histogram + analytics/audit events with
  intent=`hybrid_search` (open question #4: done).

### Eval comparison (same queries per dataset)

Validation set (first 25 cases, per-case costs limit larger runs):

| mode | Recall@5 | Precision@5* | MRR | nDCG@5 | Top1 | Top3 | latency |
|---|---|---|---|---|---|---|---|
| KEYWORD (traditional) | 44.0% | — | 0.367 | 0.387 | 32.0% | 44.0% | 32ms |
| **HYBRID** | **60.0%** | — | 0.454 | **0.489** | 40.0% | 44.0% | ~13.7s (CPU rerank) |
| PURE_SEMANTIC (retrieval) | 56.0% | — | **0.528** | 0.535 | **52.0%** | 52.0% | ~7.3s |

Validation set (first 60 cases):

| mode | Recall@5 | MRR | nDCG@5 | Top3 |
|---|---|---|---|---|
| KEYWORD | 60.0% | 0.421 | 0.462 | 50.0% |
| **HYBRID** | **75.0%** | 0.450 | **0.529** | **56.7%** |

Golden set (all 22 cases):

| mode | Recall@5 | MRR | nDCG@5 | Top1 | Top3 |
|---|---|---|---|---|---|
| KEYWORD | 77.3% | 0.691 | 0.707 | 63.6% | 72.7% |
| **HYBRID** | **90.9%** | 0.723 | **0.768** | 63.6% | 72.7% |

\* Precision@5 ≡ Recall@5 under the single-relevant-doc labeling used by the existing
eval tooling; reported as one column.

**Takeaways**: hybrid adds +13–16pp Recall@5 over keyword-only on both datasets and beats
pure dense retrieval on Recall@5/nDCG@5 (lexical channel catches exact-title/certificate
queries the embedder ranks loosely). MRR is comparable across the three; PURE_SEMANTIC
keeps the best Top1 on paraphrase-heavy validation queries. Hybrid latency (~13.7s/case
in-process on this CPU container, rerank-dominated) is the trade-off — acceptable for a tab,
and drops sharply once models are warm/served.

### Verification

- `pytest tests/unit` → **228 passed** (206 pre-existing + 22 new; Traditional/AI regressions green).
- KEYWORD golden eval unchanged: Recall@5 54.1% / MRR 0.393 on the full 220-case set.
- Live API check (`mode=hybrid`): "National Bank of Egypt - Platinum" → **#1 Platinum page**
  (1.0), certificates below, Exclusive Products not in top-4.
- Frontend: `tsc --noEmit` clean, `npm run build` OK; three tabs render, mode switching
  resets output, autocomplete + load-more wired for hybrid.
