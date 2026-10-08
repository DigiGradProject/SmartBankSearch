# Plan: Hybrid Search tab (dense + BM25, RRF-fused, no LLM)

Status: **DRAFT — awaiting review**
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
