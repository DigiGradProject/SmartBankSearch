# Plan: Hybrid Search tab (dense + BM25, RRF-fused) → Hybrid RAG (Phase 2)

Status: **IMPLEMENTED on branch `feat/hybrid-search-tab`; Phase 2 (grounded LLM answer) below**
Branch target: new branch `feat/hybrid-search-tab` off current `main`
Owner: Buffy + user review

---

## 1. Goal

Add a third search-mode tab, **Hybrid Search**, next to "AI Search Mode" and "Traditional Search".
Phase 1 shipped the retrieval pipeline — BGE-M3 dense vector results **fused with BM25 lexical
results via Reciprocal Rank Fusion (RRF)**, then the cross-encoder reranker — as a ranked list of
pages. **Phase 2 (Hybrid RAG) adds a grounded LLM answer with citations built from the top ranked
results, displayed first, followed by the unchanged ranked page list.**

- ✅ **Included**: hybrid retrieval (dense + lexical), RRF fusion, reranker, per-page dedup,
  snippets, scores, language tags, pagination ("Load more"), autocomplete dropdown,
  Arabic/English rendering, **grounded LLM answer + citations (Phase 2)**.
- ❌ **Excluded** (Phase 2 still excluded): AI-mode intent business-rule boosts, AI-mode
  confidence/decision gate, AI-mode query expansion/planning, exact + semantic caching.
  Retrieval ordering is untouched by generation — the LLM never reorders pages.

The modes after Phase 2:

| | Traditional | **Hybrid** | AI |
|---|---|---|---|
| Retrieval | BM25 only | **dense + BM25 (RRF)** | dense + BM25 (RRF) + rules/gate |
| Reranker | no | **yes** | yes |
| Business rules / gate | no | **no** | yes |
| LLM answer | no | **yes (grounded in top pages, citations)** | yes |
| Output | page list | **answer + citations first, then page list** | answer + citations |

---

## 1b. Phase 2 — Hybrid RAG design (final architecture, inspected implementation)

Verified against the actual code before writing this: `services/api/orchestrator.py`
(`Orchestrator.search` → `ContextBuilder.build` → `LLMService.generate_answer` →
`wrap_plain_answer` → `evaluate_answer`), `services/llm_service/llm.py` (grounding system
prompt, `NO_ANSWER` refusal detection, tier escalation, graceful `None` on error),
`services/context_builder/builder.py` (built-context + citation selection scoped to retrieved
chunks), `shared/schemas.py` (`SearchResponse` already carries `answer/citations/abstention_reason`).

**Contract principles (user requirements #2–#7):**

1. **Three modes stay intact.** Traditional and AI handlers are untouched; hybrid gets answer
   generation added inside its own branch/service — no routing to `mode=ai`.
2. **Retrieval pipeline unchanged**: BM25 top-50 + dense top-50 → RRF → reranker → title boost →
   page dedup → normalize. The final ranked `pages` list is *frozen* before generation; the
   `results` list in the response is built from that frozen order.
3. **Grounded generation reuses AI-mode building blocks**, in its own thin layer:
   - `ContextBuilder.build(query, top_chunks)` → `BuiltContext{context_text, citations}` —
     context text + citation list assembled only from retrieved chunks (existing junk/URL
     filtering and citation floors apply), `compress=False` (chunks already rerank-picked).
   - `LLMService.generate_answer(query, context_text, language)` → `(answer, confidence)`;
     it internally detects refusals/`NO_ANSWER` and returns `None` on failure/insufficient
     evidence (with its keyword-overlap `_fallback_answer`).
   - `wrap_plain_answer(answer, language, citations, intent="hybrid_search")` for the
     structured `### Answer / ### Key Information / ### Source` shape AI mode emits.
   - NOT reused (Phase 1 decision stands): AI-mode caches (exact + semantic), plan_query sub
   -query expansion, intent classifier, business rules, catalog answer builders, self-eval
     regeneration loop. Hybrid keeps one LLM call; a regeneration loop would double latency.
4. **Language handling**: reuse `prepare_keyword_query` resolution + LLM respond-in-user-language
   prompt rule; answer `dir` follows `result.language`/`isArabic` in the frontend (existing).
5. **Grounding / abstention**: if the context is empty → `abstention_reason="insufficient_context"`;
   if `generate_answer` yields no answer → return pages with the retrieved-pages list intact
   (graceful degradation), still `answered` pages-wise. **If the LLM call throws/times out →
   the answer is simply `None`; the request never fails** — pages are returned normally.
6. **Latency bookkeeping**: retrieval phase (existing hybrid service timing) vs. generation
   phase timed separately; result logged: `hybrid_search_completed` gains `retrieval_ms`,
   `generation_ms`, `total_ms`; top-level `pages` unaffected.
7. **API contract (additive, backward compatible)**: `SearchResponse` for hybrid may now carry
   `answer` (string, or null), `citations` (ContextBuilder citations — URLs constrained to the
   retrieved set by construction), `answered` (true only when answer present),
   `abstention_reason` (`insufficient_context`/`no_hybrid_matches`/
   `hybrid_index_unavailable`); `confidence` stays for answer confidence when answered, else
   pages-present flag. AI/traditional responses unchanged.
8. **Frontend (Hybrid tab)**: render an **answer card first** — reuse the AI-mode answer+styles
   (`result.answer`, `citations` mapping, `confidence pill`) as a shared JSX block owned by the
   hybrid section, then the existing ranked `trad-card` list *below*, order exactly as returned
   by the API.

**Files changed (Phase 2):**

| File | Change |
|---|---|
| `services/search_service/hybrid_pages.py` | `HybridPagesOutcome` gains `answer/citations/generation_error` + `generate_answer()` step (context build + LLM call, failure-tolerant) |
| `services/api/main.py` | hybrid branch forwards answer/citations/confidence/answered; analytics/audit gains `llm_ms` |
| `shared/schemas.py` | comment-only change (`hybrid` description gains "LLM answer?") |
| `frontend/ai-search-toggle/src/App.tsx` | hybrid section renders answer card (answer + citations), then ranked list; state gains `hybridAnswer` |
| `tests/unit/test_hybrid_pages.py` | new tests — grounded citation URLs ⊆ retrieved, insufficient-evidence abstention, LLM-failure fallback |
| `tests/unit/test_api_hybrid_mode.py` | contract test for answer+citations present; regressions for ai/traditional untouched |

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

---

## 10. Phase 2 implementation results (2026-10-09) — Hybrid RAG

### What changed (verbatim per §1b plan)

| File | Change |
|---|---|
| `services/search_service/hybrid_pages.py` | `HybridPagesOutcome` gains `answer/citations/answer_confidence/generation_error/retrieval_ms/generation_ms/total_ms` + internal `context_chunks`; new `search_with_answer()` (frozen ranking → grounded context from top `retrieval_top_k` chunks → existing `LLMService.generate_answer` → `wrap_plain_answer`); ctor seams `context_builder=`/`llm_service=`. Generation is fail-open: any exception → `generation_error=True`, pages returned. `search()` contract unchanged (Phase 1 tests pass untouched). |
| `services/api/main.py` | hybrid branch calls `search_with_answer`; response now carries `answer`/`citations`/`confidence` (LLM answer confidence when answered); `abstention_reason="insufficient_context"` when pages exist but the LLM abstained/failed; `_log_hybrid_event` logs retrieval vs generation vs total latency, `HYBRID_RAG_ANSWER` decision, and `llm` model fields. |
| `frontend/ai-search-toggle/src/App.tsx` | New `hybridAnswer/hybridCitations/hybridConfidence` state; answer card (kicker + confidence pill + answer + Sources citations) rendered FIRST, ranked pages below with a "ranked by the hybrid retrieval engine" note; state resets on mode switch / new search. |
| `frontend/ai-search-toggle/src/styles.css` | `.hybrid-answer` pill tweak + `.trad-rank-note`. |
| `shared/schemas.py` | comment-only (`mode="hybrid"` doc updated). |
| `tests/unit/test_hybrid_pages.py` | +7 RAG tests: frozen-ranking + answer wrapper, citations ⊆ retrieved URLs, empty-context abstention (LLM untouched), LLM-throw fail-open, LLM-refusal abstention, pagination skips generation, plain-`search()` regression. |
| `tests/unit/test_api_hybrid_mode.py` | e2e test updated: `answer` is no longer asserted `None`; grounded citations must be ⊆ retrieved result URLs when an answer is present; pages-only contract asserted when the live LLM is unavailable. |

### NOT inherited from AI mode (per requirement #6 — verified)

- No `plan_query` sub-query expansion, no intent classification, no business rules, no
  catalog answer builders, no exact/semantic caching, no self-eval regeneration loop
  (single LLM call keeps hybrid latency bounded). Grounding + `NO_ANSWER` abstention
  come from the shared `LLMService` prompt contract; citation floors/junk filtering
  come from the shared `ContextBuilder`.

### Grounding contract (test-enforced)

- Citations can only be built from chunks in the frozen pool (ContextBuilder input),
  asserted ⊆ retrieved page URLs.
- Empty grounded context → abstain (`insufficient_context`), pages untouched.
- LLM refusal/`NO_ANSWER` → abstain, pages untouched, no error state.
- LLM client failure (timeout/500) → `generation_error=True`, HTTP 200 with pages only.
- Answer text wrapped via shared `wrap_plain_answer` (### Answer / Key Information / Source).
- Latency: `retrieval_ms` (Phase-1 pipeline), `generation_ms` (LLM), `total_ms` (sum) —
  logged in `hybrid_search_with_answer_completed` and audit `latency_ms`.

### Verification (2026-10-09)

- `pytest tests/unit` → **235 passed** (228 pre-existing + 7 new; live e2e included
  a REAL grounded answer for "platinum card": structured ### sections + citation,
  retrieval_ms 46.8s cold vs generation_ms 25.2s on CPU).
- Frontend: `tsc --noEmit` clean, `npm run build` OK (new bundle)
- HYBRID retrieval regression eval on 5 golden cases: Recall@5 100%, MRR/nDCG@5 1.000,
  identical to Phase-1 HYBRID on the same subset (retrieval ordering untouched by
  generation — the eval calls the ranking-only path, and the live e2e asserted the
  same page list with and without an answer).
