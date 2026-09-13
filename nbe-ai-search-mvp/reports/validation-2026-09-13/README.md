# Validation experiments and implementation verification

Run date: 2026-09-13. Commit: `607cb249b6bb716ec210211b25651135bf2d24c1`.

Both requested experiments completed on the unchanged 220-query validation set (440 scored retrieval calls). These are newly measured values; the older 22-query golden-set figures were not reused.

## Full validation set: existing evaluator definitions (n=220 per mode)

| Metric | PURE_SEMANTIC | ENTERPRISE | E − P |
|---|---:|---:|---:|
| Top-1 hit rate | 52.73% | 80.00% | +27.27 pp |
| Top-3 hit rate | 64.09% | 80.45% | +16.36 pp |
| Hit@5 (script: Recall@5) | 71.36% | 82.27% | +10.91 pp |
| MRR@5 | 0.5903 | 0.8066 | +0.2163 |
| nDCG@5 (implementation) | 0.6134 | 0.8067 | +0.1933 |
| Intent accuracy | 85.00% | 85.00% | +0.00 pp |
| Confidence proxy | 75.45% | 87.27% | +11.82 pp |
| Mean retrieval latency (ms) | 3071.7 | 2820.7 | -251.0 |
| p95 retrieval latency (ms) | 3330.0 | 3436.8 | +106.7 |

## URL-labeled cases only (n=210 per mode)

The existing evaluator automatically assigns 1.0 to all ranking metrics for ten cases whose expected URL is empty, regardless of their retrieved results. The following view excludes those cases without changing queries, models, ranking, or judgments.

| Metric | PURE_SEMANTIC | ENTERPRISE | E − P |
|---|---:|---:|---:|
| Top-1 hit rate | 50.48% | 79.05% | +28.57 pp |
| Top-3 hit rate | 62.38% | 79.52% | +17.14 pp |
| Hit@5 (script: Recall@5) | 70.00% | 81.43% | +11.43 pp |
| MRR@5 | 0.5708 | 0.7974 | +0.2266 |
| nDCG@5 (implementation) | 0.5950 | 0.7975 | +0.2025 |
| Intent accuracy | 84.29% | 84.29% | +0.00 pp |
| Confidence proxy | 74.29% | 86.67% | +12.38 pp |
| Mean retrieval latency (ms) | 3070.0 | 2829.8 | -240.2 |
| p95 retrieval latency (ms) | 3329.5 | 3437.0 | +107.5 |

## Verified implementation findings

- BM25 is implemented using `rank_bm25.BM25Okapi` in `ingestion/lexical/bm25_index.py`. Hybrid fusion is in `services/search_service/hybrid_retriever.py`, called from `search.py`; RRF uses k=60 and merges chunk IDs. The vector branch also blends BGE-M3 dense and sparse scores before RRF.
- `PURE_SEMANTIC` retains BM25, query expansion, keyword ranking, document quality filtering, and the cross-encoder. It is a hybrid baseline without enterprise controls, not a dense-only ablation.
- `RETRY_RELATED` exists and is covered by a unit test. It sets `should_answer=False`; the API then abstains. No post-decision retrieval retry is executed. Actual related-type/broad retrieval occurs earlier inside `HybridRetriever.retrieve()` based on candidate sufficiency. Generation can independently retry once after an unsupported-answer judgment.
- The validation file contains 220 distinct questions: 127 Arabic and 93 English across 15 intent labels. No question contains both Arabic and Latin characters. Thus this set provides no direct Arabic–English mixed-script code-switching evaluation. Separate Arabic and English questions are not evidence of code-switching robustness.
- There are no exact question overlaps with the golden set, but validation is largely generated paraphrases. Independence from the reference set and real-user generalization are not established. Exchange-rate and personal-loan queries account for 142/220 cases (64.55%). Ten cases have no expected URL; 43 specify a forbidden URL.

## Decision coverage and configured acceptance gate

Observed retrieval decisions were ENTERPRISE: 214 `ANSWER`, 6 `NO_ANSWER`; PURE_SEMANTIC: 220 `ANSWER`. No `RETRY_RELATED` event occurred in these 440 measured calls. Its evidence is therefore source inspection and the passing targeted unit test, not observed validation coverage. `ANSWER` here denotes retrieval eligibility, not an LLM-generated answer. There were zero forbidden-URL violations in ENTERPRISE and one in PURE_SEMANTIC.

Applying the existing ENTERPRISE gate thresholds to this run would fail: intent accuracy is 85.00%, below the required 90.00%, although Top-3 is 80.45%, above the required 70.00%. The inference runner was not invoked with `--gate`; this conclusion is calculated from its unchanged gate expression and the recorded results.

## Metric and scope limitations

`Recall@5` is implemented as any relevant URL substring appearing in five results, so it is Hit@5 rather than recall over all relevant corpus documents. MRR is limited to the returned five results. nDCG constructs its ideal ordering from the positives retrieved in that same window; it is not based on exhaustive corpus relevance judgments. Results are chunk-based, so multiple chunks sharing one URL can contribute to nDCG.

The confidence metric is a heuristic proxy, not measured calibration or factual correctness. For a top-three URL hit it accepts confidence >= 0.357. On a miss, PURE_SEMANTIC passes automatically; ENTERPRISE passes if it abstains or confidence is below 0.57. Forbidden-URL violations force a failure. This asymmetric definition prevents treating the two confidence values as a fair calibration comparison. The two `faq` labels also differ from the classifier's `general_faq` label; original labels were preserved.

Latency measures `SearchService.retrieve()` only. The evaluator separately classifies the original question before starting the timer; query understanding inside retrieval classifies the normalized question again. The reported intent accuracy therefore refers to the evaluator's separate classifier call. API planning, answer construction, LLM generation, faithfulness evaluation, and network response time are outside these experiments.

## Reproducibility and artifacts

Both modes were warmed with an excluded probe; their execution order alternated for successive questions. Models, corpus, and query order were held fixed. GPU 0 (NVIDIA A16) was selected because GPU 1 was unavailable. This is one sequential run per mode, with no statistical significance claim. Some device memory was occupied by an existing process, so latency should be treated as environment-specific.

- `run.py`: executable experiment wrapper, calling the existing evaluator and metric functions without modifying production code.
- `metadata.json`: effective settings, collection and BM25 sizes, backends, commit, dataset hash, and runtime protocol.
- `per-query.jsonl` and `per-query.csv`: all 440 query/mode rows (CSV uses UTF-8 BOM for Arabic spreadsheet compatibility).
- `comparison.json`: original-definition aggregate numbers.
- `breakdowns.json`: URL-labeled, language, intent, answer-count, and forbidden-URL breakdowns.
- `dataset-audit.json`: dataset counts and coverage audit.
- `code-verification.md`: full requested code, BM25/fusion implementation, and the first 20 dataset rows.
- `run.log`: execution evidence; checked for embedding/reranker fallbacks and CUDA errors.
- `verification-tests.log`: 13 existing targeted tests passed (fusion, decision controls, and metrics).
- `../../docs/section-iii-framework.md`: approximately 1.5-page English framework draft (837 whitespace-delimited words; rendered length depends on manuscript format).

Reproduce from `nbe-ai-search-mvp` (the wrapper overwrites its report files):

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false .venv/bin/python -u reports/validation-2026-09-13/run.py > reports/validation-2026-09-13/run.log 2>&1
.venv/bin/python reports/validation-2026-09-13/summarize.py
```
