# PURE_SEMANTIC vs ENTERPRISE — Performance Comparison

## How to run

Requires local indexes (`data/chroma`, `data/bm25`) and models loaded.

```bash
# Reference / training set
.venv\Scripts\python.exe scripts\eval_retrieval_modes.py --golden --mode both --out data\logs\compare_golden.json

# Hold-out paraphrases (validation_set.jsonl)
.venv\Scripts\python.exe scripts\eval_retrieval_modes.py --mode both --out data\logs\compare_validation.json
```

## Metrics reported

| Metric | Meaning |
|--------|---------|
| Recall@5 | Expected URL substring appears in top-5 |
| MRR | Reciprocal rank of first relevant URL |
| nDCG@5 | Rank-sensitive discounted gain |
| Intent Accuracy | Rule classifier vs labeled intent |
| Top1 / Top3 | Hit rates |
| Confidence Accuracy | Calibration proxy vs relevance |
| Latency mean / p95 | End-to-end `retrieve()` wall time |

## Expected qualitative deltas

| Signal | PURE_SEMANTIC | ENTERPRISE |
|--------|---------------|------------|
| Independence | Measures true hybrid + reranker quality | Adds banking safety soft boosts |
| High-value intents (FX, loans, cards) | May surface related-but-wrong family pages | Preferred category / metadata weights improve precision |
| Paraphrase hold-out | Baseline semantic generalization | Soft rules help **only when intent fires**; otherwise ≈ semantic |
| Latency | Slightly lower (no rule passes / filters) | Modest overhead from filter + soft scoring |
| Overfitting risk | Low | Controlled — rules cannot force a non-retrieved URL |

## Interpreting results

1. **If PURE_SEMANTIC Top3 is strong** on validation → embeddings + BM25 + reranker are healthy.
2. **ENTERPRISE ΔTop1 / ΔMRR > 0** on banking intents → safety layer adds precision without retrieval bypass.
3. **If ENTERPRISE wins golden but loses validation** → residual rule overfitting; tighten weights or preferred categories, do **not** reintroduce inject.
4. Prefer shipping when validation ENTERPRISE ≥ PURE_SEMANTIC on MRR/nDCG for labeled banking intents, and PURE_SEMANTIC alone remains reportable for audits.

## Measured results (golden_set.jsonl, n=22)

Run date: 2026-07-14. Full JSON: `data/logs/compare_golden.json`.

| Mode | Recall@5 | MRR | Top1 | Top3 | Intent | ConfAcc | Latency mean* |
|------|----------|-----|------|------|--------|---------|---------------|
| PURE_SEMANTIC | 81.8% | 0.705 | 63.6% | 72.7% | 100% | 95.5% | ~17.3s (cold start) |
| ENTERPRISE | 100% | 1.000 | 100% | 100% | 100% | 100% | ~5.5s (warm models) |
| Δ (E − P) | +18.2pp | +0.30 | +36.4pp | +27.3pp | 0 | +4.5pp | (not comparable*) |

\*PURE_SEMANTIC ran first and paid embedding/reranker load cost; ENTERPRISE reused warm models. Re-run both with a warm process for fair latency (or discard the first query).

### Reading

1. **PURE_SEMANTIC is independently measurable** — hybrid + reranker alone recover the expected URL in top-5 for ~82% of reference queries with **zero** business-rule injection.
2. **ENTERPRISE improves precision** via soft category/metadata boosts and staged filters — **without** force-injecting documents.
3. Golden 100% ENTERPRISE is an upper bound on the reference set (intents fire cleanly). Use `validation_set.jsonl` (220 paraphrases) for generalization:

```bash
.venv\Scripts\python.exe scripts\eval_retrieval_modes.py --mode both --out data\logs\compare_validation.json
```

nDCG values historically could exceed 1.0 when multiple URL substring hits occurred; the metric helper now caps at 1.0 — re-run after pull for corrected nDCG.
