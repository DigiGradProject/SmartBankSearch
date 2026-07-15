# Migration Guide — Force Inject → Soft Boost

## Summary

Hard canonical URL injection (`score=0.99`, pin top-1) is removed. Enterprise banking safety rules now **only multiply scores of hybrid-retrieved candidates**. APIs (`POST /v1/search`, request/response schemas) are unchanged.

## Config migration

### Before

```env
FORCE_CANONICAL_INJECT=true
# Intent: regex-only (implicit)
```

### After

```env
RETRIEVAL_MODE=ENTERPRISE
BUSINESS_RULES_ENABLED=true
SEMANTIC_INTENT_ENABLED=true
SEMANTIC_INTENT_HIGH_CONFIDENCE=0.90
SEMANTIC_INTENT_REGEX_FALLBACK_THRESHOLD=0.60
SEMANTIC_INTENT_MIN_SCORE=0.42
```

### Evaluation

```env
RETRIEVAL_MODE=PURE_SEMANTIC
BUSINESS_RULES_ENABLED=false
```

Or override per call (eval scripts):

```python
service.retrieve(query, language, retrieval_mode="PURE_SEMANTIC")
service.retrieve(query, language, retrieval_mode="ENTERPRISE")
```

## Behavioral changes

1. **Retriever always wins the candidate set.** Rules cannot introduce a URL that hybrid search did not return.
2. **Intent is semantic-first** (BGE-M3 prototypes). Regex is fallback in the 0.60–0.90 band and below.
3. **Critical rules** (19623, forgot password) always run before semantic/regex.
4. **Decision engine** no longer pins `FORCE_CANONICAL`.
5. **Golden set** = training / reference. **validation_set.jsonl** = paraphrase hold-out (220 queries).

## Rollout checklist

1. Deploy with `RETRIEVAL_MODE=ENTERPRISE` (default).
2. Run unit/integration tests: soft boost + modes.
3. Run dual-mode eval:

```bash
.venv\Scripts\python.exe scripts\eval_retrieval_modes.py --golden --mode both --out data\logs\mode_compare_golden.json
.venv\Scripts\python.exe scripts\eval_retrieval_modes.py --mode both --out data\logs\mode_compare_validation.json
```

4. Compare `PURE_SEMANTIC` vs `ENTERPRISE` on Recall@5 / MRR / nDCG / Top1 / Top3 / latency.
5. Keep `FORCE_CANONICAL_INJECT` out of new `.env` files; treat as deprecated.

## Rollback

There is no safe rollback to hard inject (intentionally removed). To reduce enterprise influence without changing code:

```env
BUSINESS_RULES_ENABLED=false
# or
RETRIEVAL_MODE=PURE_SEMANTIC
```

Intent classifier and confidence gate remain available in ENTERPRISE when rules are re-enabled.
