# Layered Intent Classification

## Principle

**Regex alone overfits the golden set.** Production intent uses a layered decision stack:

```
User Query
    ↓
Critical Rules (contact, password)     ← always first, deterministic
    ↓
BGE-M3 Semantic Intent               ← primary (generalization)
    ↓
Confidence routing
    ├─ ≥ 0.90  → semantic route
    ├─ 0.60–0.90 → regex if match, else semantic
    └─ < 0.60  → regex if match, else general_faq (broad hybrid)
    ↓
Hybrid Retrieval → Reranker → Business Rules (soft, last)
```

## Why not delete Regex?

Banking requires deterministic routes for:

- `19623` / customer service
- `نسيت الباسورد` / password reset

Regex is **fallback + audit trail**, not the primary brain.

## Configuration

```env
SEMANTIC_INTENT_ENABLED=true
SEMANTIC_INTENT_HIGH_CONFIDENCE=0.90
SEMANTIC_INTENT_REGEX_FALLBACK_THRESHOLD=0.60
SEMANTIC_INTENT_MIN_SCORE=0.42
```

Legacy (regex-only):

```env
SEMANTIC_INTENT_ENABLED=false
```

## Evaluation

Intent accuracy in eval scripts now uses `classify_with_fallback` (layered), not raw `classify_query`.

Measure generalization on `validation_set.jsonl` (220 paraphrases), not `golden_set.jsonl` (22 reference queries).

## Modules

| Module | Role |
|--------|------|
| `critical_rules.py` | Deterministic safety routes |
| `semantic_intent.py` | BGE-M3 prototype similarity |
| `intent_prototypes.py` | Paraphrase-rich intent vectors |
| `intent_fallback.py` | Layered orchestration |
| `intent_classifier.py` (regex) | Fallback patterns only |
