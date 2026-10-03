"""One-off diagnostic: semantic intent scores for Al Ahly brand queries."""
import sys

sys.path.insert(0, ".")

from services.rag.semantic_intent import _cosine, _prototype_index  # noqa: E402
from ingestion.embedding.bge_m3 import get_embedder  # noqa: E402

QUERIES = [
    "What is National Bank of Egypt - Al Ahly Points?",
    "What is National Bank of Egypt - Al Ahly Business?",
    "What is National Bank of Egypt - Al Ahly Net - Platinum?",
    "alahly points",
    "alahly business",
    "alahlynet platinum",
]

labels, vectors = _prototype_index()
queries_vec = get_embedder().embed_dense(QUERIES)

for q, qv in zip(QUERIES, queries_vec):
    per_intent: dict[str, float] = {}
    for label, pv in zip(labels, vectors):
        score = _cosine(qv, pv)
        per_intent[label] = max(per_intent.get(label, 0.0), score)
    top3 = sorted(per_intent.items(), key=lambda kv: kv[1], reverse=True)[:3]
    print(f"\nQUERY: {q}")
    for label, score in top3:
        print(f"  {label:20s} {score:.3f}")
