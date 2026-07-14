# Open-Source License Pack (Phase 4 — Legal review)

| Component | Package / Model | License (verify at release) |
|-----------|-----------------|-----------------------------|
| Language detection | lingua-language-detector | Apache-2.0 |
| Embeddings | BAAI/bge-m3 | MIT |
| Reranker | BAAI/bge-reranker-v2-m3 | MIT |
| Intent fallback | paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 |
| Optional NER | GLiNER | Apache-2.0 |
| LLM | Qwen3 (Ollama) | Tongyi Qianwen / Apache-style — **Legal must confirm** |
| Vector DB | ChromaDB | Apache-2.0 |
| API | FastAPI / Pydantic | MIT |
| Lexical | rank_bm25 | Apache-2.0 |

**Policy:** No runtime calls to OpenAI/Anthropic/Google. All models hosted on bank premises.

**Action:** Legal sign-off before production cutover.
