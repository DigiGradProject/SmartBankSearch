# CPU / Quantized Deployment Profile (Phase 4)

## When to use

Branch or low-GPU environments where VRAM &lt; 8 GB.

## Recommended stack (CPU)

| Component | Setting |
|-----------|---------|
| Embeddings | `BAAI/bge-m3` FP32/CPU or ONNX quantized |
| Reranker | `bge-reranker-v2-m3` CPU; reduce pool to 12 if needed |
| LLM | Ollama `qwen3:8b` Q4_K_M or `qwen3:4b` |
| Intent MiniLM | Disabled by default (`MINILM_INTENT_FALLBACK_ENABLED=false`) |
| GLiNER | Disabled |
| LLM rewrite | Disabled |

## Expected latency (indicative)

| Stage | CPU estimate |
|-------|----------------|
| Understanding | &lt; 20 ms |
| Hybrid retrieve | 80–250 ms |
| Rerank 20 | 150–400 ms |
| LLM 8B Q4 | 1.5–4 s |
| **Total** | **~2–5 s** |

Do **not** advertise &lt;500 ms public SLA on CPU-only.

## Env flags

```
RERANK_POOL_SIZE=12
RERANK_KEEP_SIZE=5
MINILM_INTENT_FALLBACK_ENABLED=false
GLINER_ENABLED=false
LLM_REWRITE_ENABLED=false
OLLAMA_MODEL=qwen3:8b
```
