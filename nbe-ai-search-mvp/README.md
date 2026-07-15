# NBE AI Search MVP

On-premises Retrieval-Augmented Generation (RAG) search platform for National Bank of Egypt website content. Implements the architecture defined in `plan-mvp.md`.

## Features

- Batch ingestion from existing scraped JSON (`nbe_complete_scrape`)
- Arabic + English normalization and structure-aware chunking
- Embeddings: **BGE-M3** (dense + sparse hybrid retrieval)
- Reranker: **bge-reranker-v2-m3**
- LLM Tier 1: **Qwen3-8B** via Ollama; Tier 2: **Qwen3-14B** for deeper reasoning
- Idempotent upsert via `content_hash`
- Semantic search API with confidence gate and abstention
- Extractive fallback when Ollama is unavailable
- React frontend with AI Search / Traditional Search toggle
- Prometheus metrics at `/metrics`

## Quick Start (Local)

### 1. Install backend dependencies

```bash
cd nbe-ai-search-mvp
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Export documents (optional)

```bash
python scripts/export_documents.py --limit 50
```

### 3. Run ingestion

```bash
set PYTHONPATH=.
python scripts\run_ingest.py --source merged
```

First run downloads **BGE-M3** and (on first search) **bge-reranker-v2-m3**.

> Important: after switching from MiniLM to BGE-M3 you must re-ingest into the new collection `nbe_chunks_bge_m3`.

### 4. Pull LLM models (Ollama)

```bash
ollama pull qwen3:8b
ollama pull qwen3:14b
```

Optional deeper MoE tier: `ollama pull qwen3:30b` then set `OLLAMA_MODEL_TIER2=qwen3:30b`.

### 5. Start API

```bash
set PYTHONPATH=.
uvicorn services.api.main:app --reload --port 7000
```

### 6. Start frontend

```bash
cd frontend/ai-search-toggle
npm install
npm run dev
```

Open http://localhost:5173

Without Ollama, the API uses an extractive fallback grounded in retrieved chunks.

## API

- `POST /v1/search` — semantic search
- `GET /v1/health` — component health
- `POST /v1/admin/ingest` — trigger batch ingestion
- `GET /v1/admin/ingest/status/{runId}` — ingestion status

## Project Layout

Matches `plan-mvp.md` Section 13:

- `ingestion/` — document processing, chunking, embedding
- `services/` — search, context builder, LLM, REST API
- `frontend/ai-search-toggle/` — UI toggle
- `data/schemas/` — JSON contracts
- `infra/docker/` — Docker Compose stack
- `tests/` — unit tests

## Docker Compose

```bash
cd infra/docker
docker compose up --build
```

## Tests

```bash
set PYTHONPATH=.
pytest tests/unit -q
```

## Configuration

Copy `.env.example` to `.env` and adjust as needed.

## Model Stack

| Component | Model | License / notes |
|-----------|--------|-----------------|
| Embedding | `BAAI/bge-m3` | MIT, hybrid dense+sparse, strong Arabic |
| Reranker | `BAAI/bge-reranker-v2-m3` | pairs with BGE-M3 |
| LLM Tier 1 | `qwen3:8b` | Apache 2.0, fast Arabic/colloquial |
| LLM Tier 2 | `qwen3:14b` (or `qwen3:30b`) | deeper reasoning on complex queries |

## Notes

- MVP uses ChromaDB (file-based) instead of Milvus for faster local setup; swap via `ingestion/embedding/vector_store.py` when scaling.
- Live data (exchange rates, branch locator) is intentionally excluded per ADR-01.
- Re-run ingestion safely; unchanged chunks are skipped by `content_hash`.
- Collection `nbe_chunks_bge_m3` is separate from the old MiniLM index.
