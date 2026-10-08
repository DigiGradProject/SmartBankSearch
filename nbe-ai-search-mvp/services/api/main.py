from contextlib import asynccontextmanager

import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse, Response

from ingestion.embedding.vector_store import VectorStore
from ingestion.lexical.bm25_index import get_bm25_index
from ingestion.pipeline import RUNS, run_ingestion
from services.api.orchestrator import Orchestrator
from services.feedback.service import FeedbackService
from services.rag.analytics import AnalyticsEvent, write_analytics_event
from services.rag.audit import write_audit_event
from services.rag.metrics import FEEDBACK_TOTAL, TRADITIONAL_LATENCY
from services.search_service.autocomplete import build_autocomplete
from services.search_service.hybrid_pages import HybridPageSearchService
from services.search_service.traditional import TraditionalSearchService
from shared.config import settings
from shared.logging import configure_logging, get_logger
from shared.schemas import (
    AutocompleteResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthComponents,
    HealthResponse,
    IngestRequest,
    IngestRunStatus,
    SearchRequest,
    SearchResponse,
)

configure_logging()
logger = get_logger(__name__)

SEARCH_REQUESTS = Counter("nbe_search_requests_total", "Total search requests", ["answered", "mode"])
SEARCH_LATENCY = Histogram("nbe_search_latency_seconds", "Search request latency")
HYBRID_LATENCY = Histogram("nbe_hybrid_search_latency_seconds", "Hybrid search latency")
ABSTENTION_COUNT = Counter("nbe_search_abstentions_total", "Total abstentions", ["reason"])

orchestrator = Orchestrator()
vector_store = VectorStore()
feedback_service = FeedbackService()
traditional_search = TraditionalSearchService()
# Retrieval-only hybrid search (mode="hybrid"): shares the API's vector store
# and (via the singleton) the BM25 pickle with traditional search.
hybrid_page_search = HybridPageSearchService(vector_store=vector_store)


def _log_traditional_event(request: SearchRequest, outcome, total_ms: float) -> None:  # noqa: ANN001 — internal dataclass
    """Lightweight analytics + audit trail for traditional (keyword) searches."""
    import hashlib

    write_analytics_event(
        AnalyticsEvent(
            query=request.query,
            intent="traditional_search",
            rewritten_query="",
            total_ms=round(total_ms, 1),
            top_documents=[
                {"title": r.title, "url": r.url, "score": r.score, "category": r.category}
                for r in outcome.results[:5]
            ],
            confidence=1.0 if outcome.results else 0.0,
            confidence_reason="bm25_keyword_ranking",
            decision="KEYWORD_RESULTS" if outcome.results else "NO_RESULTS",
            language=outcome.language,
            answered=bool(outcome.results),
        )
    )
    if settings.audit_log_enabled:
        write_audit_event(
            {
                "query_hash": hashlib.sha256(request.query.strip().encode("utf-8")).hexdigest()[:16],
                "language": outcome.language,
                "intent": "traditional_search",
                "mode": "KEYWORD",
                "source_urls": [r.url for r in outcome.results[:10]],
                "result_count": outcome.total,
                "answered": bool(outcome.results),
                "abstention_reason": None if outcome.results else "no_keyword_matches",
                "models": {"embedding": "none", "reranker": "none", "llm": "none"},
            }
        )


def _log_hybrid_event(request: SearchRequest, outcome, total_ms: float) -> None:  # noqa: ANN001 — HybridPagesOutcome
    """Analytics + audit trail for hybrid (dense+BM25 fusion) searches."""
    import hashlib

    write_analytics_event(
        AnalyticsEvent(
            query=request.query,
            intent="hybrid_search",
            rewritten_query="",
            total_ms=round(total_ms, 1),
            top_documents=[
                {"title": r.title, "url": r.url, "score": r.score, "category": r.category}
                for r in outcome.results[:5]
            ],
            confidence=1.0 if outcome.results else 0.0,
            confidence_reason="hybrid_rrf_rerank_ranking",
            decision="HYBRID_RESULTS" if outcome.results else "NO_RESULTS",
            language=outcome.language,
            answered=bool(outcome.results),
        )
    )
    if settings.audit_log_enabled:
        write_audit_event(
            {
                "query_hash": hashlib.sha256(request.query.strip().encode("utf-8")).hexdigest()[:16],
                "language": outcome.language,
                "intent": "hybrid_search",
                "mode": "HYBRID",
                "source_urls": [r.url for r in outcome.results[:10]],
                "result_count": outcome.total,
                "answered": bool(outcome.results),
                "abstention_reason": None if outcome.results else "no_hybrid_matches",
                "models": {
                    "embedding": settings.embedding_model,
                    "reranker": settings.reranker_model if outcome.reranked else "fallback",
                    "llm": "none",
                },
            }
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("api_starting", chroma_chunks=vector_store.count())
    vector_store.embed_texts(["warmup"])
    yield


app = FastAPI(title="NBE AI Search MVP", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    llm_status = await orchestrator.llm_service.health()
    if not settings.bm25_enabled:
        # Deliberately disabled — do not even load the pickle.
        bm25_status = "degraded"
    else:
        bm25_index = get_bm25_index()
        bm25_status = "ok" if bm25_index.size > 0 else "down"
    components = HealthComponents(
        vector_db=vector_store.health(),  # type: ignore[arg-type]
        llm=llm_status,  # type: ignore[arg-type]
        api="ok",
        bm25=bm25_status,  # type: ignore[arg-type]
    )
    overall = "ok" if all(value != "down" for value in components.model_dump().values()) else "degraded"
    return HealthResponse(status=overall, components=components)  # type: ignore[arg-type]


@app.post("/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    if request.mode == "traditional":
        if not settings.keyword_search_enabled:
            raise HTTPException(status_code=404, detail="Traditional search is disabled")
        t0 = time.perf_counter()
        with TRADITIONAL_LATENCY.time():
            outcome = traditional_search.search(
                request.query, request.language, limit=request.limit, offset=request.offset
            )
        total_ms = (time.perf_counter() - t0) * 1000.0
        suggestions = build_autocomplete(
            request.query, request.language, limit=5, catalog_only=True
        )
        guidance = (
            None
            if outcome.results
            else (
                "لم نجد نتائج مطابقة. جرّب كلمات مفتاحية أقل أو مختلفة."
                if outcome.language == "ar"
                else "No matching pages found. Try fewer or different keywords."
            )
        )
        abstention = None
        if not outcome.results:
            abstention = (
                "keyword_index_unavailable"
                if not outcome.bm25_available
                else "no_keyword_matches"
            )
        response = SearchResponse(
            answer=None,
            confidence=1.0 if outcome.results else 0.0,
            citations=[],
            answered=bool(outcome.results),
            language=outcome.language,  # type: ignore[arg-type]
            abstention_reason=abstention,
            suggestions=suggestions,
            guidance=guidance,
            mode="traditional",
            results=outcome.results,
            total_results=outcome.total,
        )
        SEARCH_REQUESTS.labels(answered=str(response.answered).lower(), mode="traditional").inc()
        if abstention:
            ABSTENTION_COUNT.labels(reason=abstention).inc()
        _log_traditional_event(request, outcome, total_ms)
        return response

    if request.mode == "hybrid":
        t0 = time.perf_counter()
        with HYBRID_LATENCY.time():
            outcome = hybrid_page_search.search(
                request.query, request.language, limit=request.limit, offset=request.offset
            )
        total_ms = (time.perf_counter() - t0) * 1000.0
        suggestions = build_autocomplete(
            request.query, request.language, limit=5, catalog_only=False
        )
        guidance = (
            None
            if outcome.results
            else (
                "لم نجد نتائج مطابقة. جرّب كلمات مفتاحية أقل أو مختلفة."
                if outcome.language == "ar"
                else "No matching pages found. Try fewer or different keywords."
            )
        )
        abstention = None
        if not outcome.results:
            abstention = (
                "hybrid_index_unavailable"
                if not outcome.bm25_available and not outcome.dense_candidates
                else "no_hybrid_matches"
            )
        response = SearchResponse(
            answer=None,
            confidence=1.0 if outcome.results else 0.0,
            citations=[],
            answered=bool(outcome.results),
            language=outcome.language,  # type: ignore[arg-type]
            abstention_reason=abstention,
            suggestions=suggestions,
            guidance=guidance,
            mode="hybrid",
            results=outcome.results,
            total_results=outcome.total,
        )
        SEARCH_REQUESTS.labels(answered=str(response.answered).lower(), mode="hybrid").inc()
        if abstention:
            ABSTENTION_COUNT.labels(reason=abstention).inc()
        _log_hybrid_event(request, outcome, total_ms)
        return response

    with SEARCH_LATENCY.time():
        response = await orchestrator.search(request.query, request.language, debug=request.debug)
    SEARCH_REQUESTS.labels(answered=str(response.answered).lower(), mode="ai").inc()
    if not response.answered and response.abstention_reason:
        ABSTENTION_COUNT.labels(reason=response.abstention_reason).inc()
    return response


@app.post("/v1/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    result = feedback_service.submit(request)
    FEEDBACK_TOTAL.labels(vote=request.vote).inc()
    return result


@app.get("/v1/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(
    q: str = "",
    language: str = "auto",
    limit: int = 8,
    mode: str = "ai",
) -> AutocompleteResponse:
    from services.rag.language import detect_language

    resolved_language = detect_language(q, language)  # type: ignore[arg-type]
    catalog_only = mode == "traditional"
    suggestions = build_autocomplete(
        q, language, limit=limit, catalog_only=catalog_only
    )
    return AutocompleteResponse(query=q, language=resolved_language, suggestions=suggestions)  # type: ignore[arg-type]


@app.post("/v1/admin/ingest", response_model=IngestRunStatus)
async def trigger_ingest(request: IngestRequest) -> IngestRunStatus:
    return run_ingestion(source=request.source, limit=request.limit)


@app.get("/v1/admin/ingest/status/{run_id}", response_model=IngestRunStatus)
async def ingest_status(run_id: str) -> IngestRunStatus:
    status = RUNS.get(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="Ingestion run not found")
    return status


@app.get("/metrics")
async def metrics() -> Response:
    return PlainTextResponse(generate_latest().decode("utf-8"))


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "NBE AI Search MVP", "docs": "/docs"}
