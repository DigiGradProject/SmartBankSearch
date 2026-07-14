from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse, Response

from ingestion.embedding.vector_store import VectorStore
from ingestion.pipeline import RUNS, run_ingestion
from services.api.orchestrator import Orchestrator
from services.feedback.service import FeedbackService
from services.rag.metrics import FEEDBACK_TOTAL
from services.search_service.autocomplete import build_autocomplete
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

SEARCH_REQUESTS = Counter("nbe_search_requests_total", "Total search requests", ["answered"])
SEARCH_LATENCY = Histogram("nbe_search_latency_seconds", "Search request latency")
ABSTENTION_COUNT = Counter("nbe_search_abstentions_total", "Total abstentions", ["reason"])

orchestrator = Orchestrator()
vector_store = VectorStore()
feedback_service = FeedbackService()


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
    components = HealthComponents(
        vector_db=vector_store.health(),  # type: ignore[arg-type]
        llm=llm_status,  # type: ignore[arg-type]
        api="ok",
    )
    overall = "ok" if all(value != "down" for value in components.model_dump().values()) else "degraded"
    return HealthResponse(status=overall, components=components)  # type: ignore[arg-type]


@app.post("/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    with SEARCH_LATENCY.time():
        response = await orchestrator.search(request.query, request.language, debug=request.debug)
    SEARCH_REQUESTS.labels(answered=str(response.answered).lower()).inc()
    if not response.answered and response.abstention_reason:
        ABSTENTION_COUNT.labels(reason=response.abstention_reason).inc()
    return response


@app.post("/v1/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    result = feedback_service.submit(request)
    FEEDBACK_TOTAL.labels(vote=request.vote).inc()
    return result


@app.get("/v1/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(q: str = "", language: str = "auto", limit: int = 8) -> AutocompleteResponse:
    from services.rag.language import detect_language

    resolved_language = detect_language(q, language)  # type: ignore[arg-type]
    suggestions = build_autocomplete(q, language, limit=limit)
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
