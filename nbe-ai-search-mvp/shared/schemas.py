from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    path: str | None = None
    source_folder: str | None = None
    block_count: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    id: str
    title: str
    url: str
    language: Literal["ar", "en"]
    content: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    title: str
    url: str
    language: Literal["ar", "en"]
    text: str
    content_hash: str
    doc_type: str = "general"
    category: str = "general"
    is_stub: bool = False
    canonical_url_slug: str = ""
    quality_score: float = 0.7


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    language: Literal["ar", "en", "auto"] = "auto"


class AutocompleteResponse(BaseModel):
    query: str
    language: Literal["ar", "en"]
    suggestions: list[SearchSuggestion] = Field(default_factory=list)


class Citation(BaseModel):
    title: str
    url: str


class SearchSuggestion(BaseModel):
    query: str
    label: str
    url: str | None = None
    reason: str | None = None
    score: float | None = None


class SearchResponse(BaseModel):
    answer: str | None
    confidence: float
    citations: list[Citation]
    answered: bool
    language: Literal["ar", "en"] | None = None
    abstention_reason: str | None = None
    suggestions: list[SearchSuggestion] = Field(default_factory=list)
    guidance: str | None = None


class HealthComponents(BaseModel):
    vector_db: Literal["ok", "degraded", "down"]
    llm: Literal["ok", "degraded", "down"]
    api: Literal["ok", "degraded", "down"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: HealthComponents


class IngestRequest(BaseModel):
    source: Literal["documents_json", "scrape", "cleaned_jsonl"] = "cleaned_jsonl"
    limit: int | None = None


class IngestRunStatus(BaseModel):
    run_id: str
    status: Literal["pending", "running", "completed", "failed"]
    started_at: datetime
    finished_at: datetime | None = None
    documents_processed: int = 0
    chunks_upserted: int = 0
    chunks_skipped: int = 0
    errors: list[str] = Field(default_factory=list)
