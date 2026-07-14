"""Enterprise RAG pipeline — deterministic understanding → hybrid → rerank → LLM."""

from __future__ import annotations

from dataclasses import dataclass

from ingestion.embedding.vector_store import RetrievedChunk
from services.rag.confidence import compute_confidence
from services.rag.entity_extractor import EntityExtractionResult
from services.rag.language import detect_language
from services.rag.query_rewrite import rewrite_query
from services.rag.response_formatter import wrap_plain_answer
from services.search_service.intent_classifier import classify_query
from services.search_service.search import RetrievalResult, SearchService
from shared.schemas import Citation, SearchResponse


@dataclass
class PipelineTrace:
    language: str
    intent: str
    rewritten_query: str
    entities: EntityExtractionResult
    used_llm_rewrite: bool


class EnterpriseSearchPipeline:
    """Dependency-injectable pipeline façade for FastAPI."""

    def __init__(self, search_service: SearchService | None = None) -> None:
        self.search_service = search_service or SearchService()

    def understand(self, query: str, language: str = "auto") -> PipelineTrace:
        resolved = detect_language(query, language)
        intent = classify_query(query if resolved == "en" else query, resolved)
        expand = intent.expand_ar if resolved == "ar" else intent.expand_en
        rewritten = rewrite_query(query, resolved, intent_expand=expand)
        return PipelineTrace(
            language=resolved,
            intent=intent.intent,
            rewritten_query=rewritten.rewritten,
            entities=rewritten.entities,
            used_llm_rewrite=rewritten.used_llm,
        )

    def retrieve(self, query: str, language: str = "auto") -> tuple[RetrievalResult, PipelineTrace]:
        trace = self.understand(query, language)
        # SearchService still owns hybrid+rerank; understanding modules are composed above for tracing.
        result = self.search_service.retrieve(query, language)
        return result, trace

    def format_answer(
        self,
        answer: str,
        *,
        language: str,
        citations: list[Citation],
        intent: str,
        chunks: list[RetrievedChunk],
        llm_confidence: float,
    ) -> SearchResponse:
        structured = wrap_plain_answer(
            answer,
            language=language,
            citations=citations,
            intent=intent,
        )
        confidence = compute_confidence(chunks, llm_confidence=llm_confidence).final
        return SearchResponse(
            answer=structured,
            confidence=confidence,
            citations=citations,
            answered=True,
            language=language,  # type: ignore[arg-type]
        )


def run_retrieval(query: str, language: str = "auto") -> RetrievalResult:
    return SearchService().retrieve(query, language)
