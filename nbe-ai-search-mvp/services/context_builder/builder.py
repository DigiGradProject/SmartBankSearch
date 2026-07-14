from dataclasses import dataclass

from ingestion.embedding.vector_store import RetrievedChunk
from shared.config import settings
from shared.document_quality import is_broken_citation_url, is_junk_document, is_low_value_document, is_menu_heavy_text
from shared.schemas import Citation


@dataclass
class BuiltContext:
    context_text: str
    citations: list[Citation]
    token_estimate: int


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _is_citable(chunk: RetrievedChunk) -> bool:
    return not (
        is_junk_document(chunk.document_id)
        or is_low_value_document(chunk.document_id)
        or is_menu_heavy_text(chunk.text)
        or is_broken_citation_url(chunk.url)
    )


class ContextBuilder:
    def build(self, query: str, chunks: list[RetrievedChunk]) -> BuiltContext:
        max_tokens = settings.context_max_tokens
        primary = [chunk for chunk in chunks if _is_citable(chunk)]
        fallback = [chunk for chunk in chunks if chunk not in primary]
        ordered = primary + fallback

        used_tokens = 0
        context_parts: list[str] = []
        used_chunks: list[RetrievedChunk] = []

        for idx, chunk in enumerate(ordered, start=1):
            snippet = chunk.text.strip()
            block = f"[{idx}] {snippet}"
            block_tokens = _approx_tokens(block)
            if used_tokens + block_tokens > max_tokens:
                break
            context_parts.append(block)
            used_chunks.append(chunk)
            used_tokens += block_tokens

        citations = self._select_citations(primary or used_chunks)

        return BuiltContext(
            context_text="\n\n".join(context_parts),
            citations=citations,
            token_estimate=used_tokens,
        )

    def _select_citations(self, chunks: list[RetrievedChunk]) -> list[Citation]:
        if not chunks:
            return []

        ranked = sorted(
            [chunk for chunk in chunks if _is_citable(chunk)],
            key=lambda chunk: chunk.score,
            reverse=True,
        )
        if not ranked:
            return []

        top_score = ranked[0].score
        citations: list[Citation] = []
        seen_urls: set[str] = set()

        for chunk in ranked:
            if len(citations) >= settings.max_citations:
                break
            if chunk.score < settings.citation_min_score:
                continue
            if top_score - chunk.score > 0.12:
                continue
            if not chunk.url or chunk.url in seen_urls:
                continue

            citations.append(Citation(title=chunk.title or chunk.url, url=chunk.url))
            seen_urls.add(chunk.url)

        return citations
