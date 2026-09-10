from dataclasses import dataclass

from ingestion.embedding.vector_store import RetrievedChunk
from services.context_builder.compressor import compress_chunks
from shared.config import settings
from shared.document_quality import is_broken_citation_url, is_junk_document, is_low_value_document, is_menu_heavy_text
from shared.schemas import Citation
from shared.url_canonical import canonical_url_key


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


def _scope_citations_for_query(
    query: str, chunks: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    normalized = " ".join(query.lower().split())
    if "ahly points" not in normalized:
        return chunks
    scoped = [chunk for chunk in chunks if "ahlypoints" in (chunk.url or "").lower()]
    return scoped or chunks


class ContextBuilder:
    def build(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        compress: bool = True,
    ) -> BuiltContext:
        max_tokens = settings.context_max_tokens
        working = compress_chunks(query, chunks) if compress else list(chunks)
        primary = [chunk for chunk in working if _is_citable(chunk)]
        fallback = [chunk for chunk in working if chunk not in primary]
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

        citation_chunks = _scope_citations_for_query(query, primary or used_chunks)
        citations = self._select_citations(citation_chunks)

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
            key=lambda chunk: (
                -chunk.score,
                -(1 if getattr(chunk, "category", "") else 0),
            ),
        )
        if not ranked:
            return []

        top_score = ranked[0].score
        citations: list[Citation] = []
        seen_urls: set[str] = set()
        # Prefer dominant category among top hits for ranking boost
        categories = [getattr(c, "category", "general") or "general" for c in ranked[:5]]
        dominant = max(set(categories), key=categories.count) if categories else "general"

        for chunk in ranked:
            if len(citations) >= settings.max_citations:
                break
            if chunk.score < settings.citation_min_score:
                continue
            if top_score - chunk.score > 0.12:
                continue
            url_key = canonical_url_key(chunk.url or "")
            if not chunk.url or url_key in seen_urls:
                continue

            category = getattr(chunk, "category", None) or "general"
            # Soft boost display relevance when category matches dominant family
            relevance = chunk.score
            if category == dominant:
                relevance = min(1.0, relevance + 0.02)

            citations.append(
                Citation(
                    title=chunk.title or chunk.url,
                    url=chunk.url,
                    category=category,
                    relevance_score=round(relevance, 3),
                    reranker_score=round(chunk.score, 3),
                )
            )
            seen_urls.add(url_key)

        # Final sort by relevance descending
        citations.sort(key=lambda c: c.relevance_score or 0.0, reverse=True)
        return citations
