import re
from dataclasses import dataclass

from ingestion.embedding.vector_store import RetrievedChunk
from services.context_builder.compressor import compress_chunks
from shared.config import settings
from shared.document_quality import is_broken_citation_url, is_junk_document, is_low_value_document, is_menu_heavy_text
from shared.logging import get_logger
from shared.schemas import Citation
from shared.url_canonical import canonical_url_key
from services.search_service.keyword_rank import extract_query_terms

logger = get_logger(__name__)


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


def _dominant_category(chunks: list[RetrievedChunk]) -> str:
    """Most common category among the given chunks (ties → first seen)."""
    counts: dict[str, int] = {}
    for chunk in chunks:
        category = getattr(chunk, "category", "general") or "general"
        counts[category] = counts.get(category, 0) + 1
    if not counts:
        return "general"
    return max(counts, key=counts.get)  # type: ignore[arg-type]


def _is_offtopic_head(query: str, head: RetrievedChunk) -> bool:
    """True when the head chunk's BODY shares no vocabulary with the query.

    Titles are templated across the site ("National Bank of Egypt - X"), so
    only body text is compared. Site-generic tokens (national/bank/egypt/…)
    appear on every page body and carry no discriminative value, so they are
    ignored. English matches on word boundaries to avoid substring false
    positives ("bank" vs "banknote"); Arabic keeps substring matching for
    morphology. A legit top hit almost always mentions the query topic in its
    body; a page retrieved only via injected intent-expansion tokens does not
    (see docs/root-cause-exchange-rate-citations.md).
    """
    language = "ar" if any(ord(ch) > 127 for ch in query) else "en"
    generic = {"what", "national", "bank", "egypt", "nbe", "which", "who"}
    terms = [t for t in extract_query_terms(query, language) if t not in generic]
    if not terms:
        return False
    body = head.text or ""
    if language == "ar":
        from shared.arabic_normalize import normalize_arabic

        body = normalize_arabic(body)
        hits = sum(1 for term in terms if term in body)
    else:
        body_lower = body.lower()
        hits = sum(
            1
            for term in terms
            if re.search(rf"\b{re.escape(term)}\b", body_lower)
        )
    return hits == 0


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

        citations = self._select_citations(query, primary or used_chunks)

        return BuiltContext(
            context_text="\n\n".join(context_parts),
            citations=citations,
            token_estimate=used_tokens,
        )

    def _select_citations(self, query: str, chunks: list[RetrievedChunk]) -> list[Citation]:
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
        excluded_keys: set[str] = set()
        dominant = _dominant_category(ranked[:5])

        citation_floor = settings.citation_min_score
        rest = ranked[1:]
        if (
            rest
            and top_score - rest[0].score > settings.citation_category_gap
            and _is_offtopic_head(query, ranked[0])
        ):
            # Off-topic page saturating the reranker (e.g. an injected intent
            # expansion matched its page title). Re-anchor citations on the
            # remaining on-topic evidence so the true source is not crowded out.
            logger.info(
                "citation_offtopic_head_dropped",
                dropped_url=ranked[0].url,
                dropped_category=getattr(ranked[0], "category", ""),
                gap=round(top_score - rest[0].score, 3),
            )
            excluded_keys.add(canonical_url_key(ranked[0].url or "") or ranked[0].chunk_id)
            top_score = rest[0].score
            ranked = rest
            # The anchor is the best on-topic evidence for this answer;
            # admit it (and near-ties) even below the global floor.
            citation_floor = min(citation_floor, top_score)

        citations: list[Citation] = []
        seen_urls: set[str] = set()

        for chunk in ranked:
            if len(citations) >= settings.max_citations:
                break
            url_key = canonical_url_key(chunk.url or "")
            if url_key and url_key in excluded_keys:
                continue
            if chunk.score < citation_floor:
                continue
            if top_score - chunk.score > 0.12:
                continue
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
