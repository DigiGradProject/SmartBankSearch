"""Traditional (keyword) search — BM25-only, no LLM, no reranker, no business rules.

Design (docs/plan-traditional-search.md, Decisions #1–#3):
  * Query prep strips stopwords, expands banking synonyms, and adds light
    stem variants (Arabic prefix/suffix, English plural) so inflected queries
    match un-inflected documents (dual-token: raw OR stemmed).
  * Results are grouped per page (canonical URL) — 809 chunks must never leak
    into the UI as duplicates of the same page.
  * BM25 raw scores are min-max normalized per language block for display.
  * Same-language results rank first; when fewer than
    `keyword_fill_min_results` pages are found, the other language fills the
    remaining slots (per-result language tag drives RTL/LTR rendering).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ingestion.lexical.bm25_index import BM25Index, IndexedChunk, get_bm25_index, tokenize_text
from services.rag.language import detect_language
from services.search_service.keyword_rank import EN_SPACING_ALIASES, apply_spacing_aliases, extract_query_terms
from services.search_service.synonyms import expand_with_synonyms
from shared.arabic_normalize import normalize_arabic
from shared.config import settings
from shared.logging import get_logger
from shared.schemas import TraditionalResult
from shared.url_canonical import canonical_url_key

logger = get_logger(__name__)

WHITESPACE = re.compile(r"\s+")

# Function words only — content words must never land here.
STOPWORDS_AR = frozenset(
    {
        "من", "في", "علي", "عن", "الي", "ما", "ماذا", "هو", "هي", "هما",
        "كيف", "هل", "او", "و", "ثم", "هذا", "هذه", "ذلك", "تلك", "الذي",
        "التي", "الذين", "اي", "ايه", "عند", "مع", "بعد", "قبل", "بين",
        "كل", "بعض", "لا", "لم", "لن", "ان", "انه", "انها", "كان", "كانت",
        "ليس", "هناك", "لو", "مين", "فين", "امتي", "ازاي", "ليه", "حد",
    }
)
STOPWORDS_EN = frozenset(
    {
        "the", "and", "for", "are", "was", "were", "been", "does", "did",
        "how", "what", "when", "where", "which", "who", "whom", "why",
        "can", "could", "should", "would", "will", "shall", "may", "might",
        "must", "you", "your", "she", "its", "they", "them", "their",
        "this", "that", "these", "those", "with", "about", "from",
        "please", "tell", "show", "give", "get", "want", "need", "like",
        "his", "her", "him", "our", "not", "but", "any", "all",
        "to", "of", "in", "on", "at", "by", "is", "it", "as", "be",
        "or", "an", "do", "we", "into",
        # Filler superlatives common in navigation queries; their high IDF
        # otherwise dominates BM25 ("Where is the *nearest* branch?").
        "nearest", "closest", "nearby",
    }
)
STOPWORDS = {"ar": STOPWORDS_AR, "en": STOPWORDS_EN}

# One prefix + one suffix max, applied after normalize_arabic (ة→ه, ى→ي).
_AR_PREFIXES = ("وبال", "فبال", "وال", "بال", "كال", "فال", "لل", "ال")
_AR_SUFFIXES = ("اتها", "اتهم", "اتكم", "اتنا", "اتك", "ها", "هم", "هما", "كم", "كن", "نا", "ات", "ين", "ون", "ه")
_MIN_STEM_LEN = 3

# Arabic infix plurals (و inserted inside the root) that suffix stripping
# cannot recover: query قرض must also match documents saying قروض.
_AR_INFIX_PLURALS = {
    "قرض": "قروض",
    "بنك": "بنوك",
    "فرع": "فروع",
    "كتاب": "كتب",
    "مبلغ": "مبالغ",
    "قسم": "اقسام",
}

_TITLE_BOOST_WEIGHT = 0.15
_TITLE_PHRASE_BONUS = 0.60
_BODY_PHRASE_BONUS = 0.25
_SNIPPET_TERM_WINDOW = 60


def light_stem_ar(token: str) -> str:
    """Arabic light stemmer — conservative, query-time only."""
    stemmed = token
    for prefix in _AR_PREFIXES:
        if stemmed.startswith(prefix) and len(stemmed) - len(prefix) >= _MIN_STEM_LEN:
            stemmed = stemmed[len(prefix):]
            break
    for suffix in _AR_SUFFIXES:
        if stemmed.endswith(suffix) and len(stemmed) - len(suffix) >= _MIN_STEM_LEN:
            stemmed = stemmed[: -len(suffix)]
            break
    return stemmed


def light_stem_en(token: str) -> str:
    """English plural stripper — conservative, query-time only.

    Requires ≥4 remaining characters so "rates" stays "rates" instead of
    degrading to "rat"; short words are left untouched.
    """
    if len(token) >= 7 and token.endswith("ies"):
        return token[:-3] + "y"
    # -es is only a plural marker after sibilants (classes→class, boxes→box);
    # otherwise certificates must keep its final e (strip plain -s).
    if len(token) >= 6 and token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if (
        len(token) >= 5
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        return token[:-1]
    return token


def light_stem(token: str, language: str) -> str:
    return light_stem_ar(token) if language == "ar" else light_stem_en(token)


@dataclass(frozen=True)
class KeywordQuery:
    """Prepared keyword query: what BM25 sees vs what the UI highlights."""

    original_query: str
    language: str
    effective: bool
    bm25_query: str = ""
    terms_by_language: dict[str, list[str]] = field(default_factory=dict)
    phrases_by_language: dict[str, list[str]] = field(default_factory=dict)


def prepare_keyword_query(query: str, language: str = "auto") -> KeywordQuery:
    """Build the BM25 query text and per-language highlight terms."""
    resolved = detect_language(query.strip(), language if language in {"ar", "en"} else "auto")
    other = "en" if resolved == "ar" else "ar"

    def _terms_for(lang: str) -> list[str]:
        # Highlight terms only — function words must never be highlighted.
        terms = [t for t in extract_query_terms(query, lang) if t not in STOPWORDS[lang]]
        if lang == "ar":
            for term in list(terms):
                infix = _AR_INFIX_PLURALS.get(term)
                if infix and infix not in terms:
                    terms.append(infix)
        stems = [light_stem(term, lang) for term in terms]
        for stem in stems:
            if len(stem) >= _MIN_STEM_LEN and stem not in terms:
                terms.append(stem)
        return terms

    # Re-space compound brand words first ("alahly points" → "al ahly"): the
    # fixed index stores whole tokens, so BM25 can never match "alahly" to
    # the corpus's "Al Ahly" without this normalization.
    normalized_query = apply_spacing_aliases(query)
    tokens = tokenize_text(normalized_query, resolved)
    content = [token for token in tokens if token not in STOPWORDS[resolved]]
    content = content[: settings.keyword_max_query_tokens]

    terms_by_language = {resolved: _terms_for(resolved)}
    if not content:
        return KeywordQuery(
            original_query=query,
            language=resolved,
            effective=False,
            terms_by_language=terms_by_language,
        )

    cleaned = " ".join(content)
    expanded = expand_with_synonyms(cleaned, resolved)
    if resolved == "en":
        # Re-spaced form must reach BM25 verbatim: synonym expansion runs on
        # the raw words and can rejoin/re-shuffle them ("al ahly" → "alahly").
        expanded = apply_spacing_aliases(expanded)
        # Drop the original compound tokens ("alahly") that the re-spaced
        # query already covers, so BM25 doesn't weigh both variants.
        original_compounds = {
            token for token in cleaned.split() if token in EN_SPACING_ALIASES
        }
        if original_compounds:
            expanded = " ".join(
                token for token in expanded.split() if token not in original_compounds
            )

    extra_stems: list[str] = []
    for token in content:
        stem = light_stem(token, resolved)
        if stem != token and len(stem) >= _MIN_STEM_LEN and stem not in extra_stems:
            extra_stems.append(stem)
        # Mirror morphology: documents are often inflected while queries are
        # bare (AR: قرض vs القروض, سيارات vs سياره · EN: branch vs branches).
        # The fixed index stores whole tokens, so expand the query with the
        # inflected variants for BM25 OR matching.
        if resolved == "ar" and not token.startswith("ال"):
            variants = [f"ال{token}"]
            infix = _AR_INFIX_PLURALS.get(token)
            if infix:
                variants.extend([infix, f"ال{infix}"])
            if token.endswith("ات") and len(token) >= 5:
                singular = token[:-2] + "ه"
                variants.extend([singular, f"ال{singular}"])
            for variant in variants:
                if len(variant) >= _MIN_STEM_LEN and variant not in extra_stems:
                    extra_stems.append(variant)
        elif resolved == "en" and not token.endswith("s"):
            plural = token + ("es" if token.endswith(("x", "z", "ch", "sh")) else "s")
            if len(plural) >= _MIN_STEM_LEN and plural not in extra_stems:
                extra_stems.append(plural)

    # For re-spaced brand queries ("alahly" → "al ahly") the spaced form must
    # reach BM25 verbatim, but only then: otherwise stopword-filtered words
    # would leak back in via the verbatim copy.
    bm25_parts = [expanded]
    if normalized_query.strip() != cleaned and any(
        token in EN_SPACING_ALIASES for token in cleaned.split()
    ):
        bm25_parts.append(normalized_query)
    bm25_parts.append(" ".join(extra_stems))
    bm25_query = " ".join(part for part in bm25_parts if part)
    terms_by_language[other] = _terms_for(other)

    # Adjacent raw content words become phrases ("شهادات بلادي", "car loan")
    # used for the exact-phrase title/body bonus in group_pages.
    phrases = [
        " ".join(content[i : i + 2])
        for i in range(len(content) - 1)
        if len(content[i]) >= 3 and len(content[i + 1]) >= 3
    ]
    phrases_by_language = {resolved: phrases, other: []}

    return KeywordQuery(
        original_query=query,
        language=resolved,
        effective=True,
        bm25_query=bm25_query,
        terms_by_language=terms_by_language,
        phrases_by_language=phrases_by_language,
    )


def _normalized_text(text: str, language: str) -> str:
    """Normalized haystack so term positions align with normalized terms.

    Arabic snippets come from the normalized text (diacritics/tatweel
    removed) — normalize_arabic substitutes characters 1:1 except for
    diacritics removal, so positions are only reliable within the
    normalized string itself.
    """
    cleaned = WHITESPACE.sub(" ", text).strip()
    return normalize_arabic(cleaned) if language == "ar" else cleaned.lower()


def _term_coverage(text: str, terms: list[str], language: str) -> float:
    if not terms:
        return 0.0
    haystack = _normalized_text(text, language)
    hits = sum(1 for term in terms if term in haystack)
    return hits / len(terms)


def build_snippet(text: str, terms: list[str], language: str, max_chars: int | None = None) -> str:
    """Sentence-ish window around the first term hit, from normalized text."""
    window = max_chars or settings.keyword_snippet_chars
    haystack = _normalized_text(text, language)
    if not haystack:
        return ""

    hit_pos = -1
    for term in terms:
        pos = haystack.find(term)
        if pos >= 0 and (hit_pos < 0 or pos < hit_pos):
            hit_pos = pos
    if hit_pos < 0:
        return haystack[:window].rsplit(" ", 1)[0] if len(haystack) > window else haystack

    start = max(0, hit_pos - _SNIPPET_TERM_WINDOW)
    end = min(len(haystack), start + window)
    start = max(0, min(start, end - window))
    snippet = haystack[start:end].strip()
    if len(haystack) > end:
        snippet = snippet.rsplit(" ", 1)[0] + "…"
    if start > 0:
        snippet = "…" + snippet.lstrip().split(" ", 1)[-1]
    return snippet


def _title_boosted_score(
    chunk: IndexedChunk,
    score: float,
    terms: list[str],
    language: str,
    phrases: list[str] | None = None,
) -> float:
    """Title-coverage boost + exact-phrase bonus (adjacent raw query words).

    BM25 is bag-of-words: "شهادات بلادي" scores the Belady page and every
    sibling certificate page nearly identically. A verbatim phrase hit in the
    title is strong evidence of the intended page (stretch item 4.5).
    """
    coverage = _term_coverage(chunk.title, terms, language)
    boosted = score * (1.0 + _TITLE_BOOST_WEIGHT * coverage)
    if phrases:
        title_haystack = _normalized_text(chunk.title, language)
        body_haystack = _normalized_text(chunk.text, language)
        for phrase in phrases:
            if phrase in title_haystack:
                boosted *= 1.0 + _TITLE_PHRASE_BONUS
                break
            if phrase in body_haystack:
                boosted *= 1.0 + _BODY_PHRASE_BONUS
                break
    return boosted


def group_pages(
    hits: list[tuple[IndexedChunk, float]],
    *,
    terms: list[str],
    language: str,
    phrases: list[str] | None = None,
) -> list[tuple[IndexedChunk, float]]:
    """Best chunk per canonical page, title-boosted, ranked."""
    best: dict[str, tuple[IndexedChunk, float]] = {}
    for chunk, score in hits:
        key = canonical_url_key(chunk.url) or chunk.chunk_id
        boosted = _title_boosted_score(chunk, score, terms, language, phrases)
        current = best.get(key)
        if current is None or boosted > current[1]:
            best[key] = (chunk, boosted)
    return sorted(best.values(), key=lambda pair: pair[1], reverse=True)


def normalize_scores(pairs: list[tuple[IndexedChunk, float]]) -> list[tuple[IndexedChunk, float]]:
    """Min-max normalize a ranked block to (0.05 … 1.0]; single result → 1.0."""
    if not pairs:
        return []
    scores = [score for _, score in pairs]
    low, high = min(scores), max(scores)
    if high <= low:
        return [(chunk, 1.0) for chunk, _ in pairs]
    return [
        (chunk, round(0.05 + 0.95 * (score - low) / (high - low), 3))
        for chunk, score in pairs
    ]


@dataclass(frozen=True)
class TraditionalSearchOutcome:
    results: list[TraditionalResult] = field(default_factory=list)
    total: int = 0
    language: str = "en"
    effective_query: bool = True
    bm25_available: bool = True


class TraditionalSearchService:
    """BM25-only page search behind the `mode="traditional"` API contract."""

    def __init__(self, bm25_index: BM25Index | None = None) -> None:
        self._bm25_index = bm25_index

    def _get_bm25(self) -> BM25Index | None:
        if self._bm25_index is not None:
            return self._bm25_index if self._bm25_index.size > 0 else None
        if not settings.bm25_enabled:
            return None
        index = get_bm25_index()
        return index if index.size > 0 else None

    def search(
        self,
        query: str,
        language: str = "auto",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> TraditionalSearchOutcome:
        page_size = limit or settings.keyword_results_limit
        page_size = max(1, min(page_size, settings.keyword_results_max_limit))
        offset = max(0, offset)

        kw = prepare_keyword_query(query, language)
        if not kw.effective:
            return TraditionalSearchOutcome(
                language=kw.language, effective_query=False, bm25_available=True
            )

        index = self._get_bm25()
        if index is None:
            return TraditionalSearchOutcome(
                language=kw.language, effective_query=True, bm25_available=False
            )

        fetch_k = max(page_size * 4, settings.bm25_top_k)
        primary_terms = kw.terms_by_language.get(kw.language, [])
        other_language = "en" if kw.language == "ar" else "ar"
        other_terms = kw.terms_by_language.get(other_language, [])

        primary_phrases = kw.phrases_by_language.get(kw.language, [])
        primary_hits = index.query(kw.bm25_query, kw.language, fetch_k)
        pages = group_pages(
            primary_hits, terms=primary_terms, language=kw.language, phrases=primary_phrases
        )
        candidates = normalize_scores(pages)

        # Decision #3: fill remaining slots from the other language when the
        # primary language has too few pages. Capped at one window so the
        # fill-in cannot inflate `total` far beyond what the user asked for.
        if len(candidates) < settings.keyword_fill_min_results:
            other_hits = index.query(kw.bm25_query, other_language, fetch_k)
            other_pages = group_pages(other_hits, terms=other_terms, language=other_language)
            candidates.extend(normalize_scores(other_pages[:page_size]))

        total = len(candidates)
        window = candidates[offset : offset + page_size]

        results: list[TraditionalResult] = []
        for chunk, score in window:
            url = chunk.url or ""
            # Hardening: results become clickable hrefs in the UI — only
            # http(s) URLs from the corpus may be rendered as links.
            if not url.startswith(("http://", "https://")):
                continue
            result_language = chunk.language if chunk.language in {"ar", "en"} else kw.language
            terms = kw.terms_by_language.get(result_language, [])
            results.append(
                TraditionalResult(
                    title=chunk.title,
                    url=url,
                    snippet=build_snippet(chunk.text, terms, result_language),
                    score=round(float(score), 3),
                    terms=terms,
                    language=result_language,  # type: ignore[arg-type]
                    category=chunk.category or None,
                    doc_type=chunk.doc_type or None,
                )
            )

        logger.info(
            "traditional_search_completed",
            language=kw.language,
            effective=kw.effective,
            pages=total,
            returned=len(results),
            offset=offset,
            limit=page_size,
        )
        return TraditionalSearchOutcome(
            results=results, total=total, language=kw.language
        )
