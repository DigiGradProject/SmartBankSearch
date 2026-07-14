import re
from dataclasses import replace

from ingestion.embedding.vector_store import RetrievedChunk
from shared.arabic_normalize import normalize_arabic

ARABIC_TERM = re.compile(r"[\u0600-\u06FF]{2,}")
ENGLISH_TERM = re.compile(r"[a-zA-Z]{3,}")

TERM_ALIASES = {
    "انواع": "شهادات",
    "الشهادات": "شهادات",
    "البنكيه": "بنكي",
    "بنكية": "بنكي",
    "بلادي": "بلادى",
    "بلاد": "بلادى",
    "الدولار": "دولار",
    "الامريكي": "امريكي",
    "الأمريكي": "امريكي",
    "سنوات": "سنه",
    "سنة": "سنه",
    "شهاده": "شهادة",
    "اشتري": "شراء",
    "عايز": "شراء",
    "عاوز": "شراء",
}


def extract_query_terms(query: str, language: str) -> list[str]:
    if language == "ar":
        terms = ARABIC_TERM.findall(normalize_arabic(query))
    else:
        terms = [term.lower() for term in ENGLISH_TERM.findall(query.lower())]

    normalized: list[str] = []
    for term in terms:
        mapped = TERM_ALIASES.get(term, term)
        if len(mapped) >= 2 and mapped not in normalized:
            normalized.append(mapped)
    return normalized


def keyword_overlap_score(chunk_text: str, terms: list[str], language: str) -> float:
    if not terms:
        return 0.0

    haystack = normalize_arabic(chunk_text) if language == "ar" else chunk_text.lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / len(terms)


def rerank_chunks(chunks: list[RetrievedChunk], query: str, language: str) -> list[RetrievedChunk]:
    terms = extract_query_terms(query, language)
    if not terms:
        return chunks

    boosted: list[RetrievedChunk] = []
    for chunk in chunks:
        overlap = keyword_overlap_score(chunk.text, terms, language)
        boosted_score = min(1.0, chunk.score + overlap * 0.35)
        boosted.append(replace(chunk, score=boosted_score))

    return sorted(boosted, key=lambda item: item.score, reverse=True)
