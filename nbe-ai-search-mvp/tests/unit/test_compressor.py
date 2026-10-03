"""Unit tests for the context compressor's near-duplicate detection.

Regression: the old ``_char_jaccard`` compared character *sets*, so any two
fluent English paragraphs shared the whole alphabet and scored >= 0.82 — the
deduper collapsed the entire context into a single chunk. Similarity now uses
word 3-gram shingle Jaccard (``_shingle_jaccard``).
"""

from __future__ import annotations

from ingestion.embedding.vector_store import RetrievedChunk
from services.context_builder.compressor import _shingle_jaccard, compress_chunks


def _chunk(cid: str, url: str, text: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=f"d-{cid}",
        title=f"Title {cid}",
        url=url,
        language="en",
        text=text,
        score=score,
        doc_type="product",
        category="products",
    )


# --------------------------------------------------------------------------
# _shingle_jaccard
# --------------------------------------------------------------------------


def test_distinct_fluent_paragraphs_score_low():
    """The core regression: fluent English paragraphs share the alphabet, not
    the wording — they must NOT look like near-duplicates."""
    a = (
        "Al Ahly Points is a loyalty program that rewards customers whenever "
        "they use banking services and redeem exclusive merchant offers."
    )
    b = (
        "The certificate of deposit guarantees a fixed annual return that is "
        "paid out at maturity without any market risk to the depositor."
    )
    assert _shingle_jaccard(a, b) < 0.2


def test_near_duplicate_wording_scores_high():
    """Paragraph-sized content with a small wording change must still be
    detected as near-duplicate (production chunks are this size)."""
    a = (
        "The certificate of deposit guarantees a fixed annual return that is "
        "paid at maturity. Customers may choose flexible terms starting from "
        "three months up to five years, with the option to renew automatically "
        "at the prevailing rate announced by the bank at the time of renewal."
    )
    b = a.replace("five years", "six years")
    assert _shingle_jaccard(a, b) >= 0.8


def test_identical_text_scores_one():
    text = "The exchange rate table is updated daily before market opening."
    assert _shingle_jaccard(text, text) == 1.0


def test_empty_or_wordless_text_scores_zero():
    assert _shingle_jaccard("", "some words here now") == 0.0
    assert _shingle_jaccard("!!! ???", "some words here now") == 0.0
    assert _shingle_jaccard("", "") == 0.0


def test_short_texts_only_match_when_identical():
    """Fewer words than the shingle size: exact match dedupes, different
    short texts never do."""
    assert _shingle_jaccard("Open account", "Open account") == 1.0
    assert _shingle_jaccard("Open account", "Close account") == 0.0


# --------------------------------------------------------------------------
# compress_chunks
# --------------------------------------------------------------------------


def test_compressor_keeps_distinct_english_chunks():
    """Three distinct fluent English chunks must all survive compression.

    Under character-set Jaccard this collapsed to a single chunk.
    """
    chunks = [
        _chunk(
            "1",
            "https://nbe/points",
            "Al Ahly Points is a loyalty program that rewards customers "
            "whenever they use banking services across participating merchants.",
        ),
        _chunk(
            "2",
            "https://nbe/certificates",
            "Certificates of deposit guarantee a fixed annual return paid at "
            "maturity with flexible terms starting from three months.",
        ),
        _chunk(
            "3",
            "https://nbe/cards",
            "Credit cards include purchase protection, travel insurance and "
            "contactless payments wherever the card network is accepted.",
        ),
    ]
    compressed = compress_chunks("Al Ahly Points", chunks)
    assert len(compressed) == 3
    assert {c.chunk_id for c in compressed} == {"1", "2", "3"}


def test_compressor_dedupes_near_duplicate_english_chunks():
    """Same page content with a small trailing addition is a near-duplicate
    (production chunks are paragraph-sized, so small edits keep shingle
    Jaccard high)."""
    base = (
        "Al Ahly Points is a loyalty program that rewards customers whenever "
        "they use banking services. Earned points can be redeemed against "
        "service fees, purchases at participating merchants and a wide range "
        "of exclusive offers updated every season."
    )
    chunks = [
        _chunk("1", "https://nbe/points", base, 0.9),
        _chunk("2", "https://nbe/points", base + " Points expire after two years.", 0.85),
    ]
    compressed = compress_chunks("Al Ahly Points", chunks)
    assert len(compressed) == 1
    assert compressed[0].chunk_id == "1"


def test_compressor_dedupes_arabic_near_duplicates():
    text = "عائد الشهادة السنوي ثمانية عشر بالمائة للمستثمرين في البنك."
    chunks = [
        _chunk("1", "https://nbe/a", text, 0.9),
        _chunk("2", "https://nbe/a", text, 0.85),
    ]
    compressed = compress_chunks("عائد شهادة", chunks)
    assert len(compressed) == 1


def test_short_exact_duplicate_is_deduped_but_different_short_text_is_kept():
    """Texts under the compressor's 20-char minimum are dropped, so both
    sentences here stay above it while remaining clearly distinct."""
    chunks = [
        _chunk("1", "https://nbe/x", "Open account today in minutes."),
        _chunk("2", "https://nbe/x", "Open account today in minutes.", 0.85),
        _chunk("3", "https://nbe/y", "Close account now please.", 0.8),
    ]
    compressed = compress_chunks("account", chunks)
    ids = [c.chunk_id for c in compressed]
    assert "2" not in ids
    assert "1" in ids and "3" in ids
