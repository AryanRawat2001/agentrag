"""Semantic chunking tests.

The load-bearing invariant is **tiling**: spans must be adjacent and cover
`[0, len(text))` exactly. Chunk `text` is a byte-exact source slice that Phase 5's
citation verifier resolves against, so a gap silently loses text and an overlap
double-counts it. Both are asserted directly rather than inferred from chunk counts.

A fake embedder makes topic shifts explicit, so cut placement is checked against a
known-correct answer instead of against whatever a real model happens to do.
"""

from __future__ import annotations

import numpy as np
import pytest

from ragpipe import chunking, semantic


class ScriptedEmbedder:
    """Maps each sentence to a vector by a marker in its text.

    Sentences sharing a marker are identical in embedding space (distance 0);
    different markers are orthogonal (distance 1). That makes breakpoints exact.
    """

    dim = 4

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.batches.append(list(texts))
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for i, marker in enumerate("ABCD"):
                if f"[{marker}]" in t:
                    v[i] = 1.0
                    break
            else:
                v[0] = 1.0
            out.append(v)
        return np.stack(out)


def sentences(marker: str, n: int, filler: str = "x") -> str:
    """`n` sentences long enough to survive MIN_SENTENCE_CHARS, all one topic."""
    pad = filler * 60
    return " ".join(f"[{marker}] sentence {i} {pad}." for i in range(n))


def assert_tiles(spans, text_len: int) -> None:
    assert spans, "expected at least one span"
    assert spans[0][0] == 0, f"first span starts at {spans[0][0]}, not 0"
    assert spans[-1][1] == text_len, f"last span ends at {spans[-1][1]}, not {text_len}"
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert prev_end == next_start, f"gap or overlap at {prev_end} != {next_start}"


# -- sentence_spans ---------------------------------------------------------


def test_sentence_spans_tile_the_text():
    text = "First sentence here. Second one follows! Third one ends? And a fourth."
    spans = semantic.sentence_spans(text, min_chars=1)
    assert_tiles(spans, len(text))
    assert len(spans) == 4


def test_sentence_spans_reconstruct_the_text_exactly():
    text = "One two three. Four five six! Seven eight?"
    spans = semantic.sentence_spans(text, min_chars=1)
    assert "".join(text[s:e] for s, e in spans) == text


def test_short_fragments_merge_rather_than_disappear():
    """Dropping a fragment would break coverage and lose text silently."""
    text = "See Table 3. " + "A properly long sentence that carries actual signal " * 2 + "."
    spans = semantic.sentence_spans(text, min_chars=40)
    assert_tiles(spans, len(text))
    assert len(spans) < 3


def test_leading_fragment_merges_forward():
    text = "Yes. " + "This is a substantially longer sentence with real content in it."
    spans = semantic.sentence_spans(text, min_chars=40)
    assert spans[0][0] == 0
    assert_tiles(spans, len(text))


def test_sentence_spans_handle_empty_and_single():
    assert semantic.sentence_spans("") == []
    text = "Just one sentence with no terminal punctuation"
    assert_tiles(semantic.sentence_spans(text), len(text))


def test_sentence_spans_do_not_split_on_decimal_or_citation_dots():
    """`21 CFR 314.50` and `0.5 mg` must not become sentence boundaries — the
    pattern requires whitespace after the terminator."""
    text = "The limit in 21 CFR 314.50 is 0.5 mg per dose for all subjects enrolled."
    assert len(semantic.sentence_spans(text, min_chars=1)) == 1


# -- breakpoints ------------------------------------------------------------


def test_breakpoints_selects_the_largest_distances():
    d = np.array([0.01, 0.02, 0.9, 0.01, 0.95])
    assert set(semantic.breakpoints(d, percentile=60).tolist()) == {2, 4}


def test_breakpoints_are_relative_not_absolute():
    """The reason a percentile is used: absolute distances are not comparable
    across documents, so a fixed threshold splits one and not the other."""
    tight = np.array([0.01, 0.02, 0.03, 0.20])
    loose = np.array([0.50, 0.55, 0.60, 0.95])
    assert semantic.breakpoints(tight, 75).tolist() == semantic.breakpoints(loose, 75).tolist()


def test_flat_distribution_yields_no_breakpoints():
    """Every distance identical means no topic shift. A `>=` comparison here would
    cut at every single sentence."""
    assert semantic.breakpoints(np.full(10, 0.3), percentile=95).size == 0


def test_breakpoints_handles_empty():
    assert semantic.breakpoints(np.zeros(0)).size == 0


# -- chunk_semantic --------------------------------------------------------


def doc(text: str) -> dict:
    return {
        "doc_id": "d1",
        "source": "test",
        "text": text,
        "pages": [],
        "sections": [],
        "identifiers": [],
        "metadata": {},
    }


def test_semantic_cuts_at_the_topic_shift():
    text = sentences("A", 3) + " " + sentences("B", 3)
    spans = semantic.chunk_semantic(
        doc(text), target=100_000, overlap=0, embedder=ScriptedEmbedder()
    )
    assert_tiles([(s, e) for s, e, _ in spans], len(text))
    assert len(spans) == 2
    # The cut lands between topics, so each chunk holds exactly one marker.
    first, second = text[spans[0][0] : spans[0][1]], text[spans[1][0] : spans[1][1]]
    assert "[B]" not in first and "[A]" not in second


def test_semantic_spans_always_tile_regardless_of_topic_layout():
    for markers in (["A"], ["A", "B"], ["A", "B", "C", "D"], ["A", "B", "A", "B"]):
        text = " ".join(sentences(m, 2) for m in markers)
        spans = semantic.chunk_semantic(
            doc(text), target=100_000, overlap=0, embedder=ScriptedEmbedder()
        )
        assert_tiles([(s, e) for s, e, _ in spans], len(text))


def test_target_caps_chunk_size_even_with_no_topic_shift():
    """Failure mode 1: a uniform document must not become one enormous chunk.

    The bound is `target + BOUNDARY_SEARCH_CHARS + MIN_CHUNK_CHARS - 1`: forward
    snapping overshoots by up to the search window, and absorbing a runt extends its
    predecessor by up to just under the minimum. Both are needed, and together they
    are why the largest semantic chunk is 2,464 chars where `fixed` tops out at 2,298.
    """
    bound = 800 + chunking.BOUNDARY_SEARCH_CHARS + chunking.MIN_CHUNK_CHARS - 1
    for n in (12, 40, 91):
        text = sentences("A", n)
        spans = semantic.chunk_semantic(
            doc(text), target=800, overlap=0, embedder=ScriptedEmbedder()
        )
        assert_tiles([(s, e) for s, e, _ in spans], len(text))
        # The bound is the invariant. Span *count* is not: a document only slightly
        # over `target` snaps forward to its end and stays a single span, which is
        # correct behaviour rather than a missing split.
        assert max(e - s for s, e, _ in spans) <= bound, f"n={n}"

    # With enough text, it must actually split rather than return one giant span.
    long_spans = semantic.chunk_semantic(
        doc(sentences("A", 91)), target=800, overlap=0, embedder=ScriptedEmbedder()
    )
    assert len(long_spans) > 1


def test_no_chunk_below_the_runt_threshold():
    """Failure mode 2: trough-dense text must not become sentence fragments."""
    text = " ".join(sentences(m, 1) for m in "ABCDABCD")
    spans = semantic.chunk_semantic(
        doc(text), target=100_000, overlap=0, embedder=ScriptedEmbedder()
    )
    assert_tiles([(s, e) for s, e, _ in spans], len(text))
    assert min(e - s for s, e, _ in spans) >= chunking.MIN_CHUNK_CHARS


def test_oversize_split_absorbs_runts_it_creates():
    """Regression for the review's HIGH finding.

    `_oversize_split` was a copy of `chunking._windows` that dropped its closing
    `_absorb_runts`. Forward boundary-snapping leaves a `target - snap` remainder, so
    a span of `target + 2` split into `target + snap` plus a **1-character** chunk.
    597 of 8,001 semantic chunks (7.5%) came out under `MIN_CHUNK_CHARS`, the
    smallest being one character — while the module docstring claimed the opposite.
    """
    text = "z" * 2048 + "\n\nQ"
    spans = semantic.chunk_semantic(doc(text), target=2048, overlap=0, embedder=ScriptedEmbedder())
    assert_tiles([(s, e) for s, e, _ in spans], len(text))
    assert min(e - s for s, e, _ in spans) >= chunking.MIN_CHUNK_CHARS


@pytest.mark.parametrize("tail", [1, 2, 5, 50, 199])
def test_no_runt_for_any_short_remainder(tail):
    """The defect only showed for particular remainders, so sweep them."""
    text = "z" * 2048 + "\n\n" + "Q" * tail
    spans = semantic.chunk_semantic(doc(text), target=2048, overlap=0, embedder=ScriptedEmbedder())
    assert_tiles([(s, e) for s, e, _ in spans], len(text))
    assert min(e - s for s, e, _ in spans) >= chunking.MIN_CHUNK_CHARS


def test_no_runt_on_the_single_sentence_return_path():
    """Two of three `chunk_semantic` return paths reach `_oversize_split` without
    passing through `_spans_from_cuts`, so absorbing only there was not enough."""
    text = "row " * 513  # 2052 chars, no sentence terminator
    spans = semantic.chunk_semantic(doc(text), target=2048, overlap=0, embedder=ScriptedEmbedder())
    assert min(e - s for s, e, _ in spans) >= chunking.MIN_CHUNK_CHARS


def test_no_runt_on_the_oversized_document_fallback_path():
    text = ("Short one. " * (semantic.MAX_SENTENCES_PER_DOC + 10)) + "x" * 5
    spans = semantic.chunk_semantic(doc(text), target=2048, overlap=0, embedder=ScriptedEmbedder())
    assert min(e - s for s, e, _ in spans) >= chunking.MIN_CHUNK_CHARS


def test_tail_text_is_never_lost():
    """Failure mode 3: off-by-one in the breakpoint loop drops the final region."""
    text = sentences("A", 5) + " " + sentences("B", 1) + " TAIL MARKER SENTENCE ENDS HERE."
    spans = semantic.chunk_semantic(doc(text), target=600, overlap=0, embedder=ScriptedEmbedder())
    assert_tiles([(s, e) for s, e, _ in spans], len(text))
    assert "TAIL MARKER" in text[spans[-1][0] : spans[-1][1]]


def test_single_oversize_sentence_is_hard_split():
    """Reachable in this corpus: extraction produces un-punctuated table rows far
    longer than the target."""
    text = "word " * 600  # no terminal punctuation at all
    spans = semantic.chunk_semantic(doc(text), target=500, overlap=0, embedder=ScriptedEmbedder())
    assert_tiles([(s, e) for s, e, _ in spans], len(text))
    assert len(spans) > 1


def test_empty_document_yields_no_spans():
    assert (
        semantic.chunk_semantic(doc(""), target=500, overlap=0, embedder=ScriptedEmbedder()) == []
    )


def test_document_with_one_sentence_under_target_is_one_span():
    text = "A single lonely sentence that is quite long but still only one."
    spans = semantic.chunk_semantic(doc(text), target=5000, overlap=0, embedder=ScriptedEmbedder())
    assert spans == [(0, len(text), None)]


def test_single_sentence_over_target_is_still_split():
    """The early return for `len(sentences) < 2` skipped the size cap entirely, so a
    document of un-punctuated table rows became one span of arbitrary length."""
    text = "row " * 500  # 2000 chars, no sentence terminator anywhere
    spans = semantic.chunk_semantic(doc(text), target=400, overlap=0, embedder=ScriptedEmbedder())
    assert_tiles([(s, e) for s, e, _ in spans], len(text))
    assert len(spans) > 1
    assert max(e - s for s, e, _ in spans) <= 400 + chunking.BOUNDARY_SEARCH_CHARS


def test_oversized_document_falls_back_without_embedding():
    """The sentence cap exists so one outlier document cannot dominate a run."""
    e = ScriptedEmbedder()
    text = "Short one. " * (semantic.MAX_SENTENCES_PER_DOC + 50)
    spans = semantic.chunk_semantic(doc(text), target=2000, overlap=0, embedder=e)
    assert_tiles([(s, e_) for s, e_, _ in spans], len(text))
    assert e.batches == []  # never embedded


def test_sentences_embedded_without_query_prefix():
    """These are document-side vectors compared to each other, not queries."""
    e = ScriptedEmbedder()
    text = sentences("A", 2) + " " + sentences("B", 2)
    semantic.chunk_semantic(doc(text), target=100_000, overlap=0, embedder=e)
    assert e.batches, "expected one document-side batch"
    assert all("Represent this sentence" not in t for t in e.batches[0])


def test_identical_sentences_produce_no_spurious_cut():
    """Clipping guard: float error on an identical pair can push cosine above 1.0
    and make the distance negative."""
    text = sentences("A", 6)
    spans = semantic.chunk_semantic(
        doc(text), target=100_000, overlap=0, embedder=ScriptedEmbedder()
    )
    assert len(spans) == 1


def test_build_chunks_produces_byte_exact_slices():
    """End-to-end: the span-to-text contract the citation verifier depends on."""
    text = sentences("A", 4) + " " + sentences("B", 4)
    chunks = chunking.build_chunks(
        doc(text), "semantic", target=100_000, overlap=0, embedder=ScriptedEmbedder()
    )
    assert chunks
    for c in chunks:
        assert c.text == text[c.start : c.end]
        assert c.strategy == "semantic"


def test_assembled_chunks_lose_no_non_whitespace_character():
    """The invariant that actually matters end to end.

    `build_chunks` drops whitespace-only spans, so assembled chunks need not tile
    `[0, len(text))` — and sentence splitting *does* produce whitespace-only spans
    where a blank-line block sits between sentences, which `fixed` and `structural`
    never generate. On the real corpus that leaves 1,522 characters outside the
    chunks across 116 documents, all of it blank. Losing a whitespace run is fine;
    losing a character of prose is not, so assert the strong version.
    """
    text = (
        sentences("A", 3)
        + "\n   \n  \n   \n     \n   \n"  # blank-line block between topics
        + sentences("B", 3)
        + "\n\n \n"
        + sentences("C", 3)
    )
    chunks = chunking.build_chunks(
        doc(text), "semantic", target=100_000, overlap=0, embedder=ScriptedEmbedder()
    )
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    lost = [i for i, ch in enumerate(text) if i not in covered and not ch.isspace()]
    assert lost == [], f"lost non-whitespace at offsets {lost[:20]}"


def test_overlap_is_documented_as_unused_not_silently_applied():
    """Overlapping across a boundary re-introduces the bleed the method exists to
    remove, so `overlap` is accepted for signature compatibility and ignored."""
    text = sentences("A", 3) + " " + sentences("B", 3)
    with_overlap = semantic.chunk_semantic(
        doc(text), target=800, overlap=300, embedder=ScriptedEmbedder()
    )
    without = semantic.chunk_semantic(doc(text), target=800, overlap=0, embedder=ScriptedEmbedder())
    assert with_overlap == without


@pytest.mark.parametrize("percentile", [50.0, 75.0, 95.0])
def test_tiling_holds_across_percentiles(percentile):
    text = " ".join(sentences(m, 2) for m in "ABCA")
    spans = semantic.chunk_semantic(
        doc(text), target=100_000, overlap=0, embedder=ScriptedEmbedder(), percentile=percentile
    )
    assert_tiles([(s, e) for s, e, _ in spans], len(text))
