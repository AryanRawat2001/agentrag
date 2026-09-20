"""Tests for chunking strategies.

The invariant that matters most: `chunk.text` must equal
`extracted.text[chunk.start:chunk.end]` exactly. Phase 5's citation verifier
asserts that a cited span exists in the source document, so a chunk whose text has
drifted from its offsets breaks citation verification rather than chunking.
"""

from __future__ import annotations

import pytest

from ragpipe import chunking


def _doc(text: str, sections: list[dict] | None = None, pages: list[dict] | None = None) -> dict:
    if pages is None:
        pages = [{"page": 1, "start": 0, "end": len(text), "chars": len(text)}]
    return {
        "doc_id": "test-doc",
        "source": "test",
        "text": text,
        "pages": pages,
        "sections": sections or [],
        "identifiers": [],
        "metadata": {},
    }


PROSE = (
    "This is the first paragraph of a document about regulatory compliance. "
    "It continues for a while to give the chunker something to work with.\n\n"
    "The second paragraph discusses reporting obligations under the relevant rule. "
    "It also runs on for a similar length so that windows must split somewhere.\n\n"
) * 12


class TestSpanExactness:
    @pytest.mark.parametrize("strategy", ["fixed", "structural"])
    def test_text_equals_its_own_source_slice(self, strategy):
        doc = _doc(
            PROSE,
            sections=[
                {
                    "idx": 0,
                    "heading": "I. Scope",
                    "level": 1,
                    "page": 1,
                    "start": 0,
                    "end": len(PROSE),
                }
            ],
        )
        for c in chunking.build_chunks(doc, strategy):
            assert c.text == doc["text"][c.start : c.end]

    @pytest.mark.parametrize("strategy", ["fixed", "structural"])
    def test_spans_cover_all_text(self, strategy):
        doc = _doc(PROSE)
        covered = set()
        for c in chunking.build_chunks(doc, strategy):
            covered.update(range(c.start, c.end))
        assert len(covered) == len(PROSE)

    @pytest.mark.parametrize("strategy", ["fixed", "structural"])
    def test_spans_are_ordered_and_non_negative(self, strategy):
        chunks = chunking.build_chunks(_doc(PROSE), strategy)
        for c in chunks:
            assert c.end > c.start
        assert [c.start for c in chunks] == sorted(c.start for c in chunks)


class TestTextVersusEmbedText:
    def test_embed_text_carries_the_heading_but_text_does_not(self):
        """The two fields exist for different consumers: `text` must stay a byte-exact
        source slice for citation verification, while `embed_text` gets the heading
        because a chunk is far more retrievable when it carries its section title."""
        body = "Reports must be submitted within fifteen days. " * 30
        doc = _doc(
            body,
            sections=[
                {
                    "idx": 0,
                    "heading": "Subpart C — Reporting",
                    "level": 1,
                    "page": 1,
                    "start": 0,
                    "end": len(body),
                }
            ],
        )
        chunk = chunking.build_chunks(doc, "structural")[0]
        assert "Subpart C" not in chunk.text
        assert chunk.embed_text.startswith("Subpart C — Reporting")
        assert chunk.text == doc["text"][chunk.start : chunk.end]

    def test_embed_text_equals_text_when_there_is_no_heading(self):
        chunk = chunking.build_chunks(_doc(PROSE), "fixed")[0]
        assert chunk.embed_text == chunk.text


class TestSizing:
    @pytest.mark.parametrize("strategy", ["fixed", "structural"])
    def test_no_chunk_below_the_minimum(self, strategy):
        """A front-matter window shorter than the minimum used to be emitted as its
        own 6-character chunk; it is now folded into the first section."""
        text = "Hdr\n\n" + PROSE  # tiny pre-section preamble
        doc = _doc(
            text,
            sections=[
                {
                    "idx": 0,
                    "heading": "I. Scope",
                    "level": 1,
                    "page": 1,
                    "start": 5,
                    "end": len(text),
                }
            ],
        )
        sizes = [c.n_chars for c in chunking.build_chunks(doc, strategy)]
        assert min(sizes) >= chunking.MIN_CHUNK_CHARS

    def test_target_size_is_respected_within_snap_tolerance(self):
        chunks = chunking.build_chunks(_doc(PROSE), "fixed", target=800, overlap=100)
        assert max(c.n_chars for c in chunks) <= 800 + chunking.BOUNDARY_SEARCH_CHARS

    def test_short_document_yields_one_chunk(self):
        doc = _doc("A short regulatory note about nothing in particular.")
        assert len(chunking.build_chunks(doc, "fixed")) == 1

    def test_whitespace_only_document_yields_no_chunks(self):
        assert chunking.build_chunks(_doc("   \n\n  \t "), "fixed") == []

    def test_empty_document_yields_no_chunks(self):
        assert chunking.build_chunks(_doc(""), "fixed") == []


class TestStructuralFallback:
    def test_falls_back_to_fixed_when_there_are_no_sections(self):
        doc = _doc(PROSE, sections=[])
        chunks = chunking.build_chunks(doc, "structural")
        assert chunks
        assert all(c.structural_fallback for c in chunks)
        assert all(c.section_idx is None for c in chunks)

    def test_fallback_flag_is_false_when_structure_exists(self):
        doc = _doc(
            PROSE,
            sections=[
                {
                    "idx": 0,
                    "heading": "I. Scope",
                    "level": 1,
                    "page": 1,
                    "start": 0,
                    "end": len(PROSE),
                }
            ],
        )
        assert not any(c.structural_fallback for c in chunking.build_chunks(doc, "structural"))

    def test_section_index_is_none_not_zero_for_unsectioned_spans(self):
        """`pending_idx or 0` mislabelled an absent section as section 0."""
        text = ("x" * 500) + PROSE
        doc = _doc(
            text,
            sections=[
                {
                    "idx": 7,
                    "heading": "VII. Later",
                    "level": 1,
                    "page": 1,
                    "start": 500,
                    "end": len(text),
                }
            ],
        )
        chunks = chunking.build_chunks(doc, "structural")
        preamble = [c for c in chunks if c.start < 500]
        assert preamble
        assert all(c.section_idx is None for c in preamble)


class TestIdentifierRebasing:
    def test_identifier_offsets_are_rebased_onto_the_chunk(self):
        body = ("filler " * 40) + "see 21 CFR 312.32 for detail. " + ("filler " * 40)
        pos = body.index("21 CFR 312.32")
        doc = _doc(body)
        doc["identifiers"] = [
            {
                "kind": "cfr",
                "raw": "21 CFR 312.32",
                "canonical": "21 CFR 312.32",
                "canonical_detailed": "21 CFR 312.32",
                "start": pos,
                "end": pos + len("21 CFR 312.32"),
                "repaired": False,
            }
        ]
        chunks = chunking.build_chunks(doc, "fixed")
        holder = next(c for c in chunks if c.identifiers)
        ident = holder.identifiers[0]
        assert holder.text[ident["start"] : ident["end"]] == "21 CFR 312.32"

    def test_identifier_straddling_a_boundary_is_excluded(self):
        doc = _doc(PROSE)
        doc["identifiers"] = [
            {
                "kind": "cfr",
                "raw": "x",
                "canonical": "x",
                "canonical_detailed": "x",
                "start": 0,
                "end": len(PROSE),
                "repaired": False,
            }
        ]
        chunks = chunking.build_chunks(doc, "fixed", target=500, overlap=50)
        # Only a chunk fully containing the span may claim it; none can here.
        assert all(not c.identifiers for c in chunks)


class TestParameterValidation:
    """Both of these degraded silently rather than failing, which is worse."""

    def test_target_below_minimum_is_rejected(self):
        """With target <= MIN_CHUNK_CHARS every window is a runt, so `_absorb_runts`
        folded an entire document into a single chunk — a 9,600-character "chunk"
        from a 150-character target."""
        with pytest.raises(ValueError, match="MIN_CHUNK_CHARS"):
            chunking.build_chunks(_doc(PROSE), "fixed", target=150, overlap=20)

    def test_overlap_at_or_above_target_is_rejected(self):
        """Overlap >= target advanced the cursor one character at a time, emitting a
        span per character with near-total duplication."""
        with pytest.raises(ValueError, match="less than target"):
            chunking.build_chunks(_doc(PROSE), "fixed", target=2048, overlap=2048)

    def test_negative_overlap_is_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            chunking.build_chunks(_doc(PROSE), "fixed", target=2048, overlap=-1)

    def test_valid_parameters_are_accepted(self):
        assert chunking.build_chunks(_doc(PROSE), "fixed", target=800, overlap=100)


class TestGapCoverage:
    """`chunk_structural` advanced only from section starts, so any gap between
    sections — or text past the final section — was silently dropped. The *leading*
    case was handled explicitly; these are the same problem."""

    def test_gap_between_sections_is_covered(self):
        text = ("A" * 600) + ("GAP " * 200) + ("B" * 600)
        doc = _doc(
            text,
            sections=[
                {
                    "idx": 0,
                    "heading": "I. First Section Here",
                    "level": 1,
                    "page": 1,
                    "start": 0,
                    "end": 600,
                },
                {
                    "idx": 1,
                    "heading": "II. Second Section Here",
                    "level": 1,
                    "page": 1,
                    "start": 1400,
                    "end": 2000,
                },
            ],
        )
        covered = set()
        for c in chunking.build_chunks(doc, "structural"):
            covered.update(range(c.start, c.end))
        assert len(covered) == len(text)

    def test_front_matter_is_emitted_exactly_once(self):
        """Regression: handling the leading region both in a special case and in the
        gap loop emitted it twice, giving 104 documents duplicate spans."""
        text = ("F" * 1500) + ("B" * 3000)
        doc = _doc(
            text,
            sections=[
                {
                    "idx": 0,
                    "heading": "I. The Only Section Here",
                    "level": 1,
                    "page": 1,
                    "start": 1500,
                    "end": len(text),
                }
            ],
        )
        spans = [(c.start, c.end) for c in chunking.build_chunks(doc, "structural")]
        assert len(spans) == len(set(spans)), f"duplicate spans: {spans}"

    def test_short_front_matter_is_folded_not_emitted_alone(self):
        text = ("F" * 50) + ("B" * 3000)
        doc = _doc(
            text,
            sections=[
                {
                    "idx": 0,
                    "heading": "I. The Only Section Here",
                    "level": 1,
                    "page": 1,
                    "start": 50,
                    "end": len(text),
                }
            ],
        )
        chunks = chunking.build_chunks(doc, "structural")
        assert chunks[0].start == 0
        assert min(c.n_chars for c in chunks) >= chunking.MIN_CHUNK_CHARS

    def test_text_after_the_last_section_is_covered(self):
        text = ("A" * 600) + ("TAIL " * 200)
        doc = _doc(
            text,
            sections=[
                {
                    "idx": 0,
                    "heading": "I. Only Section Here",
                    "level": 1,
                    "page": 1,
                    "start": 0,
                    "end": 600,
                }
            ],
        )
        covered = set()
        for c in chunking.build_chunks(doc, "structural"):
            covered.update(range(c.start, c.end))
        assert len(covered) == len(text)


class TestDenseIndex:
    def test_idx_has_no_holes_when_whitespace_spans_are_dropped(self):
        """`idx` was assigned by enumerating raw spans, so dropping a whitespace-only
        span left a hole — which would break neighbour expansion by `idx +/- 1`."""
        text = ("word " * 300) + (" " * 3000) + ("other " * 300)
        doc = _doc(text)
        chunks = chunking.build_chunks(doc, "fixed", target=600, overlap=60)
        assert [c.idx for c in chunks] == list(range(len(chunks)))


class TestStrategyRegistry:
    def test_semantic_is_now_implemented(self):
        """Was deferred through Phase 2 for want of an embedding model; Phase 3
        supplied one. Asserted so the registry and the docs cannot disagree."""
        assert "semantic" in chunking.STRATEGIES
        assert "semantic" not in chunking.DEFERRED_STRATEGIES

    def test_semantic_demands_an_embedder_instead_of_faking_one(self):
        """Falling back to a size-based split would put a row labelled 'semantic'
        in the comparison table that was nothing of the kind."""
        with pytest.raises(ValueError, match="requires an embedder"):
            chunking.build_chunks(_doc(PROSE), "semantic")

    def test_embedding_strategies_are_all_registered(self):
        assert set(chunking.STRATEGIES) >= chunking.EMBEDDING_STRATEGIES

    def test_unknown_strategy_raises(self):
        with pytest.raises(KeyError):
            chunking.build_chunks(_doc(PROSE), "nonsense")

    def test_deferred_and_implemented_do_not_overlap(self):
        assert not set(chunking.STRATEGIES) & set(chunking.DEFERRED_STRATEGIES)

    def test_deferred_strategy_still_raises_when_one_exists(self, monkeypatch):
        """The deferral mechanism outlives its last user: it is how a future
        unimplemented strategy stays honest rather than shipping a fake row."""
        monkeypatch.setitem(chunking.DEFERRED_STRATEGIES, "future", "needs a thing we lack")
        monkeypatch.setitem(chunking.STRATEGIES, "future", chunking.chunk_fixed)
        with pytest.raises(NotImplementedError, match="needs a thing we lack"):
            chunking.build_chunks(_doc(PROSE), "future")


class TestContentHash:
    def test_reflowed_whitespace_hashes_the_same(self):
        a = "The applicant shall submit\nthe report within 15 days."
        b = "The applicant shall submit the report   within 15 days."
        assert chunking.content_hash(a) == chunking.content_hash(b)

    def test_case_differences_hash_the_same(self):
        assert chunking.content_hash("Report Required") == chunking.content_hash("report required")

    def test_different_text_hashes_differently(self):
        assert chunking.content_hash("15 days") != chunking.content_hash("30 days")
