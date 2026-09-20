"""Tests for tier-1 citation verification.

Phase 5 gets the stricter verification gate, and this module is why: a citation
verifier that accepts a fabricated quote produces a faithfulness table that looks
excellent and means nothing. So the tests here lean hard on the *rejection* paths
and on the offset arithmetic, not just the happy path.

The whitespace fixture is a verbatim prefix of a real chunk in
`data/chunks/structural.jsonl` (`ctgov-NCT00567567-Prot_SAP_000::structural::00000`).
Synthetic whitespace would not reproduce the `' \\n \\n'` runs that pypdf actually
emits, and those runs are the entire reason the normalized tier exists.
"""

from __future__ import annotations

import pytest

from ragpipe import citations as cit

# Verbatim from the corpus -- note the trailing space before each newline, the
# lone-space line, and the six-space run before "Version Date".
REAL_TEXT = (
    "ANBL0532 \nPage 1 \n \nActivated: 11/05/07      Version Date: 08/16/11 \n"
    "Children's Oncology Group protocol for high-risk neuroblastoma. "
    "Patients must have adequate renal function defined as a serum creatinine "
    "within institutional normal limits for age."
)

# Verified exact substrings of REAL_TEXT, derived by searching it rather than
# written by eye -- an earlier draft of this file invented "adequate renal function
# within", which reads plausibly but skips "defined as a serum creatinine".
EXACT_QUOTE = "adequate renal function defined as a serum creatinine"


def _chunk(chunk_id: str, text: str, doc_id: str = "doc-a") -> dict:
    return {"chunk_id": chunk_id, "doc_id": doc_id, "text": text}


class TestNormalizeWhitespace:
    def test_collapses_runs_to_single_space(self):
        norm, _ = cit.normalize_whitespace("a  \n \t b")
        assert norm == "a b"

    def test_index_map_length_matches_normalized(self):
        norm, imap = cit.normalize_whitespace(REAL_TEXT)
        assert len(norm) == len(imap)

    def test_index_map_points_at_real_characters(self):
        """Every non-space normalized character must map to the same character
        in the source. This is the invariant the whole normalized tier rests on."""
        norm, imap = cit.normalize_whitespace(REAL_TEXT)
        for i, ch in enumerate(norm):
            if ch != " ":
                assert REAL_TEXT[imap[i]] == ch, f"mismatch at normalized {i}"

    def test_collapsed_space_anchors_at_run_start(self):
        norm, imap = cit.normalize_whitespace("ab   cd")
        assert norm == "ab cd"
        assert imap[2] == 2  # the synthesized space points at the run's first char
        assert imap[3] == 5  # 'c'

    def test_strips_leading_and_trailing_whitespace(self):
        norm, imap = cit.normalize_whitespace("   hi   ")
        assert norm == "hi"
        assert imap == [3, 4]

    def test_never_ends_on_synthesized_space(self):
        """A trailing collapsed space would make the end-offset arithmetic
        under-cover, since its map entry points at the run's start."""
        for raw in ("hi   ", "hi \n \n", REAL_TEXT + "  \n  "):
            norm, _ = cit.normalize_whitespace(raw)
            assert not norm.endswith(" ")

    def test_whitespace_only_input(self):
        assert cit.normalize_whitespace("  \n\t ") == ("", [])

    def test_empty_input(self):
        assert cit.normalize_whitespace("") == ("", [])


class TestLocateQuote:
    def test_exact_match_reports_exact(self):
        quote = "adequate renal function"
        m = cit.locate_quote(quote, REAL_TEXT)
        assert m is not None
        assert m.method == cit.EXACT
        assert m.is_exact
        assert REAL_TEXT[m.start : m.end] == quote

    def test_normalized_match_across_pdf_whitespace(self):
        """The failure the tier exists for: a model collapses ' \\n \\n' when quoting."""
        quote = "Page 1 Activated: 11/05/07 Version Date: 08/16/11"
        assert quote not in REAL_TEXT  # exact tier genuinely cannot find it
        m = cit.locate_quote(quote, REAL_TEXT)
        assert m is not None
        assert m.method == cit.NORMALIZED

    def test_normalized_match_offsets_recover_real_span(self):
        """A normalized match must still name a real span -- its offsets, applied
        to the untouched source, must normalize back to the quote."""
        quote = "Page 1 Activated: 11/05/07 Version Date: 08/16/11"
        m = cit.locate_quote(quote, REAL_TEXT)
        assert m is not None
        assert m.matched_text == REAL_TEXT[m.start : m.end]
        assert cit.normalize_whitespace(m.matched_text)[0] == quote

    def test_matched_text_differs_from_quote_under_normalization(self):
        """What the document says is not what the model wrote -- callers showing
        provenance to a user must be able to tell."""
        quote = "Page 1 Activated: 11/05/07 Version Date: 08/16/11"
        m = cit.locate_quote(quote, REAL_TEXT)
        assert m is not None and m.matched_text != quote
        assert "\n" in m.matched_text

    def test_absent_quote_returns_none(self):
        assert cit.locate_quote("the maximum tolerated dose was 40 mg", REAL_TEXT) is None

    def test_case_change_is_not_tolerated(self):
        """Normalization stops at whitespace by design."""
        assert cit.locate_quote("ADEQUATE RENAL FUNCTION", REAL_TEXT) is None

    def test_punctuation_change_is_not_tolerated(self):
        assert cit.locate_quote("Children s Oncology Group", REAL_TEXT) is None

    def test_word_insertion_is_not_tolerated(self):
        assert cit.locate_quote("adequate hepatic renal function", REAL_TEXT) is None

    def test_empty_inputs(self):
        assert cit.locate_quote("", REAL_TEXT) is None
        assert cit.locate_quote("something", "") is None

    def test_whitespace_only_quote(self):
        assert cit.locate_quote("   \n  ", REAL_TEXT) is None

    def test_exact_tier_preferred_over_normalized(self):
        """When both tiers could match, the exact offset wins -- otherwise the
        reported method would understate how faithful the model actually was."""
        source = "alpha beta gamma  alpha beta gamma"
        m = cit.locate_quote("alpha beta gamma", source)
        assert m is not None and m.method == cit.EXACT and m.start == 0


class TestVerifyCitation:
    def setup_method(self):
        self.chunks = [
            _chunk("c1", REAL_TEXT),
            _chunk("c2", "Unrelated section about statistical analysis plans and power."),
            _chunk("c3", "A different document entirely.", doc_id="doc-b"),
        ]
        self.index = {c["chunk_id"]: c for c in self.chunks}
        self.by_doc = cit._chunks_by_doc(self.chunks)

    def _verify(self, chunk_id, quote, **kw):
        return cit.verify_citation(chunk_id, quote, self.index, by_doc=self.by_doc, **kw)

    def test_verified_exact(self):
        r = self._verify("c1", "Patients must have adequate renal function")
        assert r.method == cit.EXACT and r.verified

    def test_short_quote_is_its_own_bucket_not_verified(self):
        """'the' appears everywhere; accepting it would inflate precision."""
        r = self._verify("c1", "Page 1")
        assert r.method == cit.TOO_SHORT
        assert not r.verified

    def test_short_quote_checked_before_lookup(self):
        """Ordering matters: a short quote against an unknown chunk is still
        too_short, so the bucket cannot be confused with fabrication."""
        r = self._verify("nonexistent", "tiny")
        assert r.method == cit.TOO_SHORT

    def test_min_quote_chars_is_configurable(self):
        # 17 chars, below the default 24. It matches only under the normalized
        # tier: the source has an extra ' \n' between "Page 1 " and "Activated".
        quote = "Page 1 \nActivated"
        assert self._verify("c1", quote).method == cit.TOO_SHORT
        assert self._verify("c1", quote, min_quote_chars=8).method == cit.NORMALIZED

    def test_wrong_chunk_when_quote_lives_in_a_sibling(self):
        """Real text, wrong provenance -- a plumbing bug, not a hallucination."""
        r = self._verify("c2", "Patients must have adequate renal function")
        assert r.method == cit.WRONG_CHUNK
        assert r.found_in_chunk_id == "c1"
        assert not r.verified

    def test_wrong_chunk_does_not_cross_document_boundaries(self):
        """c3 is in doc-b, so a doc-a quote must not be found for it."""
        r = self._verify("c3", "Patients must have adequate renal function")
        assert r.method == cit.UNVERIFIED

    def test_fabricated_quote_is_unverified(self):
        r = self._verify("c1", "The maximum tolerated dose was determined to be 40 mg.")
        assert r.method == cit.UNVERIFIED
        assert not r.verified

    def test_unknown_chunk_id_is_unverified(self):
        r = self._verify("does-not-exist", "Patients must have adequate renal function")
        assert r.method == cit.UNVERIFIED

    def test_without_by_doc_misattribution_collapses_to_unverified(self):
        r = cit.verify_citation("c2", "Patients must have adequate renal function", self.index)
        assert r.method == cit.UNVERIFIED

    def test_quote_is_stripped_before_matching(self):
        r = self._verify("c1", "  \n Patients must have adequate renal function \n ")
        assert r.method == cit.EXACT

    def test_offsets_index_the_cited_chunk_text(self):
        quote = "Patients must have adequate renal function"
        r = self._verify("c1", quote)
        assert REAL_TEXT[r.start : r.end] == quote


class TestVerificationReport:
    def _report(self, citations):
        chunks = [_chunk("c1", REAL_TEXT), _chunk("c2", "Statistical analysis plan section.")]
        return cit.verify_answer(citations, chunks)

    def test_counts_include_every_bucket_even_at_zero(self):
        rep = self._report([{"chunk_id": "c1", "quote": EXACT_QUOTE}])
        assert set(rep.counts) == {
            cit.EXACT,
            cit.NORMALIZED,
            cit.LINE_NUMBER_AMBIGUOUS,
            cit.WRONG_CHUNK,
            cit.UNVERIFIED,
            cit.TOO_SHORT,
        }
        assert rep.counts[cit.EXACT] == 1
        assert rep.counts[cit.UNVERIFIED] == 0

    def test_counts_sum_to_claimed(self):
        rep = self._report(
            [
                {"chunk_id": "c1", "quote": EXACT_QUOTE},
                {"chunk_id": "c1", "quote": "fabricated text that is long enough"},
                {"chunk_id": "c1", "quote": "short"},
                {"chunk_id": "c2", "quote": "Patients must have adequate renal function"},
            ]
        )
        assert sum(rep.counts.values()) == rep.n_claimed == 4
        assert rep.counts[cit.EXACT] == 1
        assert rep.counts[cit.UNVERIFIED] == 1
        assert rep.counts[cit.TOO_SHORT] == 1
        assert rep.counts[cit.WRONG_CHUNK] == 1

    def test_precision_is_verified_over_claimed(self):
        rep = self._report(
            [
                {"chunk_id": "c1", "quote": EXACT_QUOTE},
                {"chunk_id": "c1", "quote": "fabricated text that is long enough"},
            ]
        )
        assert rep.citation_precision == pytest.approx(0.5)

    def test_precision_is_none_not_zero_when_nothing_claimed(self):
        """A refusal must not drag down a mean measuring citation quality."""
        rep = self._report([])
        assert rep.citation_precision is None
        assert rep.n_claimed == 0

    def test_too_short_counts_against_precision(self):
        """It is not verified, so it must not be silently excluded either."""
        rep = self._report(
            [
                {"chunk_id": "c1", "quote": EXACT_QUOTE},
                {"chunk_id": "c1", "quote": "the"},
            ]
        )
        assert rep.citation_precision == pytest.approx(0.5)

    def test_fully_grounded_requires_at_least_one_citation(self):
        assert self._report([]).fully_grounded is False
        assert self._report([{"chunk_id": "c1", "quote": EXACT_QUOTE}]).fully_grounded is True

    def test_fully_grounded_false_with_any_failure(self):
        rep = self._report(
            [
                {"chunk_id": "c1", "quote": EXACT_QUOTE},
                {"chunk_id": "c1", "quote": "fabricated text that is long enough"},
            ]
        )
        assert rep.fully_grounded is False

    def test_missing_keys_do_not_crash(self):
        """A malformed model response is a data point, not an exception."""
        rep = self._report([{}, {"chunk_id": "c1"}, {"quote": "x"}])
        assert rep.n_claimed == 3
        assert rep.n_verified == 0

    def test_verification_is_scoped_to_shown_context(self):
        """A quote from elsewhere in the corpus is still a fabrication for
        *this* answer -- the model could not have seen it."""
        rep = cit.verify_answer(
            [{"chunk_id": "c1", "quote": "Patients must have adequate renal function"}],
            [_chunk("c1", "Only this short text was retrieved for this query.")],
        )
        assert rep.counts[cit.UNVERIFIED] == 1


class TestLineNumberAmbiguousTier:
    """The tier that was briefly counted as verified, and must never be again.

    Fixtures are verbatim from `fda-184904::structural::00012` and
    `fda-106721::structural::00009` -- FDA guidances typeset with line numbers that
    pypdf extracts *inside* the sentence.
    """

    DRAFT = (
        "ropic virus, Treponema 28 \npallidum (syphilis), vaccinia virus, West Nile "
        "virus, and communicable disease risks associated 29 \nwith "
        "xenotransplantation.   30 \n 31 \nThis guidance applies to h"
    )
    # Real text where a content number sits at a line boundary -- the case that
    # falsified the tier's safety premise.
    CFR = (
        "subject to 21 CFR part 123, Fish and Fishery Products or 21 \nCFR part 120, "
        "Hazard Analysis and Critical Control Point (HACCP) Systems, must comply."
    )

    def test_hit_is_reported_but_not_verified(self):
        m = cit.locate_quote(
            "Treponema pallidum (syphilis), vaccinia virus, West Nile virus", self.DRAFT
        )
        assert m is not None
        assert m.method == cit.LINE_NUMBER_AMBIGUOUS
        assert cit.LINE_NUMBER_AMBIGUOUS not in cit.VERIFIED_METHODS

    def test_does_not_count_toward_citation_precision(self):
        rep = cit.verify_answer(
            [
                {
                    "chunk_id": "c1",
                    "quote": "Treponema pallidum (syphilis), vaccinia virus, West Nile virus",
                }
            ],
            [_chunk("c1", self.DRAFT)],
        )
        assert rep.counts[cit.LINE_NUMBER_AMBIGUOUS] == 1
        assert rep.n_verified == 0
        assert rep.citation_precision == 0.0
        assert rep.fully_grounded is False

    def test_changing_a_cfr_title_is_not_verified(self):
        """The reproduction that demoted this tier. The document says 21 CFR part 120;
        a model asserting 40 CFR must not be scored as having cited the document."""
        altered = self.CFR.replace("or 21 \nCFR part 120", "or 40 \nCFR part 120")
        m = cit.locate_quote(altered, self.CFR)
        assert m is None or m.method not in cit.VERIFIED_METHODS

    def test_omitting_a_stated_number_is_not_verified(self):
        src = "The sponsor must respond within 30 \ndays of receiving the request."
        m = cit.locate_quote("The sponsor must respond within days of receiving", src)
        assert m is None or m.method not in cit.VERIFIED_METHODS

    def test_stricter_tiers_still_win(self):
        m = cit.locate_quote("This guidance applies to h", self.DRAFT)
        assert m is not None and m.method == cit.EXACT

    def test_whitespace_tier_still_wins(self):
        m = cit.locate_quote("Page 1 Activated: 11/05/07 Version Date: 08/16/11", REAL_TEXT)
        assert m is not None and m.method == cit.NORMALIZED

    def test_fabricated_quote_still_fails_outright(self):
        assert cit.locate_quote("The maximum tolerated dose was 40 mg per day", self.DRAFT) is None

    def test_span_covers_real_source_including_the_line_number(self):
        m = cit.locate_quote(
            "communicable disease risks associated with xenotransplantation", self.DRAFT
        )
        assert m is not None
        assert m.matched_text == self.DRAFT[m.start : m.end]
        assert "29" in m.matched_text

    def test_numbers_not_adjacent_to_a_line_break_survive(self):
        """Scoped honestly: this holds only for numbers away from a line boundary.
        Two earlier versions of this test claimed the general property and could not
        fail -- the first had no newline at all, the second put a comma between the
        number and the newline, so neither exercised any line-number pattern."""
        src = "shall report within 15 days under 21 CFR 211.192 as required."
        norm, _ = cit.normalize_line_numbers(src)
        assert "15" in norm and "211.192" in norm

    def test_line_adjacent_content_numbers_ARE_deleted(self):
        """The measured defect, pinned. This normalizer cannot tell a line number from
        a CFR title that pypdf split across a line break, so it deletes both. That is
        the entire reason a hit here is not counted as verified -- if this assertion
        ever needs changing, `VERIFIED_METHODS` must be re-examined first."""
        norm, _ = cit.normalize_line_numbers(self.CFR)
        assert "or CFR part 120" in norm, "the line-adjacent 21 is deleted"
        assert cit.LINE_NUMBER_AMBIGUOUS not in cit.VERIFIED_METHODS

    def test_number_change_at_a_line_break_is_not_verified(self):
        """It may still be *detected* by the diagnostic tier, but it must never come
        back as verified."""
        src = "shall report within 15 \ndays of becoming aware of the event."
        m = cit.locate_quote("shall report within 30 days of becoming aware", src)
        assert m is None or m.method not in cit.VERIFIED_METHODS

    def test_words_join_across_a_removed_line_number(self):
        norm, _ = cit.normalize_line_numbers(self.DRAFT)
        assert "Treponema pallidum" in norm
        assert "Treponemapallidum" not in norm

    def test_index_map_aligns_for_real_characters(self):
        norm, imap = cit.normalize_line_numbers(self.DRAFT)
        assert len(norm) == len(imap)
        for i, ch in enumerate(norm):
            if ch != " ":
                assert self.DRAFT[imap[i]] == ch

    def test_empty_and_whitespace_inputs(self):
        assert cit.normalize_line_numbers("") == ("", [])
        assert cit.normalize_line_numbers("  \n 12 \n ")[0] == ""
