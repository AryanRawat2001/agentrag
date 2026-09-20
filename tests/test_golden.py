"""Tests for golden answer-set drafting and validation.

Weighted toward *rejection*. A golden set sits upstream of every generation metric, so
a bad reference answer does not fail loudly — it makes a correct model look unfaithful
in a table nobody re-derives. The validator is the only thing standing between a
hallucinated reference and that outcome.

The chunk fixture is verbatim from `fda-184901::structural::00004`, an FDA draft
guidance, so it carries the interleaved legislative line numbers that Phase 5 found
break naive quote location.
"""

from __future__ import annotations

from ragpipe import citations as cit
from ragpipe import golden

REAL_TEXT = (
    "I. INTRODUCTION 15 \n 16 \nWe, FDA or Agency, are issuing this guidance to assist "
    "you, establishments making donor 17 \neligibility determinations, in complying "
    "with the applicable requirements. The sponsor 18 \nmust respond within 30 days of "
    "receiving a written request from the Agency."
)


def _chunk(chunk_id="c1", text=REAL_TEXT, doc_id="doc-a"):
    return {"chunk_id": chunk_id, "doc_id": doc_id, "text": text}


def _pair(**kw):
    # The answer is covered by its evidence. Fixtures predating the coverage check had
    # answers that outran their quotes -- realistic of the defect, useless as a fixture
    # for anything else, because every one of them now rejects for that reason first.
    base = dict(
        pair_id="doc-a::gold::000",
        question="How long does the sponsor have to respond to a written request?",
        answer=(
            "The sponsor must respond within 30 days of receiving a written request "
            "from the Agency."
        ),
        evidence=(
            {
                "chunk_id": "c1",
                "quote": (
                    "must respond within 30 days of receiving a written request from the Agency"
                ),
            },
        ),
        doc_id="doc-a",
        shape="quantity",
    )
    base.update(kw)
    return golden.GoldenPair(**base)


class TestValidationAcceptsGoodPairs:
    def test_a_well_formed_pair_is_accepted(self):
        r = golden.validate_pair(_pair(), [_chunk()])
        assert r.status == golden.STATUS_ACCEPTED
        assert r.accepted
        assert r.reject_reason == ""
        assert r.evidence_methods == (cit.EXACT,)

    def test_whitespace_normalised_evidence_is_still_accepted(self):
        """PDF artifacts are not the drafter's fault; tier 1 tolerates whitespace."""
        p = _pair(
            answer="This introduction is issued by FDA, also called the Agency.",
            evidence=({"chunk_id": "c1", "quote": "I. INTRODUCTION 15 16 We, FDA or Agency"},),
        )
        r = golden.validate_pair(p, [_chunk()])
        assert r.accepted
        assert r.evidence_methods == (cit.NORMALIZED,)

    def test_multiple_verified_quotes_are_accepted(self):
        p = _pair(
            answer=(
                "The Agency is issuing this guidance to assist you in complying with "
                "the applicable requirements."
            ),
            evidence=(
                {"chunk_id": "c1", "quote": "are issuing this guidance to assist you"},
                {"chunk_id": "c1", "quote": "in complying with the applicable requirements"},
            ),
        )
        assert golden.validate_pair(p, [_chunk()]).accepted


class TestValidationRejects:
    def test_fabricated_evidence_is_rejected(self):
        """The case the whole module exists for."""
        p = _pair(evidence=({"chunk_id": "c1", "quote": "The maximum tolerated dose is 40 mg"},))
        r = golden.validate_pair(p, [_chunk()])
        assert r.status == golden.STATUS_REJECTED
        assert "no evidence verified" in r.reject_reason

    def test_partially_verified_evidence_is_rejected_not_salvaged(self):
        """Dropping the bad quote would keep a reference answer that may depend on the
        span which does not exist."""
        p = _pair(
            answer=(
                "The Agency is issuing this guidance to assist you, and it must be "
                "submitted within fourteen calendar days."
            ),
            evidence=(
                {"chunk_id": "c1", "quote": "are issuing this guidance to assist you"},
                {"chunk_id": "c1", "quote": "must be submitted within fourteen calendar days"},
            ),
        )
        r = golden.validate_pair(p, [_chunk()])
        assert r.status == golden.STATUS_REJECTED
        assert "1/2" in r.reject_reason

    def test_line_number_ambiguous_evidence_is_not_accepted(self):
        """Phase 5 demoted that tier because it blesses meaning changes. A golden set
        must not be built on evidence the verifier itself refuses to trust."""
        p = _pair(
            evidence=({"chunk_id": "c1", "quote": "The sponsor must respond within 30 days"},)
        )
        r = golden.validate_pair(p, [_chunk()])
        assert cit.LINE_NUMBER_AMBIGUOUS in r.evidence_methods
        assert r.status == golden.STATUS_REJECTED

    def test_short_answer_is_rejected(self):
        r = golden.validate_pair(_pair(answer="Establishments."), [_chunk()])
        assert "under" in r.reject_reason

    def test_answer_restating_the_question_is_rejected(self):
        q = "Who is this guidance intended to assist, in full and at length please now?"
        r = golden.validate_pair(_pair(question=q, answer=q), [_chunk()])
        assert r.reject_reason == "answer restates the question"

    def test_empty_fields_are_rejected_by_name(self):
        assert (
            golden.validate_pair(_pair(question=""), [_chunk()]).reject_reason == "empty question"
        )
        assert golden.validate_pair(_pair(answer=""), [_chunk()]).reject_reason == "empty answer"
        assert (
            golden.validate_pair(_pair(evidence=()), [_chunk()]).reject_reason
            == "no evidence supplied"
        )

    def test_evidence_citing_an_unknown_chunk_is_rejected(self):
        p = _pair(evidence=({"chunk_id": "nope", "quote": "are issuing this guidance to assist"},))
        assert golden.validate_pair(p, [_chunk()]).status == golden.STATUS_REJECTED

    def test_rejection_never_silently_becomes_acceptance(self):
        """Property: nothing in the reject path can return an accepted pair."""
        bad = [
            _pair(question=""),
            _pair(answer=""),
            _pair(answer="tiny"),
            _pair(evidence=()),
            _pair(evidence=({"chunk_id": "c1", "quote": "invented text that is long enough"},)),
        ]
        for p in bad:
            assert not golden.validate_pair(p, [_chunk()]).accepted


class TestParsing:
    def test_parses_a_batch_and_ids_are_unique(self):
        payload = {
            "pairs": [
                {
                    "shape": "scope",
                    "question": f"q{i}",
                    "answer": f"a{i}",
                    "evidence": [{"chunk_id": "c1", "quote": "x"}],
                }
                for i in range(10)
            ]
        }
        pairs = golden.parse_pairs(payload, "doc-a")
        assert len(pairs) == 10
        assert len({p.pair_id for p in pairs}) == 10

    def test_start_index_prevents_id_collision_across_calls(self):
        payload = {"pairs": [{"shape": "scope", "question": "q", "answer": "a", "evidence": []}]}
        first = golden.parse_pairs(payload, "doc-a", start_index=0)[0]
        second = golden.parse_pairs(payload, "doc-a", start_index=10)[0]
        assert first.pair_id != second.pair_id

    def test_malformed_entries_are_skipped_not_fatal(self):
        """Nine usable pairs out of ten beats zero."""
        payload = {
            "pairs": [
                {
                    "shape": "scope",
                    "question": "q",
                    "answer": "a",
                    "evidence": [{"chunk_id": "c1", "quote": "x"}],
                },
                "garbage",
                None,
            ]
        }
        assert len(golden.parse_pairs(payload, "doc-a")) == 1

    def test_missing_pairs_key_yields_nothing(self):
        assert golden.parse_pairs({}, "doc-a") == []


class TestReviewSampling:
    def _pairs(self):
        out = []
        for doc in ("doc-a", "doc-b", "doc-c"):
            for i in range(5):
                out.append(
                    _pair(
                        pair_id=f"{doc}::gold::{i:03d}", doc_id=doc, status=golden.STATUS_ACCEPTED
                    )
                )
        return out

    def test_sample_spreads_across_documents(self):
        picked = golden.sample_for_review(self._pairs(), 6)
        assert len(picked) == 6
        assert len({p.doc_id for p in picked}) == 3, "a prefix would give one document"

    def test_sample_is_deterministic(self):
        a = golden.sample_for_review(self._pairs(), 7)
        b = golden.sample_for_review(self._pairs(), 7)
        assert [p.pair_id for p in a] == [p.pair_id for p in b]

    def test_rejected_pairs_are_never_sampled(self):
        pairs = self._pairs() + [
            _pair(pair_id="doc-d::gold::000", doc_id="doc-d", status=golden.STATUS_REJECTED)
        ]
        assert all(p.doc_id != "doc-d" for p in golden.sample_for_review(pairs, 20))

    def test_asking_for_more_than_exists_returns_what_exists(self):
        assert len(golden.sample_for_review(self._pairs(), 99)) == 15


class TestYieldReport:
    def test_reports_acceptance_and_reasons(self):
        pairs = [
            _pair(pair_id="a", status=golden.STATUS_ACCEPTED, evidence_methods=(cit.EXACT,)),
            _pair(
                pair_id="b",
                status=golden.STATUS_REJECTED,
                reject_reason="no evidence verified (methods: unverified)",
            ),
            _pair(pair_id="c", status=golden.STATUS_REJECTED, reject_reason="empty answer"),
        ]
        r = golden.yield_report(pairs)
        assert r["n_drafted"] == 3 and r["n_accepted"] == 1 and r["n_rejected"] == 2
        assert r["acceptance_rate"] == 0.3333
        assert r["reject_reasons"]["no evidence verified"] == 1
        assert r["reject_reasons"]["empty answer"] == 1

    def test_empty_input_has_no_rate_rather_than_zero(self):
        assert golden.yield_report([])["acceptance_rate"] is None


class TestRoundTrip:
    def test_pairs_survive_write_and_load(self, tmp_path):
        pairs = [
            _pair(pair_id="a", status=golden.STATUS_ACCEPTED, evidence_methods=(cit.EXACT,)),
            _pair(pair_id="b", status=golden.STATUS_REJECTED, reject_reason="empty answer"),
        ]
        path = golden.write_golden(pairs, tmp_path / "golden.jsonl")
        back = golden.load_golden(path)
        assert [p.as_dict() for p in back] == [p.as_dict() for p in pairs]

    def test_rejects_are_persisted_too(self, tmp_path):
        """The reject log is the input to improving the prompt."""
        path = golden.write_golden(
            [_pair(pair_id="b", status=golden.STATUS_REJECTED, reject_reason="x")],
            tmp_path / "g.jsonl",
        )
        assert golden.load_golden(path)[0].status == golden.STATUS_REJECTED

    def test_missing_file_loads_empty(self, tmp_path):
        assert golden.load_golden(tmp_path / "absent.jsonl") == []


class TestPromptWiring:
    def test_both_floors_are_interpolated_from_their_constants(self):
        p = golden.draft_system_prompt()
        assert str(golden.MIN_ANSWER_CHARS) in p
        assert str(cit.MIN_QUOTE_CHARS) in p

    def test_prompt_warns_that_a_mention_is_not_a_statement(self):
        """The Phase 5 finding, carried into the drafting instructions."""
        assert "cites" in golden.draft_system_prompt()

    def test_schema_constrains_shape_to_the_known_set(self):
        shape = golden.GOLDEN_SCHEMA["properties"]["pairs"]["items"]["properties"]["shape"]
        enum = shape["enum"]
        assert set(enum) == set(golden.QUESTION_SHAPES)

    def test_draft_prompt_uses_text_not_embed_text(self):
        chunk = {**_chunk(), "embed_text": "HEADING PREPENDED " + REAL_TEXT}
        prompt = golden.build_draft_prompt("doc-a", [chunk], 10)
        assert "HEADING PREPENDED" not in prompt
        assert "I. INTRODUCTION" in prompt


class TestChunkSelection:
    def _doc(self, n, chars=1000):
        return [{"chunk_id": f"c{i}", "doc_id": "d", "text": "x" * chars} for i in range(n)]

    def test_small_document_is_passed_whole(self):
        doc = self._doc(10)
        assert golden.select_chunks_for_drafting(doc) == doc

    def test_large_document_is_sampled_within_budget(self):
        picked = golden.select_chunks_for_drafting(self._doc(765))
        total = sum(len(c["text"]) for c in picked)
        assert total <= golden.DRAFT_CHAR_BUDGET
        assert picked

    def test_sampling_reaches_the_end_not_just_a_prefix(self):
        """A prefix of a long protocol is its title page and table of contents, which
        supports almost no answerable question.

        Uses *varying* chunk lengths so the trim loop actually fires. With 765 identical
        chunks the tail is never popped and reaching the end is structural rather than
        tested — which is what the earlier fixture did.
        """
        doc = [
            {"chunk_id": f"c{i}", "doc_id": "d", "text": "x" * (400 + (i % 7) * 300)}
            for i in range(765)
        ]
        picked = golden.select_chunks_for_drafting(doc)
        last_index = int(picked[-1]["chunk_id"][1:])
        assert last_index > 600, f"only reached chunk {last_index}"
        assert sum(len(c["text"]) for c in picked) <= golden.DRAFT_CHAR_BUDGET

    def test_document_order_is_preserved(self):
        picked = golden.select_chunks_for_drafting(self._doc(300))
        ids = [int(c["chunk_id"][1:]) for c in picked]
        assert ids == sorted(ids)

    def test_selection_is_deterministic(self):
        doc = self._doc(400)
        assert golden.select_chunks_for_drafting(doc) == golden.select_chunks_for_drafting(doc)

    def test_empty_document(self):
        assert golden.select_chunks_for_drafting([]) == []

    def test_an_oversized_chunk_still_yields_one_excerpt(self):
        """The trim loop must terminate *and* leave something to draft from.

        This test previously asserted `== []` — it enshrined the emptying behaviour as
        intended, when in fact `build_draft_prompt` would then render zero excerpts and
        still ask for ten pairs: one request for guaranteed zero yield. Flagged by the
        Phase 6 code-review gate.
        """
        picked = golden.select_chunks_for_drafting(self._doc(3, chars=golden.DRAFT_CHAR_BUDGET * 2))
        assert len(picked) == 1
        prompt = golden.build_draft_prompt("d", picked, 10)
        assert picked[0]["text"][:50] in prompt

    def test_prompt_discloses_when_it_sampled(self):
        long_prompt = golden.build_draft_prompt("d", self._doc(765), 10)
        assert "sampled evenly across the document" in long_prompt
        short_prompt = golden.build_draft_prompt("d", self._doc(5), 10)
        assert "sampled evenly" not in short_prompt

    def test_budget_bounds_what_it_selects(self):
        """The constant's job, not its value.

        This previously asserted `DRAFT_CHAR_BUDGET == 40_000` under a docstring claiming
        to guard a measured claim about real documents. It tested neither.
        """
        for n in (1, 40, 300, 765):
            picked = golden.select_chunks_for_drafting(self._doc(n))
            total = sum(len(c["text"]) for c in picked)
            assert total <= golden.DRAFT_CHAR_BUDGET or len(picked) == 1, n


class TestBoilerplateRejection:
    """Pairs that verify perfectly and measure nothing.

    The standard FDA nonbinding-recommendations disclaimer appears in 39 documents of
    this corpus. A question about it is answerable from any of them, so retrieving the
    *right* document is not required and a model scores by recognising a template. Two
    pairs in the first batch were literally the same question phrased twice.
    """

    SHARED = "the word should means that something is suggested or recommended"
    UNIQUE = "the maximum permitted residue level is four parts per million by weight"

    def _index(self, n_shared: int, n_total: int = 40):
        """`n_shared` documents carry the shared passage; one also carries the unique one."""
        idx = {f"d{i}": f"filler text {i}. {self.SHARED}." for i in range(n_shared)}
        for i in range(n_shared, n_total):
            idx[f"d{i}"] = f"filler text {i}. unrelated content here."
        idx["d0"] = idx["d0"] + " " + self.UNIQUE
        return idx

    def _pair(self, quote):
        return golden.GoldenPair(
            pair_id="d0::gold::000",
            question="What does the guidance say about this, in a full sentence?",
            # Echoes the quote so the coverage check passes and the test isolates the
            # boilerplate behaviour it is actually about.
            answer=quote.capitalize() + " That is what the guidance states.",
            evidence=({"chunk_id": "c1", "quote": quote},),
            doc_id="d0",
            shape="definition",
        )

    def _chunk(self, text):
        return [{"chunk_id": "c1", "doc_id": "d0", "text": text}]

    def test_widely_repeated_evidence_is_rejected(self):
        idx = self._index(39)
        r = golden.validate_pair(self._pair(self.SHARED), self._chunk(idx["d0"]), doc_index=idx)
        assert r.status == golden.STATUS_REJECTED
        assert "boilerplate" in r.reject_reason
        assert "39 documents" in r.reject_reason

    def test_document_specific_evidence_is_accepted(self):
        idx = self._index(39)
        r = golden.validate_pair(self._pair(self.UNIQUE), self._chunk(idx["d0"]), doc_index=idx)
        assert r.accepted

    def test_two_documents_is_allowed_for_draft_final_pairs(self):
        """The corpus deliberately contains 8 complete draft/final guidance pairs, which
        legitimately share long passages. Rejecting at 2 would discard real
        version-currency questions."""
        idx = self._index(2)
        r = golden.validate_pair(self._pair(self.SHARED), self._chunk(idx["d0"]), doc_index=idx)
        assert r.accepted

    def test_three_documents_is_rejected(self):
        idx = self._index(3)
        assert not golden.validate_pair(
            self._pair(self.SHARED), self._chunk(idx["d0"]), doc_index=idx
        ).accepted

    def test_threshold_is_the_documented_measured_value(self):
        assert golden.MAX_EVIDENCE_DOCUMENT_SPREAD == 2

    def test_check_is_skipped_when_no_index_is_supplied(self):
        """Callers without a corpus index must still be able to validate."""
        assert golden.validate_pair(self._pair(self.SHARED), self._chunk(self.SHARED)).accepted

    def test_spread_ignores_whitespace_differences(self):
        """The index must be built with `build_document_index`, which normalises once.
        An earlier version of this test passed a raw dict and asserted the function
        normalised for it — it does not, by design."""
        idx = golden.build_document_index(
            [
                {
                    "doc_id": "a",
                    "text": "the  word\nshould means that something is suggested or recommended",
                }
            ]
        )
        assert golden.evidence_document_spread(self.SHARED, idx) == 1

    def test_spread_ignores_quotes_below_the_length_floor(self):
        """A three-character quote would match nearly every document."""
        assert golden.evidence_document_spread("the", {"a": "the", "b": "the"}) == 0

    def test_unverifiable_evidence_is_reported_as_such_not_as_boilerplate(self):
        """Ordering: a fabricated quote appears in zero documents, and must not be
        mislabelled as boilerplate by a spread of 0."""
        idx = self._index(39)
        r = golden.validate_pair(
            self._pair("this text appears in no document anywhere at all"),
            self._chunk(idx["d0"]),
            doc_index=idx,
        )
        assert "no evidence verified" in r.reject_reason

    def test_document_index_normalises_once(self):
        chunks = [
            {"chunk_id": "c1", "doc_id": "d1", "text": "alpha  beta"},
            {"chunk_id": "c2", "doc_id": "d1", "text": "gamma\n\ndelta"},
            {"chunk_id": "c3", "doc_id": "d2", "text": "epsilon"},
        ]
        idx = golden.build_document_index(chunks)
        assert set(idx) == {"d1", "d2"}
        assert "alpha beta" in idx["d1"] and "gamma delta" in idx["d1"]


class TestRedundantAnswerRejection:
    """Two pairs asking the same thing of different documents weight one template twice
    in every mean. The first curated batch had exactly one such pair: two "where do I
    mail written comments" questions whose answers were the same FDA docket address.

    This is the case `MAX_EVIDENCE_DOCUMENT_SPREAD` structurally cannot catch — slight
    wording drift makes each variant look document-unique, so the redundancy has to be
    measured between pairs rather than within the corpus.
    """

    def _p(self, pid, answer):
        return golden.GoldenPair(
            pair_id=pid,
            question="Where should written comments be submitted for this guidance?",
            answer=answer,
            evidence=({"chunk_id": "c1", "quote": "q" * 30},),
            doc_id=pid.split("::")[0],
            shape="procedure",
            status=golden.STATUS_ACCEPTED,
        )

    A = (
        "Written comments may be submitted to the Dockets Management Staff HFA-305 "
        "Food and Drug Administration 5630 Fishers Lane Room 1061 Rockville MD 20852"
    )
    B = (
        "Written comments regarding the document may be submitted to the Division of "
        "Dockets Management HFA-305 Food and Drug Administration 5630 Fishers Lane "
        "rm 1061 Rockville MD 20852"
    )
    C = (
        "The applicant must retain distribution records for a period of not less than "
        "three years after the date of shipment"
    )

    def test_near_identical_answers_are_detected(self):
        assert golden.answer_overlap(self.A, self.B) > golden.MAX_ANSWER_OVERLAP

    def test_unrelated_answers_are_not(self):
        assert golden.answer_overlap(self.A, self.C) < golden.MAX_ANSWER_OVERLAP

    def test_one_of_a_duplicate_group_survives(self):
        out = golden.drop_redundant(
            [self._p("d1::gold::000", self.A), self._p("d2::gold::000", self.B)]
        )
        assert sum(1 for p in out if p.accepted) == 1
        assert sum(1 for p in out if p.status == golden.STATUS_REJECTED) == 1

    def test_the_survivor_is_deterministic_by_pair_id(self):
        a, b = self._p("d1::gold::000", self.A), self._p("d2::gold::000", self.B)
        first = [p.pair_id for p in golden.drop_redundant([a, b]) if p.accepted]
        second = [p.pair_id for p in golden.drop_redundant([b, a]) if p.accepted]
        assert first == second == ["d1::gold::000"]

    def test_rejection_names_the_pair_it_duplicates(self):
        out = golden.drop_redundant(
            [self._p("d1::gold::000", self.A), self._p("d2::gold::000", self.B)]
        )
        rej = next(p for p in out if p.status == golden.STATUS_REJECTED)
        assert "d1::gold::000" in rej.reject_reason

    def test_distinct_answers_all_survive(self):
        pairs = [self._p("d1::gold::000", self.A), self._p("d2::gold::000", self.C)]
        assert sum(1 for p in golden.drop_redundant(pairs) if p.accepted) == 2

    def test_already_rejected_pairs_pass_through_untouched(self):
        rej = golden.GoldenPair(
            pair_id="d3::gold::000",
            question="q",
            answer=self.A,
            evidence=(),
            doc_id="d3",
            shape="procedure",
            status=golden.STATUS_REJECTED,
            reject_reason="empty answer",
        )
        out = golden.drop_redundant([self._p("d1::gold::000", self.A), rej])
        assert next(p for p in out if p.pair_id == "d3::gold::000").reject_reason == "empty answer"

    def test_empty_answers_do_not_match_each_other(self):
        assert golden.answer_overlap("", "") == 0.0

    def test_threshold_sits_in_the_measured_gap(self):
        """Duplicate scored 0.750, runner-up 0.304."""
        assert 0.304 < golden.MAX_ANSWER_OVERLAP < 0.750


class TestFactualRiskTriage:
    """Numbers asserted in an answer but absent from the cited source.

    The highest-value factual error to catch: regulatory answers are dense with
    quantities, a wrong one is materially wrong, and unlike prose entailment it is
    checkable without a model. Triage only — a flag means "a human should look".
    """

    def test_unsupported_number_is_flagged(self):
        assert golden.unsupported_numeric_claims(
            "The applicant must respond within 45 days.", "must respond within 30 days"
        ) == ["45"]

    def test_supported_number_is_not_flagged(self):
        assert (
            golden.unsupported_numeric_claims(
                "The limit is 21 CFR 507 as stated.", "governed by 21 CFR 507 and related parts"
            )
            == []
        )

    def test_written_numeral_in_source_supports_a_digit_in_the_answer(self):
        assert (
            golden.unsupported_numeric_claims("retained for 3 years", "kept for three years") == []
        )

    def test_each_year_supports_once_a_year(self):
        """The second false positive this check raised on real data."""
        assert (
            golden.unsupported_numeric_claims(
                "must test at least once a year", "required to monitor at least once each year"
            )
            == []
        )

    def test_annually_supports_one_year(self):
        assert golden.unsupported_numeric_claims("once a year", "reported annually") == []

    def test_a_year_supports_one_year(self):
        """The only flag raised on the first 83 pairs, and it was this false positive."""
        assert (
            golden.unsupported_numeric_claims(
                "retained for 1 year", "will be kept for a year, for follow up"
            )
            == []
        )

    def test_the_indefinite_article_rule_is_scoped_to_time_units(self):
        """'a supplier' must not be read as the number one."""
        assert golden.unsupported_numeric_claims(
            "exactly 1 supplier", "a supplier is defined as"
        ) == ["1"]

    def test_thousands_separators_are_not_a_discrepancy(self):
        assert (
            golden.unsupported_numeric_claims("at 5,630 Fishers Lane", "at 5630 Fishers Lane") == []
        )
        assert (
            golden.unsupported_numeric_claims("at 5630 Fishers Lane", "at 5,630 Fishers Lane") == []
        )

    def test_trailing_period_is_stripped(self):
        assert golden.unsupported_numeric_claims("the limit is 40.", "the limit is 40") == []

    def test_answers_with_no_numbers_are_never_flagged(self):
        assert (
            golden.unsupported_numeric_claims("The supplier must be identified.", "anything") == []
        )

    def test_flag_factual_risk_uses_the_whole_cited_chunk(self):
        """A drafter legitimately summarises more of a chunk than it quotes."""
        pair = golden.GoldenPair(
            pair_id="d::gold::000",
            question="q" * 50,
            answer="The period is 6 years and the retention is 1 year.",
            evidence=({"chunk_id": "c1", "quote": "stored for 6 years"},),
            doc_id="d",
            shape="quantity",
            status=golden.STATUS_ACCEPTED,
        )
        idx = {"c1": {"chunk_id": "c1", "text": "stored for 6 years. kept for a year."}}
        assert golden.flag_factual_risk([pair], idx) == []

    def test_rejected_pairs_are_not_triaged(self):
        pair = golden.GoldenPair(
            pair_id="d::gold::000",
            question="q",
            answer="999 days",
            evidence=({"chunk_id": "c1", "quote": "x"},),
            doc_id="d",
            shape="quantity",
            status=golden.STATUS_REJECTED,
            reject_reason="empty answer",
        )
        assert golden.flag_factual_risk([pair], {"c1": {"text": "nothing"}}) == []


class TestLineNumberDensity:
    """Documents typeset with line numbers are poor sources of verbatim quotes.

    Two such documents yielded 0/10 and 1/10 accepted pairs against 7/10-10/10 for
    everything else. They stay indexed; they are only skipped as drafting sources.
    """

    NUMBERED = "\n".join(
        f"the sponsor shall submit the required documentation {i}" for i in range(1, 40)
    )
    PLAIN = "\n".join("the sponsor shall submit the required documentation" for _ in range(40))

    def test_line_numbered_text_scores_high(self):
        assert golden.line_number_density(self.NUMBERED) > golden.MAX_LINE_NUMBER_DENSITY

    def test_plain_text_scores_low(self):
        assert golden.line_number_density(self.PLAIN) < golden.MAX_LINE_NUMBER_DENSITY

    def test_bare_number_lines_count(self):
        text = "\n".join(["some prose here", "17"] * 10)
        assert golden.line_number_density(text) >= 0.5

    def test_short_text_is_never_flagged(self):
        assert golden.line_number_density("a 1\nb 2\nc 3") == 0.0

    def test_threshold_lies_between_the_observed_outcomes(self):
        """0.04-0.18 drafted well, 0.57-0.65 did not, 0.18-0.57 is untested. The
        constant must sit inside the untested band, not inside either observation."""
        assert 0.18 < golden.MAX_LINE_NUMBER_DENSITY < 0.57


class TestAnswerEvidenceCoverage:
    """A free proxy for the defect tier 2 found: answers that synthesise across a chunk
    while citing one span of it.

    Validated against the 16 judged positives from the first calibration — judge-supported
    pairs scored median 0.888 (min 0.471), judge-rejected ones median 0.225 (max 0.611).
    The threshold sits below the optimum on purpose, so it keeps every pair the judge
    accepted.
    """

    QUOTE = (
        "The sponsor must submit an annual report to the Agency describing the status "
        "of each ongoing investigation"
    )

    def test_a_grounded_answer_scores_high(self):
        answer = "The sponsor must submit an annual report describing each ongoing investigation."
        assert golden.answer_evidence_coverage(answer, [self.QUOTE]) > 0.9

    def test_an_answer_that_adds_unsupported_content_scores_low(self):
        """The HL7 shape: the quote supports part of it and the answer adds a fact."""
        answer = (
            "The sponsor must submit an annual report, which is an HL7 structured "
            "document filed through the electronic gateway maintained by the district "
            "office and countersigned by the qualified investigator."
        )
        assert golden.answer_evidence_coverage(answer, [self.QUOTE]) < 0.45

    def test_multiple_quotes_can_cover_a_longer_answer(self):
        """Why the prompt asks for as many quotes as the answer needs."""
        answer = "The sponsor must submit an annual report, and retain records for three years."
        one = golden.answer_evidence_coverage(answer, [self.QUOTE])
        two = golden.answer_evidence_coverage(
            answer, [self.QUOTE, "shall retain the records for three years after shipment"]
        )
        assert two > one
        assert two > 0.9

    def test_an_answer_with_no_content_words_is_not_penalised(self):
        """It is caught by the length floor, not reported as ungrounded."""
        assert golden.answer_evidence_coverage("the and of", [self.QUOTE]) == 1.0

    def test_stopwords_do_not_inflate_coverage(self):
        """Regulatory filler appears in every document and would give free credit.

        The evidence deliberately *shares* the filler words with the answer. An earlier
        version paired the answer with "wholly unrelated text here", which shares no token
        at all — so coverage was 0.0 whether or not stopwords were filtered, and the
        assertion could not fail for its stated reason.
        """
        answer = "The applicant shall provide including such other information as required."
        evidence = ["provided that including such other as required by the provisions"]
        assert golden.answer_evidence_coverage(answer, evidence) < 0.5

    def test_validation_rejects_a_low_coverage_pair(self):
        pair = golden.GoldenPair(
            pair_id="d::gold::000",
            question="What must the sponsor submit and how is it filed?",
            answer=(
                "It is an HL7 structured document filed through the electronic gateway "
                "maintained by the district office and countersigned by the investigator."
            ),
            evidence=({"chunk_id": "c1", "quote": self.QUOTE},),
            doc_id="d",
            shape="requirement",
        )
        r = golden.validate_pair(pair, [{"chunk_id": "c1", "doc_id": "d", "text": self.QUOTE}])
        assert r.status == golden.STATUS_REJECTED
        assert "outruns its evidence" in r.reject_reason

    def test_threshold_keeps_every_judge_supported_pair(self):
        """Read from the persisted calibration rather than restating the constant.

        This previously asserted `MIN_ANSWER_EVIDENCE_COVERAGE < 0.471`, which is a
        restatement of the constant against a hardcoded number. The per-pair data it
        claimed to guard is on disk.
        """
        import json
        import pathlib as _p

        cal = _p.Path("reports/judge_calibration.json")
        if not cal.exists():
            import pytest as _pytest

            _pytest.skip("no calibration artifact on disk")
        items = [
            i
            for i in json.loads(cal.read_text())["items"]
            if i["control"] == "positive" and i["verdict"] == "supported"
        ]
        if not items:
            import pytest as _pytest

            _pytest.skip("no supported positives in the artifact")
        worst = min(golden.answer_evidence_coverage(i["claim"], [i["quote"]]) for i in items)
        assert worst >= golden.MIN_ANSWER_EVIDENCE_COVERAGE, (
            f"threshold {golden.MIN_ANSWER_EVIDENCE_COVERAGE} would reject a "
            f"judge-supported pair scoring {worst:.3f}"
        )

    def test_prompt_demands_coverage_not_just_one_quote(self):
        assert "cover the whole answer" in golden.draft_system_prompt()


class TestLeadingLineNumbers:
    """FDA draft guidances put the line number at the *start* of the line, and the
    detector could not see them.

    `fda-71536` is unambiguously line-numbered — "1 This guidance has been prepared by
    the Division of...", about 45% of its lines — and scored 0.255, comfortably under the
    0.35 gate, so it was admitted as a drafting source. The existing fixture put its
    numbers at line ends, which is why this went unexercised for the whole phase.
    """

    LEADING = "\n".join(
        f"{i} the sponsor shall submit the required documentation" for i in range(1, 40)
    )
    TRAILING = "\n".join(
        f"the sponsor shall submit the required documentation {i}" for i in range(1, 40)
    )
    PLAIN = "\n".join("the sponsor shall submit the required documentation" for _ in range(40))

    def test_leading_numbers_are_detected(self):
        assert golden.line_number_density(self.LEADING) > golden.MAX_LINE_NUMBER_DENSITY

    def test_trailing_numbers_are_still_detected(self):
        assert golden.line_number_density(self.TRAILING) > golden.MAX_LINE_NUMBER_DENSITY

    def test_plain_prose_is_still_clean(self):
        assert golden.line_number_density(self.PLAIN) < golden.MAX_LINE_NUMBER_DENSITY

    def test_the_real_document_is_now_caught(self):
        """Reads the corpus if present: `fda-71536` must clear the gate."""
        import json
        import pathlib as _p

        path = _p.Path("data/chunks/structural.jsonl")
        if not path.exists():
            import pytest as _pytest

            _pytest.skip("corpus not present")
        # Iterate the file: `splitlines()` also breaks on U+2028, of which this corpus
        # contains six, which tears six JSONL records mid-string. That is how this hazard
        # was found -- this very test raised JSONDecodeError.
        with path.open(encoding="utf-8") as fh:
            texts = [
                json.loads(line)["text"]
                for line in fh
                if line.strip() and json.loads(line)["doc_id"] == "fda-71536"
            ]
        if not texts:
            import pytest as _pytest

            _pytest.skip("fda-71536 not in this corpus")
        assert golden.line_number_density("\n".join(texts)) > golden.MAX_LINE_NUMBER_DENSITY

    def test_a_leading_number_needs_whitespace_after_it(self):
        """'21 CFR 314.50' at the start of a line is content, not a line number -- but so
        is a line number followed by a space. The pattern cannot separate them, which is
        why this is a *density* proxy over many lines and not a per-line verdict."""
        one_line = "21 CFR 314.50 applies here"
        assert golden.line_number_density(one_line) == 0.0


class TestJsonlIsReadByRecord:
    """`str.splitlines()` is not a JSONL record splitter.

    It splits on U+2028, U+2029, form feed, U+0085 and more, none of which terminate a
    JSONL record. `data/chunks/structural.jsonl` contains six literal U+2028 characters,
    so `read_text().splitlines()` reports 13,429 lines for 13,423 records and tears six of
    them mid-string. Found because a test in this file raised JSONDecodeError.
    """

    def test_load_golden_survives_a_unicode_line_separator(self, tmp_path):
        pair = golden.GoldenPair(
            pair_id="d::gold::000",
            question="What does the provision require of the sponsor here?",
            answer="It requires the sponsor to retain the records for three years.",
            evidence=({"chunk_id": "c1", "quote": "retain the records for three years"},),
            doc_id="d",
            shape="requirement",
            status=golden.STATUS_ACCEPTED,
        )
        path = golden.write_golden([pair], tmp_path / "g.jsonl")
        back = golden.load_golden(path)
        assert len(back) == 1
        assert " " in back[0].evidence[0]["quote"]

    def test_splitlines_would_have_broken_it(self):
        """Guard against a false pass: the separator must actually be a splitline."""
        assert len("a b".splitlines()) == 2
        assert len(["a\u2028b"]) == 1
