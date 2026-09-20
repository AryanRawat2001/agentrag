"""Tests for the Phase 5 generation eval aggregation and report.

Weighted toward the guards, because the first real run of this module produced an
artifact that was arithmetically perfect and completely misleading: 77 of 80 queries
died to a rate-limit retry storm, and the report headlined "mean citation precision
1.0" computed from the one surviving answer. The defect is never the arithmetic.
"""

from __future__ import annotations

from ragpipe import citations as cit
from ragpipe import gen_eval


def _outcome(
    query_id="q1",
    slice_name="section_lookup",
    refused=False,
    n_claimed=1,
    n_verified=1,
    buckets=None,
    precision=1.0,
    top_score=10.0,
    error=None,
):
    return gen_eval.QueryOutcome(
        query_id=query_id,
        slice_name=slice_name,
        query="q",
        refused=refused,
        refusal_source="model" if refused else None,
        n_claimed=n_claimed,
        n_verified=n_verified,
        buckets=buckets if buckets is not None else {cit.EXACT: 1},
        citation_precision=precision,
        top_score=top_score,
        prompt_tokens=100,
        output_tokens=50,
        thinking_tokens=0,
        latency_s=2.0,
        attempts=1,
        error=error,
    )


class TestScoredPopulation:
    def test_errors_are_counted_but_never_averaged_in(self):
        r = gen_eval.GenEvalResult(
            outcomes=[_outcome(), _outcome(query_id="q2", error="GenerationError: 429")]
        )
        assert len(r.scored()) == 1
        assert r.to_payload()["n_errors"] == 1
        assert r.to_payload()["n_queries"] == 2

    def test_precision_excludes_answers_that_claimed_nothing(self):
        """A refusal has undefined precision. Scoring it 0.0 would make correct
        refusal look like unfaithfulness."""
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome(precision=1.0),
                _outcome(query_id="q2", refused=True, n_claimed=0, n_verified=0, precision=None),
            ]
        )
        assert r.mean_precision() == 1.0


class TestRefusalIsNotBlended:
    def _result(self):
        return gen_eval.GenEvalResult(
            outcomes=[
                # mention-labelled slice: refusing is often correct here
                _outcome(query_id="i1", slice_name="exact_identifier", refused=True),
                _outcome(query_id="i2", slice_name="exact_identifier", refused=True),
                _outcome(query_id="s1", slice_name="section_lookup", refused=False),
                _outcome(query_id="s2", slice_name="section_lookup", refused=False),
                _outcome(query_id="u1", slice_name="unanswerable", refused=True, top_score=2.0),
            ]
        )

    def test_unanswerable_refusal_is_its_own_figure(self):
        assert self._result().refusal_rate("unanswerable") == 1.0

    def test_answerable_rate_differs_with_and_without_mention_labelled(self):
        """The two readings differ and only one means what the name suggests."""
        r = self._result()
        assert r.answerable_refusal_rate(exclude_mention_labelled=False) == 0.5
        assert r.answerable_refusal_rate(exclude_mention_labelled=True) == 0.0

    def test_exact_identifier_is_flagged_as_mention_labelled(self):
        assert "exact_identifier" in gen_eval.MENTION_LABELLED_SLICES
        slices = self._result().to_payload()["slices"]
        assert slices["exact_identifier"]["mention_labelled"] is True
        assert slices["section_lookup"]["mention_labelled"] is False


class TestSeparability:
    def test_perfect_separation_scores_one(self):
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome(query_id="a", top_score=10.0),
                _outcome(query_id="u", slice_name="unanswerable", top_score=5.0),
            ]
        )
        auc, pairs = r.separability_auc()
        assert (auc, pairs) == (1.0, 1)

    def test_ties_count_as_half(self):
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome(query_id="a", top_score=7.0),
                _outcome(query_id="u", slice_name="unanswerable", top_score=7.0),
            ]
        )
        assert r.separability_auc()[0] == 0.5

    def test_inverted_scores_score_zero(self):
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome(query_id="a", top_score=1.0),
                _outcome(query_id="u", slice_name="unanswerable", top_score=9.0),
            ]
        )
        assert r.separability_auc()[0] == 0.0

    def test_none_without_both_populations(self):
        r = gen_eval.GenEvalResult(outcomes=[_outcome()])
        assert r.separability_auc() == (None, 0)

    def test_pair_count_is_reported_so_small_samples_are_visible(self):
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome(query_id="a1", top_score=9.0),
                _outcome(query_id="a2", top_score=8.0),
                _outcome(query_id="u1", slice_name="unanswerable", top_score=2.0),
            ]
        )
        assert r.separability_auc()[1] == 2


class TestThresholdSweep:
    def test_sweep_reports_both_error_kinds(self):
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome(query_id="a", top_score=10.0),
                _outcome(query_id="u", slice_name="unanswerable", top_score=2.0),
            ]
        )
        rows = r.threshold_sweep(n_steps=4)
        assert rows
        assert rows[0]["unanswerable_gated"] == 0  # threshold at the minimum gates nothing
        last = rows[-1]
        assert last["unanswerable_gated"] == 1  # at the max, the low scorer is gated
        assert last["answerable_total"] == 1

    def test_empty_when_scores_do_not_vary(self):
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome(query_id="a", top_score=5.0),
                _outcome(query_id="u", slice_name="unanswerable", top_score=5.0),
            ]
        )
        assert r.threshold_sweep() == []


class TestStratifiedSample:
    def test_takes_n_per_slice(self):
        rows = [{"slice_name": "a", "query_id": f"a{i}"} for i in range(5)]
        rows += [{"slice_name": "b", "query_id": f"b{i}"} for i in range(5)]
        picked = gen_eval.stratified_sample(rows, 2)
        assert [p["query_id"] for p in picked] == ["a0", "a1", "b0", "b1"]

    def test_slices_smaller_than_n_are_taken_whole(self):
        rows = [{"slice_name": "a", "query_id": "a0"}]
        assert len(gen_eval.stratified_sample(rows, 10)) == 1


class TestReportGuards:
    def _payload(self, n_queries, n_scored, n_errors):
        r = gen_eval.GenEvalResult(
            outcomes=[_outcome(query_id=f"q{i}") for i in range(n_scored)]
            + [_outcome(query_id=f"e{i}", error="GenerationError: 429") for i in range(n_errors)]
        )
        payload = r.to_payload()
        payload["n_queries"] = n_queries
        return payload

    def test_mostly_failed_run_is_flagged_as_not_evidence(self):
        """The exact artifact the first real run produced: 3 of 80 scored, and a
        headline precision of 1.0 from one answer."""
        md = gen_eval.render_gen_eval(self._payload(80, 3, 77))
        assert "not usable as evidence" in md
        assert "3 of 80" in md

    def test_healthy_run_has_no_alarm_banner(self):
        md = gen_eval.render_gen_eval(self._payload(10, 10, 0))
        assert "not usable as evidence" not in md
        assert "errored and are excluded" not in md

    def test_a_few_errors_are_disclosed_without_the_alarm(self):
        md = gen_eval.render_gen_eval(self._payload(10, 9, 1))
        assert "not usable as evidence" not in md
        assert "errored and are excluded" in md

    def test_threshold_is_a_named_constant(self):
        assert 0.0 < gen_eval.MIN_SCORED_FRACTION <= 1.0

    def test_report_names_the_mention_labelled_caveat(self):
        md = gen_eval.render_gen_eval(self._payload(10, 10, 0))
        assert "mention" in md.lower()

    def test_report_states_the_score_gate_is_off_by_default(self):
        md = gen_eval.render_gen_eval(self._payload(10, 10, 0))
        assert "off by default" in md

    def test_report_declines_to_invent_a_cost_figure(self):
        md = gen_eval.render_gen_eval(self._payload(10, 10, 0))
        assert "No dollar figure" in md


class TestCitationDetailIsAuditable:
    """The report JSON is the only record an independent audit can re-verify from.
    At a 160-char cap, 6 of 21 citations in the first shipped run were truncated and
    could only be prefix-checked."""

    def test_cap_exceeds_a_chunk(self):
        from ragpipe import chunking

        assert gen_eval.MAX_STORED_QUOTE_CHARS > chunking.DEFAULT_TARGET_CHARS

    def test_cap_is_bounded(self):
        """A pathological response must not bloat the artifact without limit."""
        assert gen_eval.MAX_STORED_QUOTE_CHARS <= 10_000
