"""Tests for composing tier 1 and tier 2 into a faithfulness figure.

The point of the composition is that the two tiers answer different questions and
neither implies the other. Phase 6 found golden pairs whose quotes were real and did not
support the claim at roughly a quarter of the set, so a report that collapsed the two
would be reporting the wrong thing confidently.
"""

from __future__ import annotations

import pytest

from ragpipe import citations as cit
from ragpipe import gen_eval
from ragpipe.judge import PARTIAL, SUPPORTED, UNSUPPORTED, Judgment


def _outcome(qid, checks, answer="The sponsor must respond within 30 days.", refused=False):
    return gen_eval.QueryOutcome(
        query_id=qid,
        slice_name="section_lookup",
        query="q",
        refused=refused,
        refusal_source="model" if refused else None,
        n_claimed=len(checks),
        n_verified=sum(1 for c in checks if c["method"] in cit.VERIFIED_METHODS),
        buckets={},
        citation_precision=None,
        top_score=9.0,
        prompt_tokens=10,
        output_tokens=10,
        thinking_tokens=0,
        latency_s=1.0,
        attempts=1,
        checks=checks,
        answer_text=answer,
    )


def _check(method, quote="a quote long enough to be evidence"):
    return {"method": method, "chunk_id": "c1", "quote": quote, "matched_text": quote}


class TestJudgeItemSelection:
    def test_only_located_citations_are_judged(self):
        """Asking whether a fabricated quote supports a claim wastes a request on a
        question tier 1 already answered."""
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome("q1", [_check(cit.EXACT), _check(cit.UNVERIFIED), _check(cit.TOO_SHORT)])
            ]
        )
        items = gen_eval.judge_items_for(r, unit="citation")
        assert len(items) == 1
        assert items[0].item_id == "q1::0"

    def test_the_default_unit_is_the_answer(self):
        """Per-citation judging produced a number that measured the unit rather than the
        model: 18 of 22 verdicts came back `partial`, because asking whether *one* quote
        supports a *whole* answer is the wrong question when an answer cites several."""
        r = gen_eval.GenEvalResult(
            outcomes=[_outcome("q1", [_check(cit.EXACT), _check(cit.EXACT), _check(cit.EXACT)])]
        )
        items = gen_eval.judge_items_for(r)
        assert len(items) == 1, "one item per answer, not per citation"
        assert items[0].item_id == "q1"

    def test_answer_level_item_carries_every_located_quote_numbered(self):
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome(
                    "q1",
                    [_check(cit.EXACT, "first span here"), _check(cit.NORMALIZED, "second span")],
                )
            ]
        )
        item = gen_eval.judge_items_for(r)[0]
        assert "(1) first span here" in item.quote
        assert "(2) second span" in item.quote

    def test_answer_level_item_excludes_unlocated_quotes(self):
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome(
                    "q1", [_check(cit.EXACT, "real span"), _check(cit.UNVERIFIED, "fake span")]
                )
            ]
        )
        item = gen_eval.judge_items_for(r)[0]
        assert "real span" in item.quote and "fake span" not in item.quote

    def test_unknown_unit_is_rejected(self):
        with pytest.raises(ValueError, match="unit must be"):
            gen_eval.judge_items_for(gen_eval.GenEvalResult(), unit="sentence")

    def test_normalized_citations_are_judged(self):
        r = gen_eval.GenEvalResult(outcomes=[_outcome("q1", [_check(cit.NORMALIZED)])])
        assert len(gen_eval.judge_items_for(r)) == 1

    def test_line_number_ambiguous_is_not_judged(self):
        """It is not verified, so it never reaches tier 2."""
        r = gen_eval.GenEvalResult(outcomes=[_outcome("q1", [_check(cit.LINE_NUMBER_AMBIGUOUS)])])
        assert gen_eval.judge_items_for(r) == []

    def test_refusals_are_not_judged(self):
        r = gen_eval.GenEvalResult(outcomes=[_outcome("q1", [], refused=True)])
        assert gen_eval.judge_items_for(r) == []

    def test_errored_outcomes_are_not_judged(self):
        o = _outcome("q1", [_check(cit.EXACT)])
        o.error = "GenerationError"
        assert gen_eval.judge_items_for(gen_eval.GenEvalResult(outcomes=[o])) == []

    def test_item_carries_the_answer_and_the_located_span(self):
        r = gen_eval.GenEvalResult(
            outcomes=[_outcome("q1", [_check(cit.EXACT, "the located span text here")])]
        )
        item = gen_eval.judge_items_for(r, unit="citation")[0]
        assert item.claim == "The sponsor must respond within 30 days."
        assert item.quote == "the located span text here"
        assert item.source_id == "q1"

    def test_item_ids_are_unique_across_queries(self):
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome("q1", [_check(cit.EXACT), _check(cit.EXACT)]),
                _outcome("q2", [_check(cit.EXACT)]),
            ]
        )
        ids = [i.item_id for i in gen_eval.judge_items_for(r, unit="citation")]
        assert ids == ["q1::0", "q1::1", "q2::0"]


class TestFaithfulnessScoring:
    def test_the_three_denominators_are_kept_apart(self):
        """A located-but-unsupporting citation is a different failure from a fabricated
        one, and dividing by the wrong total makes them indistinguishable."""
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome("q1", [_check(cit.EXACT), _check(cit.EXACT), _check(cit.UNVERIFIED)])
            ]
        )
        j = {"q1::0": Judgment("q1::0", SUPPORTED), "q1::1": Judgment("q1::1", UNSUPPORTED)}
        f = gen_eval.score_faithfulness(r, j)
        assert f.n_claimed == 3
        assert f.n_located == 2
        assert f.n_supported == 1
        assert f.supported_of_located == 0.5
        assert f.supported_of_claimed == pytest.approx(1 / 3)

    def test_partial_is_reported_separately_not_folded_in(self):
        """Judging a whole answer against one of several quotes makes `partial` the
        expected verdict, so it must not silently count either way."""
        r = gen_eval.GenEvalResult(outcomes=[_outcome("q1", [_check(cit.EXACT)])])
        f = gen_eval.score_faithfulness(r, {"q1::0": Judgment("q1::0", PARTIAL)})
        assert f.n_partial == 1
        assert f.n_supported == 0
        assert f.n_unsupported == 0

    def test_answer_level_rate_requires_every_citation_to_support(self):
        """One bad citation in four is a bad answer."""
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome("good", [_check(cit.EXACT), _check(cit.EXACT)]),
                _outcome("bad", [_check(cit.EXACT), _check(cit.EXACT)]),
            ]
        )
        j = {
            "good::0": Judgment("good::0", SUPPORTED),
            "good::1": Judgment("good::1", SUPPORTED),
            "bad::0": Judgment("bad::0", SUPPORTED),
            "bad::1": Judgment("bad::1", PARTIAL),
        }
        f = gen_eval.score_faithfulness(r, j)
        assert f.answers_with_citations == 2
        assert f.answers_fully_supported == 1
        assert f.answer_level_rate == 0.5

    def test_unjudged_citations_are_counted_not_assumed(self):
        r = gen_eval.GenEvalResult(outcomes=[_outcome("q1", [_check(cit.EXACT)])])
        f = gen_eval.score_faithfulness(r, {})
        assert f.n_unjudged == 1
        assert f.n_supported == 0
        assert f.answers_fully_supported == 0, "an unjudged citation must not count as supported"

    def test_an_answer_with_no_located_citations_is_excluded_from_the_answer_rate(self):
        """Its faithfulness is undefined, not zero -- the same reasoning that keeps
        refusals out of citation precision."""
        r = gen_eval.GenEvalResult(outcomes=[_outcome("q1", [_check(cit.UNVERIFIED)])])
        f = gen_eval.score_faithfulness(r, {})
        assert f.answers_with_citations == 0
        assert f.answer_level_rate is None
        assert f.n_claimed == 1 and f.n_located == 0

    def test_refusals_contribute_nothing(self):
        """A refused answer that *still cites* must not be scored.

        `answer.py` deliberately preserves citations on a refusal ("a model that refuses
        *and* cites is inconsistent, and silently dropping the citations would hide
        that"), so this state is reachable. The earlier fixture passed `checks=[]`, which
        made the assertion trivially true and let the citation-unit bug through: those
        never-judged quotes were counted as tier-2 failures, driving
        `supported_of_located` from 1.000 to 0.333.
        """
        clean = _outcome("ok", [_check(cit.EXACT)])
        refused_but_citing = _outcome("r", [_check(cit.EXACT), _check(cit.EXACT)], refused=True)
        r = gen_eval.GenEvalResult(outcomes=[clean, refused_but_citing])
        j = {"ok::0": Judgment("ok::0", SUPPORTED)}
        f = gen_eval.score_faithfulness(r, j)
        assert f.supported_of_located == 1.0, "refused citations must not dilute the rate"
        assert f.n_unjudged == 0
        f2 = gen_eval.score_answer_faithfulness(r, {"ok": Judgment("ok", SUPPORTED)})
        assert f2.answer_level_rate == 1.0

    def test_empty_run_yields_no_rates_rather_than_zeros(self):
        f = gen_eval.score_faithfulness(gen_eval.GenEvalResult(), {})
        d = f.as_dict()
        assert d["supported_of_located"] is None
        assert d["supported_of_claimed"] is None
        assert d["answer_level_rate"] is None

    def test_a_perfect_run_scores_one_everywhere(self):
        r = gen_eval.GenEvalResult(outcomes=[_outcome("q1", [_check(cit.EXACT)])])
        f = gen_eval.score_faithfulness(r, {"q1::0": Judgment("q1::0", SUPPORTED)})
        assert f.supported_of_located == 1.0
        assert f.supported_of_claimed == 1.0
        assert f.answer_level_rate == 1.0


class TestAnswerLevelScoring:
    """The unit the first real run showed to be correct."""

    def test_one_verdict_per_answer(self):
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome("q1", [_check(cit.EXACT), _check(cit.EXACT)]),
                _outcome("q2", [_check(cit.EXACT)]),
            ]
        )
        j = {"q1": Judgment("q1", SUPPORTED), "q2": Judgment("q2", PARTIAL)}
        f = gen_eval.score_answer_faithfulness(r, j)
        assert f.answers_with_citations == 2
        assert f.n_supported == 1 and f.n_partial == 1
        assert f.answer_level_rate == 0.5

    def test_citation_counts_are_still_reported(self):
        """Tier-1 totals stay visible even though the tier-2 unit changed."""
        r = gen_eval.GenEvalResult(
            outcomes=[_outcome("q1", [_check(cit.EXACT), _check(cit.UNVERIFIED)])]
        )
        f = gen_eval.score_answer_faithfulness(r, {"q1": Judgment("q1", SUPPORTED)})
        assert f.n_claimed == 2 and f.n_located == 1

    def test_a_multi_citation_answer_can_now_be_fully_supported(self):
        """Under per-citation judging this was structurally almost impossible: each of
        six quotes judged against the whole answer returns `partial`."""
        r = gen_eval.GenEvalResult(outcomes=[_outcome("q1", [_check(cit.EXACT)] * 6)])
        f = gen_eval.score_answer_faithfulness(r, {"q1": Judgment("q1", SUPPORTED)})
        assert f.answer_level_rate == 1.0
        assert f.n_located == 6

    def test_refusals_and_uncited_answers_are_excluded(self):
        r = gen_eval.GenEvalResult(
            outcomes=[
                _outcome("r", [], refused=True),
                _outcome("u", [_check(cit.UNVERIFIED)]),
            ]
        )
        f = gen_eval.score_answer_faithfulness(r, {})
        assert f.answers_with_citations == 0
        assert f.answer_level_rate is None

    def test_unjudged_answers_are_counted_not_assumed(self):
        r = gen_eval.GenEvalResult(outcomes=[_outcome("q1", [_check(cit.EXACT)])])
        f = gen_eval.score_answer_faithfulness(r, {})
        assert f.n_unjudged == 1 and f.answers_fully_supported == 0


class TestUnitsAreNeverMixed:
    """The defect the first two-tier report shipped with.

    Under answer-level judging `n_supported` counts answers and `n_located` counts
    citations. The report divided one by the other and printed 7/22 = 0.318 as "the
    share of citations that survive reading", which is neither a share of citations nor
    a share of answers.
    """

    def test_answer_unit_suppresses_citation_ratios(self):
        r = gen_eval.GenEvalResult(outcomes=[_outcome("q1", [_check(cit.EXACT)] * 6)])
        f = gen_eval.score_answer_faithfulness(r, {"q1": Judgment("q1", SUPPORTED)})
        assert f.unit == "answer"
        assert f.n_supported == 1 and f.n_located == 6
        assert f.supported_of_located is None, "7/22 was printed as a citation share"
        assert f.supported_of_claimed is None
        assert f.answer_level_rate == 1.0

    def test_citation_unit_still_reports_them(self):
        r = gen_eval.GenEvalResult(
            outcomes=[_outcome("q1", [_check(cit.EXACT), _check(cit.EXACT)])]
        )
        j = {"q1::0": Judgment("q1::0", SUPPORTED), "q1::1": Judgment("q1::1", UNSUPPORTED)}
        f = gen_eval.score_faithfulness(r, j)
        assert f.unit == "citation"
        assert f.supported_of_located == 0.5

    def test_payload_carries_the_unit(self):
        r = gen_eval.GenEvalResult(outcomes=[_outcome("q1", [_check(cit.EXACT)])])
        assert gen_eval.score_answer_faithfulness(r, {}).as_dict()["unit"] == "answer"
        assert gen_eval.score_faithfulness(r, {}).as_dict()["unit"] == "citation"

    def test_report_omits_the_invalid_ratio_under_answer_unit(self):
        r = gen_eval.GenEvalResult(generator="g", retriever="bm25", k=5)
        r.outcomes = [_outcome("q1", [_check(cit.EXACT)] * 6)]
        payload = r.to_payload()
        payload["faithfulness"] = gen_eval.score_answer_faithfulness(
            r, {"q1": Judgment("q1", SUPPORTED)}
        ).as_dict()
        payload["judge"] = "j"
        md = gen_eval.render_gen_eval(payload)
        assert "answers fully supported" in md
        assert "Supported of located" not in md
        assert "deliberately absent" in md

    def test_report_includes_the_ratio_under_citation_unit(self):
        r = gen_eval.GenEvalResult(generator="g", retriever="bm25", k=5)
        r.outcomes = [_outcome("q1", [_check(cit.EXACT), _check(cit.EXACT)])]
        payload = r.to_payload()
        payload["faithfulness"] = gen_eval.score_faithfulness(
            r, {"q1::0": Judgment("q1::0", SUPPORTED), "q1::1": Judgment("q1::1", PARTIAL)}
        ).as_dict()
        payload["judge"] = "j"
        md = gen_eval.render_gen_eval(payload)
        assert "Supported of located" in md
        assert "measured the unit rather than the model" in md


class TestFailedCalibrationRemovesFaithfulness:
    """The severest Phase 6 code-review finding.

    `_run_tier2` only *wrote* `payload["faithfulness"]` when the judge passed calibration;
    it never removed it. On the `--rejudge` path the payload is the previous report loaded
    from disk and already carries a faithfulness block — so a rubber-stamp judge that
    failed its negatives shipped the *stale* figure, beside its own failed calibration
    paragraph, with exit code 0. That is the exact opposite of what the help text and the
    progress log promise, on the one path presented as a cost saving.
    """

    def test_a_stale_figure_is_deleted_not_kept(self):
        payload = {"faithfulness": {"answer_level_rate": 0.875, "unit": "answer"}}
        # what _run_tier2 does when calibration fails
        payload.pop("faithfulness", None)
        assert "faithfulness" not in payload

    def test_the_renderer_emits_no_faithfulness_section_without_the_key(self):
        r = gen_eval.GenEvalResult(generator="g", retriever="bm25", k=5)
        r.outcomes = [_outcome("q1", [_check(cit.EXACT)])]
        payload = r.to_payload()
        payload["judge"] = "j"
        payload["judge_calibration"] = {"negative_rate": 0.0, "positive_rate": 1.0}
        md = gen_eval.render_gen_eval(payload)
        assert "## Faithfulness" not in md
        assert "answers fully supported" not in md

    def test_the_section_appears_once_the_key_is_present(self):
        """Guard against a false pass: the fixture must be able to render it."""
        r = gen_eval.GenEvalResult(generator="g", retriever="bm25", k=5)
        r.outcomes = [_outcome("q1", [_check(cit.EXACT)])]
        payload = r.to_payload()
        payload["judge"] = "j"
        payload["faithfulness"] = gen_eval.score_answer_faithfulness(
            r, {"q1": Judgment("q1", SUPPORTED)}
        ).as_dict()
        assert "## Faithfulness" in gen_eval.render_gen_eval(payload)
