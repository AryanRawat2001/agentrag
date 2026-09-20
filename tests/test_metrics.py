"""Direct tests for `ragpipe.metrics`.

This module had **29% line coverage and `separability` had none at all**, which is the
worst place in the project for a coverage hole: every retrieval figure in the README --
the 75x identifier finding, the chunk-size decision, the 0.992 refusal separability that
Phase 5's refusal path was built on -- is an average over these six functions. They were
exercised only transitively, through harness tests that would have been equally happy with
a subtly wrong denominator.

Each function's docstring makes a specific claim about *why* it is written the way it is.
These tests check the claims, not just the lines: a test that only re-derives the
implementation's arithmetic would pass against the bug the docstring says was avoided.
"""

from __future__ import annotations

import math

import pytest

from ragpipe import metrics


def _q(query_id="q", slice_name="s", ranked=(), relevant=(), top_score=0.0):
    return metrics.QueryResult(
        query_id=query_id,
        slice_name=slice_name,
        ranked_ids=list(ranked),
        relevant_ids=frozenset(relevant),
        top_score=top_score,
    )


class TestQueryResult:
    def test_a_query_with_relevant_ids_is_answerable(self):
        assert _q(relevant=["a"]).is_answerable

    def test_an_empty_relevant_set_marks_a_query_unanswerable(self):
        """This is the flag the unanswerable slice rides on, and the reason `aggregate`
        can exclude those queries rather than averaging structural zeros."""
        assert not _q(relevant=[]).is_answerable


class TestHitAtK:
    def test_a_relevant_id_inside_k_hits(self):
        assert metrics.hit_at_k(["a", "b"], frozenset({"b"}), 2) == 1.0

    def test_a_relevant_id_beyond_k_misses(self):
        assert metrics.hit_at_k(["a", "b"], frozenset({"b"}), 1) == 0.0

    def test_k_larger_than_the_result_list_is_not_an_error(self):
        assert metrics.hit_at_k(["a"], frozenset({"a"}), 100) == 1.0

    def test_k_of_zero_cannot_hit(self):
        assert metrics.hit_at_k(["a"], frozenset({"a"}), 0) == 0.0

    def test_an_empty_ranking_misses(self):
        assert metrics.hit_at_k([], frozenset({"a"}), 10) == 0.0


class TestRecallAtK:
    def test_half_the_relevant_set_scores_one_half(self):
        assert metrics.recall_at_k(["a", "x"], frozenset({"a", "b"}), 10) == 0.5

    def test_an_empty_relevant_set_scores_zero_rather_than_dividing_by_zero(self):
        assert metrics.recall_at_k(["a"], frozenset(), 10) == 0.0

    def test_a_repeated_chunk_cannot_push_recall_above_one(self):
        """The documented reason for counting distinct ids. Counting *positions* would
        score 2.0 here, and "an impossible recall value is the kind of thing that would be
        read as a win"."""
        assert metrics.recall_at_k(["a", "a"], frozenset({"a"}), 10) == 1.0

    def test_duplicates_do_not_inflate_a_partial_recall_either(self):
        assert metrics.recall_at_k(["a", "a", "a"], frozenset({"a", "b"}), 10) == 0.5

    def test_recall_never_exceeds_one_for_any_duplication(self):
        for n in range(1, 8):
            assert metrics.recall_at_k(["a"] * n, frozenset({"a"}), 10) <= 1.0


class TestPrecisionAtK:
    def test_the_denominator_is_k_not_the_number_of_results_returned(self):
        """The documented choice: "a retriever returning 3 results for k=10 has not earned
        the precision of one returning 10." Dividing by `len(ranked[:k])` would score 1.0
        here and make a retriever look perfect for being nearly empty."""
        assert metrics.precision_at_k(["a"], frozenset({"a"}), 10) == pytest.approx(0.1)

    def test_a_full_relevant_page_scores_one(self):
        assert metrics.precision_at_k(["a", "b"], frozenset({"a", "b"}), 2) == 1.0

    def test_k_of_zero_returns_zero_rather_than_dividing_by_zero(self):
        assert metrics.precision_at_k(["a"], frozenset({"a"}), 0) == 0.0

    def test_negative_k_returns_zero(self):
        """Guarding `k <= 0` rather than `k == 0` matters: `ranked[:-1]` is a *valid*
        slice, so a negative k would otherwise score a silently truncated ranking."""
        assert metrics.precision_at_k(["a", "b"], frozenset({"a", "b"}), -1) == 0.0


class TestReciprocalRank:
    def test_the_first_position_scores_one(self):
        assert metrics.reciprocal_rank(["a"], frozenset({"a"})) == 1.0

    def test_the_third_position_scores_one_third(self):
        assert metrics.reciprocal_rank(["x", "y", "a"], frozenset({"a"})) == pytest.approx(1 / 3)

    def test_only_the_first_relevant_hit_counts(self):
        assert metrics.reciprocal_rank(["x", "a", "b"], frozenset({"a", "b"})) == 0.5

    def test_omitting_k_searches_the_whole_ranking(self):
        assert metrics.reciprocal_rank(["x"] * 19 + ["a"], frozenset({"a"})) == pytest.approx(0.05)

    def test_k_truncates_the_search(self):
        assert metrics.reciprocal_rank(["x"] * 19 + ["a"], frozenset({"a"}), 10) == 0.0

    def test_no_relevant_hit_scores_zero(self):
        assert metrics.reciprocal_rank(["x"], frozenset({"a"})) == 0.0


class TestNdcgAtK:
    def test_a_single_relevant_item_at_rank_two_uses_the_log_discount(self):
        assert metrics.ndcg_at_k(["x", "a"], frozenset({"a"}), 10) == pytest.approx(
            1.0 / math.log2(3)
        )

    def test_a_perfect_ranking_scores_one(self):
        assert metrics.ndcg_at_k(["a", "b"], frozenset({"a", "b"}), 10) == pytest.approx(1.0)

    def test_more_relevant_chunks_than_k_can_still_score_one(self):
        """The documented cap. Without `min(len(relevant), k)` the ideal DCG would include
        rank-2 credit that k makes unreachable, scoring this perfect top-1 ranking at
        0.613 -- "nDCG would penalise a perfect ranking for the size of its own ground
        truth"."""
        assert metrics.ndcg_at_k(["a", "b"], frozenset({"a", "b"}), 1) == pytest.approx(1.0)

    def test_the_uncapped_ideal_would_have_scored_it_lower(self):
        uncapped = (1.0 / math.log2(2)) / (1.0 / math.log2(2) + 1.0 / math.log2(3))
        assert uncapped == pytest.approx(0.6131, abs=1e-4)
        assert metrics.ndcg_at_k(["a", "b"], frozenset({"a", "b"}), 1) > uncapped

    def test_an_empty_relevant_set_scores_zero(self):
        assert metrics.ndcg_at_k(["a"], frozenset(), 10) == 0.0

    def test_ordering_matters(self):
        early = metrics.ndcg_at_k(["a", "x", "y"], frozenset({"a"}), 10)
        late = metrics.ndcg_at_k(["x", "y", "a"], frozenset({"a"}), 10)
        assert early > late


class TestScoreQuery:
    def test_every_advertised_metric_is_present(self):
        scores = metrics.score_query(_q(ranked=["a"], relevant=["a"]))
        for k in metrics.K_VALUES:
            assert {f"hit@{k}", f"recall@{k}", f"precision@{k}"} <= set(scores)
        assert "mrr@10" in scores and "ndcg@10" in scores

    def test_mrr_is_capped_at_ten_not_at_the_full_ranking(self):
        """`mrr@10` is named for a cutoff, so it has to apply one. A relevant hit at rank
        11 must not contribute."""
        scores = metrics.score_query(_q(ranked=["x"] * 10 + ["a"], relevant=["a"]))
        assert scores["mrr@10"] == 0.0


class TestAggregate:
    def test_unanswerable_queries_are_excluded_from_the_mean(self):
        """The documented reason: averaging in a structural zero "would silently drag the
        whole table down in proportion to how many unanswerable queries the set happens to
        contain" -- which would make the table depend on slice composition."""
        results = [_q(ranked=["a"], relevant=["a"]), _q(ranked=["x"], relevant=[])]
        out = metrics.aggregate(results)
        assert out["n_queries"] == 1.0
        assert out["hit@1"] == 1.0

    def test_an_all_unanswerable_set_reports_no_queries_rather_than_nan(self):
        out = metrics.aggregate([_q(relevant=[])])
        assert out == {"n_queries": 0}

    def test_an_empty_input_reports_no_queries(self):
        assert metrics.aggregate([]) == {"n_queries": 0}

    def test_the_mean_is_over_answerable_queries_only(self):
        results = [
            _q(query_id="1", ranked=["a"], relevant=["a"]),
            _q(query_id="2", ranked=["x"], relevant=["a"]),
            _q(query_id="3", ranked=["x"], relevant=[]),
        ]
        assert metrics.aggregate(results)["hit@1"] == 0.5


class TestSeparability:
    """The one function here with no coverage at all, behind the 0.992 README figure and
    the decision that Phase 5 could refuse on a threshold."""

    def test_perfectly_separated_scores_reach_full_accuracy(self):
        results = [
            _q(query_id="a1", relevant=["a"], top_score=10.0),
            _q(query_id="a2", relevant=["a"], top_score=20.0),
            _q(query_id="u1", relevant=[], top_score=1.0),
            _q(query_id="u2", relevant=[], top_score=2.0),
        ]
        out = metrics.separability(results)
        assert out["best_accuracy"] == 1.0
        assert out["best_threshold"] == 10.0
        assert out["majority_baseline"] == 0.5
        assert out["gain_over_baseline"] == 0.5

    def test_the_predict_everything_unanswerable_corner_is_reachable(self):
        """The documented reason for appending a threshold above every observed score.
        Here every answerable query scores *below* every unanswerable one, so the best
        available classifier is the degenerate one. Without that extra candidate the
        search tops out at 0.5 -- **below** the 0.75 majority baseline -- and the function
        would report a worse-than-baseline optimum as the best achievable."""
        results = [
            _q(query_id="a1", relevant=["a"], top_score=1.0),
            _q(query_id="u1", relevant=[], top_score=5.0),
            _q(query_id="u2", relevant=[], top_score=6.0),
            _q(query_id="u3", relevant=[], top_score=7.0),
        ]
        out = metrics.separability(results)
        assert out["best_accuracy"] == 0.75
        assert out["best_threshold"] == 8.0
        assert out["majority_baseline"] == 0.75
        assert out["gain_over_baseline"] == 0.0

    def test_the_baseline_is_reported_so_the_headline_is_readable(self):
        """ "0.879 against a 0.750 baseline is a far weaker result than it appears" -- the
        accuracy is meaningless without the class balance beside it."""
        results = [_q(query_id=f"a{i}", relevant=["a"], top_score=9.0) for i in range(3)]
        results.append(_q(query_id="u", relevant=[], top_score=1.0))
        out = metrics.separability(results)
        assert out["majority_baseline"] == 0.75
        assert out["best_accuracy"] == 1.0
        assert out["gain_over_baseline"] == pytest.approx(0.25)

    def test_the_mean_top_scores_are_reported_for_both_classes(self):
        results = [
            _q(query_id="a1", relevant=["a"], top_score=10.0),
            _q(query_id="a2", relevant=["a"], top_score=20.0),
            _q(query_id="u1", relevant=[], top_score=2.0),
            _q(query_id="u2", relevant=[], top_score=4.0),
        ]
        out = metrics.separability(results)
        assert out["answerable_mean_top1"] == 15.0
        assert out["unanswerable_mean_top1"] == 3.0

    def test_no_unanswerable_queries_returns_counts_only(self):
        """A slice with nothing to separate from must not report an accuracy. Returning a
        number here would put a separability figure in the table for a run that never
        measured one."""
        out = metrics.separability([_q(relevant=["a"], top_score=1.0)])
        assert out == {"n_answerable": 1, "n_unanswerable": 0}
        assert "best_accuracy" not in out

    def test_no_answerable_queries_returns_counts_only(self):
        out = metrics.separability([_q(relevant=[], top_score=1.0)])
        assert out == {"n_answerable": 0, "n_unanswerable": 1}
        assert "best_accuracy" not in out

    def test_an_empty_input_returns_counts_only(self):
        assert metrics.separability([]) == {"n_answerable": 0, "n_unanswerable": 0}

    def test_identical_scores_across_classes_cannot_beat_the_baseline(self):
        """Overlapping distributions are the case the refusal threshold has to survive. If
        every score is the same, no threshold carries information and the best available
        accuracy is the majority baseline."""
        results = [
            _q(query_id="a1", relevant=["a"], top_score=5.0),
            _q(query_id="u1", relevant=[], top_score=5.0),
        ]
        out = metrics.separability(results)
        assert out["best_accuracy"] == pytest.approx(out["majority_baseline"])
        assert out["gain_over_baseline"] == pytest.approx(0.0)

    def test_the_threshold_is_one_a_caller_could_actually_apply(self):
        """The reported threshold has to reproduce the reported accuracy when applied as
        `predict answerable if top_score >= threshold`, or Phase 5 cannot use it."""
        results = [
            _q(query_id="a1", relevant=["a"], top_score=3.0),
            _q(query_id="a2", relevant=["a"], top_score=8.0),
            _q(query_id="u1", relevant=[], top_score=1.0),
            _q(query_id="u2", relevant=[], top_score=4.0),
        ]
        out = metrics.separability(results)
        t = out["best_threshold"]
        correct = sum(1 for r in results if (r.top_score >= t) == r.is_answerable)
        assert correct / len(results) == pytest.approx(out["best_accuracy"])

    def test_counts_are_reported_alongside_the_accuracy(self):
        results = [
            _q(query_id="a1", relevant=["a"], top_score=9.0),
            _q(query_id="u1", relevant=[], top_score=1.0),
            _q(query_id="u2", relevant=[], top_score=2.0),
        ]
        out = metrics.separability(results)
        assert out["n_answerable"] == 1.0
        assert out["n_unanswerable"] == 2.0
