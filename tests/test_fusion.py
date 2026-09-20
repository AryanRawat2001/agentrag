"""Fusion tests.

The three methods are easy to conflate and easy to get subtly wrong in ways that
still produce a plausible ranking, so the properties that distinguish them are
asserted directly: RRF must ignore magnitude, weighted RRF must respond to weights
without responding to magnitude, and min-max must respond to magnitude.
"""

from __future__ import annotations

import pytest

from ragpipe.fusion import (
    DEFAULT_RANK_CONSTANT,
    HybridRetriever,
    minmax_weighted,
    rrf,
    weighted_rrf,
)
from ragpipe.retrieval import Hit


def hits(*pairs: tuple[str, float]) -> list[Hit]:
    return [Hit(chunk_id=c, score=s, rank=i) for i, (c, s) in enumerate(pairs, start=1)]


class FakeRetriever:
    def __init__(self, name: str, ranked: list[Hit]) -> None:
        self.name = name
        self._ranked = ranked
        self.last_k: int | None = None

    def __len__(self) -> int:
        return len(self._ranked)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        self.last_k = k
        return self._ranked[:k]


# -- vanilla RRF ------------------------------------------------------------


def test_rrf_matches_hand_computed_scores():
    fused = rrf([hits(("a", 9.0), ("b", 1.0)), hits(("b", 0.5), ("c", 0.4))], k=10)
    by_id = {h.chunk_id: h.score for h in fused}
    k = DEFAULT_RANK_CONSTANT
    assert by_id["a"] == pytest.approx(1 / (k + 1))
    assert by_id["b"] == pytest.approx(1 / (k + 2) + 1 / (k + 1))
    assert by_id["c"] == pytest.approx(1 / (k + 2))
    # Agreed-on-by-both beats first-in-one: the defining RRF behaviour.
    assert fused[0].chunk_id == "b"


def test_rrf_ignores_score_magnitude():
    """Rank-only. Scaling scores must not move the ranking at all."""
    a = [hits(("x", 1000.0), ("y", 0.001)), hits(("y", 5.0), ("z", 4.0))]
    b = [hits(("x", 0.1), ("y", 0.09)), hits(("y", 1e6), ("z", 1.0))]
    assert [h.chunk_id for h in rrf(a, k=5)] == [h.chunk_id for h in rrf(b, k=5)]
    assert [h.score for h in rrf(a, k=5)] == [h.score for h in rrf(b, k=5)]


def test_rrf_rank_constant_controls_flatness():
    """Small constant sharpens top-rank advantage; large flattens it."""
    lists = [hits(("a", 1.0)), hits(("b", 1.0), ("c", 1.0), ("d", 1.0))]
    sharp = {h.chunk_id: h.score for h in rrf(lists, k=4, rank_constant=1)}
    flat = {h.chunk_id: h.score for h in rrf(lists, k=4, rank_constant=1000)}
    assert sharp["a"] / sharp["d"] > flat["a"] / flat["d"]


def test_rrf_rejects_nonpositive_rank_constant():
    for bad in (0, -1):
        with pytest.raises(ValueError, match="rank_constant"):
            rrf([hits(("a", 1.0))], rank_constant=bad)


def test_rrf_handles_empty_and_single_lists():
    assert rrf([[], []], k=5) == []
    assert [h.chunk_id for h in rrf([hits(("a", 1.0)), []], k=5)] == ["a"]


def test_fused_ranks_are_contiguous_from_one():
    fused = rrf([hits(("a", 1.0), ("b", 1.0)), hits(("c", 1.0))], k=10)
    assert [h.rank for h in fused] == [1, 2, 3]


def test_ties_broken_deterministically_by_chunk_id():
    """Two runs must not disagree on tied documents; a metric moving for no
    reason is worse than a wrong one, because nothing points at the cause."""
    forward = rrf([hits(("b", 1.0)), hits(("a", 1.0))], k=5)
    reverse = rrf([hits(("a", 1.0)), hits(("b", 1.0))], k=5)
    assert [h.chunk_id for h in forward] == [h.chunk_id for h in reverse] == ["a", "b"]


def test_k_truncates_output():
    fused = rrf([hits(("a", 3.0), ("b", 2.0), ("c", 1.0))], k=2)
    assert len(fused) == 2


# -- weighted RRF -----------------------------------------------------------


def test_weighted_rrf_weights_are_normalized():
    """[3, 1] and [0.75, 0.25] are the same weighting, so they must agree."""
    lists = [hits(("a", 1.0), ("b", 1.0)), hits(("b", 1.0), ("c", 1.0))]
    assert [h.score for h in weighted_rrf(lists, [3, 1], k=5)] == pytest.approx(
        [h.score for h in weighted_rrf(lists, [0.75, 0.25], k=5)]
    )


def test_weighted_rrf_equals_vanilla_under_equal_weights():
    """Equal weights normalize to 1/n, so scores are vanilla RRF scaled by 1/n —
    the same ranking. Asserting the ranking, not the scores, is the real claim."""
    lists = [hits(("a", 1.0), ("b", 1.0)), hits(("b", 1.0), ("c", 1.0))]
    assert [h.chunk_id for h in weighted_rrf(lists, [1, 1], k=5)] == [
        h.chunk_id for h in rrf(lists, k=5)
    ]


def test_weighted_rrf_weight_can_flip_the_winner():
    sparse = hits(("s", 10.0))
    dense = hits(("d", 0.9))
    assert weighted_rrf([sparse, dense], [0.9, 0.1], k=2)[0].chunk_id == "s"
    assert weighted_rrf([sparse, dense], [0.1, 0.9], k=2)[0].chunk_id == "d"


def test_weighted_rrf_still_ignores_magnitude():
    """The weights are on lists, not on scores. Magnitude must stay irrelevant."""
    a = [hits(("x", 1.0), ("y", 0.9)), hits(("y", 0.5), ("z", 0.4))]
    b = [hits(("x", 500.0), ("y", 0.001)), hits(("y", 99.0), ("z", 98.0))]
    assert [h.chunk_id for h in weighted_rrf(a, [0.3, 0.7], k=5)] == [
        h.chunk_id for h in weighted_rrf(b, [0.3, 0.7], k=5)
    ]


def test_weighted_rrf_zero_weight_excludes_a_list():
    fused = weighted_rrf([hits(("a", 1.0)), hits(("b", 1.0))], [1.0, 0.0], k=5)
    assert {h.chunk_id for h in fused if h.score > 0} == {"a"}


@pytest.mark.parametrize(
    ("weights", "match"),
    [([1.0], "expected 2 weights"), ([-1.0, 2.0], "non-negative"), ([0.0, 0.0], "sum to zero")],
)
def test_weighted_rrf_rejects_bad_weights(weights, match):
    with pytest.raises(ValueError, match=match):
        weighted_rrf([hits(("a", 1.0)), hits(("b", 1.0))], weights)


# -- min-max weighted score fusion -----------------------------------------


def test_minmax_normalizes_each_list_independently():
    """BM25's unbounded scores and cosine's [-1,1] must not be summed raw."""
    fused = minmax_weighted([hits(("a", 100.0), ("b", 0.0)), hits(("c", 0.9), ("d", 0.1))], [1, 1])
    by_id = {h.chunk_id: h.score for h in fused}
    assert by_id["a"] == pytest.approx(0.5)  # top of list 1, weight 0.5
    assert by_id["c"] == pytest.approx(0.5)  # top of list 2, same weight
    assert by_id["b"] == pytest.approx(0.0)
    assert by_id["d"] == pytest.approx(0.0)


def test_minmax_responds_to_magnitude_where_rrf_does_not():
    """The distinguishing property, isolated.

    `b` and `c` occupy mirror-image *ranks* (2nd in one list, 3rd in the other), so
    RRF must score them identically — it sees only positions. Their *scores* are not
    mirror images: `b` loses by a hair in list 1 while `c` loses badly in list 2. So
    min-max must separate them. Same input, one method blind to it and one not.
    """
    lists = [hits(("a", 10.0), ("b", 9.9), ("c", 0.0)), hits(("a", 10.0), ("c", 5.0), ("b", 0.0))]

    by_rrf = {h.chunk_id: h.score for h in rrf(lists, k=3)}
    assert by_rrf["b"] == pytest.approx(by_rrf["c"])

    by_minmax = {h.chunk_id: h.score for h in minmax_weighted(lists, [1, 1], k=3)}
    assert by_minmax["b"] == pytest.approx(0.5 * 0.99)
    assert by_minmax["c"] == pytest.approx(0.5 * 0.5)
    assert by_minmax["b"] > by_minmax["c"]


def test_minmax_discards_agreement_at_the_bottom_of_both_lists():
    """The flip side, and a genuine weakness worth stating.

    A document ranked last in every list normalizes to 0 everywhere, so min-max
    scores it zero no matter how many retrievers found it. RRF rewards that
    agreement instead. This is why the two methods get separate rows rather than
    one being presented as strictly better.
    """
    lists = [hits(("a", 1.0), ("shared", 0.99)), hits(("c", 1.0), ("shared", 0.99))]
    assert rrf(lists, k=3)[0].chunk_id == "shared"
    assert {h.chunk_id: h.score for h in minmax_weighted(lists, [1, 1], k=3)}["shared"] == 0.0


def test_minmax_zero_spread_maps_to_half_not_one():
    """A lone hit was the only candidate, not a confident one. Mapping it to 1.0
    would let a single-hit list outvote a full list of real evidence."""
    fused = minmax_weighted([hits(("solo", 42.0)), hits(("a", 1.0), ("b", 0.0))], [0.5, 0.5])
    by_id = {h.chunk_id: h.score for h in fused}
    assert by_id["solo"] == pytest.approx(0.25)  # 0.5 spread-less * 0.5 weight
    assert by_id["a"] == pytest.approx(0.5)
    assert by_id["a"] > by_id["solo"]


def test_minmax_all_tied_scores_do_not_divide_by_zero():
    fused = minmax_weighted([hits(("a", 5.0), ("b", 5.0), ("c", 5.0))], [1.0], k=3)
    assert [h.score for h in fused] == pytest.approx([0.5, 0.5, 0.5])


def test_minmax_empty_list_contributes_nothing():
    fused = minmax_weighted([hits(("a", 1.0), ("b", 0.0)), []], [0.5, 0.5], k=5)
    assert {h.chunk_id for h in fused} == {"a", "b"}


def test_minmax_window_dependence_is_real():
    """Documented failure mode, asserted so it stays documented: min-max
    normalizes against the retrieved window, so truncating the input changes
    scores. RRF has no equivalent sensitivity."""
    full = [hits(("a", 10.0), ("b", 5.0), ("c", 0.0))]
    narrow = [hits(("a", 10.0), ("b", 5.0))]
    b_full = {h.chunk_id: h.score for h in minmax_weighted(full, [1.0], k=5)}["b"]
    b_narrow = {h.chunk_id: h.score for h in minmax_weighted(narrow, [1.0], k=5)}["b"]
    assert b_full == pytest.approx(0.5)
    assert b_narrow == pytest.approx(0.0)
    assert b_full != b_narrow


# -- HybridRetriever --------------------------------------------------------


def test_hybrid_fetches_more_than_it_returns():
    """fetch_k, not k, is what gives fusion room to promote a mid-ranked
    document that both retrievers liked moderately."""
    a = FakeRetriever("a", hits(*[(f"a{i}", 1.0 / (i + 1)) for i in range(50)]))
    b = FakeRetriever("b", hits(*[(f"b{i}", 1.0 / (i + 1)) for i in range(50)]))
    hybrid = HybridRetriever([a, b], method="rrf", fetch_k=40)
    out = hybrid.search("q", k=5)
    assert len(out) == 5
    assert a.last_k == b.last_k == 40


def test_hybrid_promotes_document_ranked_mid_by_both():
    """The case fusion exists for, and the case a top-10-only fusion misses."""
    a = FakeRetriever("a", hits(*[(f"x{i}", 1.0) for i in range(20)], ("shared", 0.5)))
    b = FakeRetriever("b", hits(*[(f"y{i}", 1.0) for i in range(20)], ("shared", 0.5)))
    wide = HybridRetriever([a, b], method="rrf", fetch_k=100).search("q", k=1)
    assert wide[0].chunk_id == "shared"
    narrow = HybridRetriever([a, b], method="rrf", fetch_k=10).search("q", k=1)
    assert narrow[0].chunk_id != "shared"


def test_hybrid_rejects_bad_config():
    r = FakeRetriever("a", hits(("a", 1.0)))
    with pytest.raises(ValueError, match="at least two"):
        HybridRetriever([r], method="rrf")
    with pytest.raises(ValueError, match="method must be"):
        HybridRetriever([r, r], method="nope")


def test_hybrid_default_names_distinguish_methods_and_weights():
    r1, r2 = FakeRetriever("bm25", []), FakeRetriever("dense", [])
    assert HybridRetriever([r1, r2], method="rrf").name == "bm25+dense rrf"
    weighted = HybridRetriever([r1, r2], method="minmax", weights=[0.3, 0.7]).name
    assert "minmax" in weighted and "0.7" in weighted.replace("0.70", "0.7")
