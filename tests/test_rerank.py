"""Reranker tests.

The reranker's defining constraint is what these assert: it permutes a candidate
list and can never add to it. A rerank row that beats its own candidate list's
recall ceiling is a wiring bug, and the failure mode is a *better-looking* table,
which is the hardest kind to notice.

No cross-encoder is loaded — a fake scorer stands in, so the ordering logic is
tested without a model download.
"""

from __future__ import annotations

import pytest

from ragpipe.rerank import (
    RERANKERS,
    CrossEncoderReranker,
    ceiling_from_ids,
    recall_ceiling,
)
from ragpipe.retrieval import Hit


class FakeBase:
    def __init__(self, ranked: list[str]) -> None:
        self.name = "fake-base"
        self._ranked = ranked
        self.last_k: int | None = None

    def __len__(self) -> int:
        return len(self._ranked)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        self.last_k = k
        return [
            Hit(chunk_id=c, score=1.0 / i, rank=i) for i, c in enumerate(self._ranked[:k], start=1)
        ]


class FakeCE:
    """Scores by a fixed table, so the desired permutation is explicit."""

    def __init__(self, table: dict[str, float]) -> None:
        self.table = table
        self.seen: list[tuple[str, str]] = []

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        self.seen.extend(pairs)
        return [self.table.get(doc, 0.0) for _q, doc in pairs]


def build(ranked: list[str], scores: dict[str, float], **kw) -> CrossEncoderReranker:
    chunks = [{"chunk_id": c, "text": c, "embed_text": c, "section_heading": None} for c in ranked]
    r = CrossEncoderReranker(FakeBase(ranked), chunks, **kw)
    r._ce = FakeCE(scores)
    return r


def test_rerank_reorders_by_cross_encoder_score():
    r = build(["a", "b", "c"], {"a": 0.1, "b": 0.9, "c": 0.5}, candidate_k=3)
    assert [h.chunk_id for h in r.search("q", k=3)] == ["b", "c", "a"]


def test_rerank_ranks_are_contiguous_and_scores_descending():
    r = build(["a", "b", "c"], {"a": 0.1, "b": 0.9, "c": 0.5}, candidate_k=3)
    hits = r.search("q", k=3)
    assert [h.rank for h in hits] == [1, 2, 3]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_rerank_returns_cross_encoder_scores_not_first_stage_scores():
    """The score column must reflect the model that produced the ordering,
    otherwise the table shows scores inconsistent with its own ranking."""
    r = build(["a", "b"], {"a": 0.25, "b": 0.75}, candidate_k=2)
    assert {h.chunk_id: h.score for h in r.search("q", k=2)} == pytest.approx(
        {"a": 0.25, "b": 0.75}
    )


def test_rerank_cannot_retrieve_outside_the_candidate_window():
    """The core constraint. `good` sits at position 5, outside candidate_k=3, so no
    cross-encoder score can rescue it."""
    r = build(
        ["a", "b", "c", "d", "good"],
        {"good": 99.0, "a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4},
        candidate_k=3,
    )
    assert "good" not in [h.chunk_id for h in r.search("q", k=3)]


def test_widening_candidate_k_lets_the_reranker_find_it():
    scores = {"good": 99.0, "a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    wide = build(["a", "b", "c", "d", "good"], scores, candidate_k=5)
    assert wide.search("q", k=1)[0].chunk_id == "good"


def test_rerank_output_is_a_subset_of_its_candidates():
    """Property form of the same constraint: reranking must never invent an id."""
    ranked = [f"c{i}" for i in range(20)]
    r = build(ranked, {c: float(i % 7) for i, c in enumerate(ranked)}, candidate_k=8)
    got = {h.chunk_id for h in r.search("q", k=8)}
    assert got <= set(ranked[:8])


def test_rerank_never_returns_fewer_than_k_when_candidates_allow():
    """candidate_k below k must not shrink the result list."""
    r = build([f"c{i}" for i in range(10)], {}, candidate_k=2)
    assert len(r.search("q", k=5)) == 5
    assert r.base.last_k == 5  # asked the base for k, not candidate_k


def test_rerank_requests_candidate_k_when_it_exceeds_k():
    r = build([f"c{i}" for i in range(60)], {}, candidate_k=50)
    r.search("q", k=10)
    assert r.base.last_k == 50


def test_rerank_ties_broken_deterministically():
    a = build(["x", "y"], {"x": 1.0, "y": 1.0}, candidate_k=2).search("q", k=2)
    b = build(["y", "x"], {"x": 1.0, "y": 1.0}, candidate_k=2).search("q", k=2)
    assert [h.chunk_id for h in a] == [h.chunk_id for h in b] == ["x", "y"]


def test_rerank_handles_empty_candidate_list():
    r = build([], {}, candidate_k=5)
    assert r.search("q", k=5) == []


def test_rerank_scores_query_against_chunk_text():
    r = build(["a"], {"a": 1.0}, candidate_k=1)
    r.search("what is the dose?", k=1)
    assert r._ce.seen == [("what is the dose?", "a")]


def test_rerank_tolerates_candidate_missing_from_text_map():
    """A base retriever built over different chunks must degrade, not crash."""
    chunks = [{"chunk_id": "a", "text": "a", "embed_text": "a", "section_heading": None}]
    r = CrossEncoderReranker(FakeBase(["a", "ghost"]), chunks, candidate_k=2)
    r._ce = FakeCE({"a": 0.5})
    assert [h.chunk_id for h in r.search("q", k=2)] == ["a", "ghost"]


def test_rerank_rejects_bad_config():
    chunks = [{"chunk_id": "a", "text": "a", "embed_text": "a", "section_heading": None}]
    with pytest.raises(ValueError, match="unknown reranker"):
        CrossEncoderReranker(FakeBase(["a"]), chunks, model="nope")
    with pytest.raises(ValueError, match="candidate_k"):
        CrossEncoderReranker(FakeBase(["a"]), chunks, candidate_k=0)


def test_rerank_name_records_model_and_candidate_k():
    """Both belong in the row label: candidate_k sets the recall ceiling, so a row
    without it cannot be read."""
    r = build(["a"], {}, candidate_k=25, model="minilm-l6")
    assert "minilm-l6" in r.name and "25" in r.name and "fake-base" in r.name


def test_reranker_registry_records_windows():
    assert RERANKERS["bge-reranker-v2-m3"]["window"] > RERANKERS["bge-reranker-base"]["window"]
    assert all("hf_id" in v for v in RERANKERS.values())


# -- recall ceiling ---------------------------------------------------------


def test_recall_ceiling_measures_what_the_window_contains():
    base = [Hit(chunk_id=f"c{i}", score=1.0, rank=i) for i in range(1, 11)]
    assert recall_ceiling(base, {"c1", "c2"}, candidate_k=5) == pytest.approx(1.0)
    assert recall_ceiling(base, {"c1", "c9"}, candidate_k=5) == pytest.approx(0.5)
    assert recall_ceiling(base, {"c9", "c10"}, candidate_k=5) == pytest.approx(0.0)


def test_recall_ceiling_is_zero_for_unanswerable():
    base = [Hit(chunk_id="c1", score=1.0, rank=1)]
    assert recall_ceiling(base, set(), candidate_k=5) == 0.0


def test_recall_ceiling_never_exceeds_one_with_duplicate_hits():
    base = [Hit(chunk_id="c1", score=1.0, rank=1), Hit(chunk_id="c1", score=0.9, rank=2)]
    assert recall_ceiling(base, {"c1"}, candidate_k=5) == pytest.approx(1.0)


def test_ceiling_from_ids_matches_recall_ceiling():
    """The harness uses the id-based form so it scores the list actually reranked.
    Both forms must agree, or the reported ceiling describes a different list."""
    base = [Hit(chunk_id=f"c{i}", score=1.0, rank=i) for i in range(1, 11)]
    ids = [h.chunk_id for h in base]
    for relevant in ({"c1", "c2"}, {"c1", "c9"}, {"c9", "c10"}, set()):
        assert ceiling_from_ids(ids, relevant, 5) == pytest.approx(
            recall_ceiling(base, relevant, 5)
        )


def test_search_records_the_candidates_it_scored():
    """Recomputing the ceiling by re-running the base retriever could describe a
    different list than the one reranked."""
    r = build(["a", "b", "c", "d"], {"a": 1.0}, candidate_k=3)
    r.search("q", k=2)
    assert r.last_candidate_ids == ["a", "b", "c"]


def test_last_candidates_recorded_even_when_empty():
    r = build([], {}, candidate_k=3)
    r.search("q", k=2)
    assert r.last_candidate_ids == []


def test_ceiling_reflects_candidate_k_not_the_returned_k():
    r = build([f"c{i}" for i in range(20)], {}, candidate_k=10)
    r.search("q", k=5)
    assert ceiling_from_ids(r.last_candidate_ids, {"c7"}, r.candidate_k) == pytest.approx(1.0)
    assert ceiling_from_ids(r.last_candidate_ids, {"c15"}, r.candidate_k) == pytest.approx(0.0)


@pytest.mark.parametrize("candidate_k", [1, 5, 10, 20, 50])
def test_recall_never_exceeds_the_ceiling_for_any_candidate_k(candidate_k):
    """Regression for the review's MEDIUM finding.

    `search` scores `max(candidate_k, k)` candidates while the ceiling was computed
    over `candidate_k`, so any `--candidate-k` below the harness's k (20) produced
    rows whose recall legitimately exceeded their own printed ceiling — which the
    report prose calls a wiring bug. The default of 50 hid it.
    """
    ranked = [f"c{i}" for i in range(60)]
    relevant = {"c8"}  # inside a 20-wide window, outside a 5-wide one
    r = build(
        ranked, {c: float(len(ranked) - i) for i, c in enumerate(ranked)}, candidate_k=candidate_k
    )
    hits = r.search("q", k=20)
    ceiling = ceiling_from_ids(r.last_candidate_ids, relevant, r.effective_candidate_k)
    recall = len({h.chunk_id for h in hits[:20]} & relevant) / len(relevant)
    assert recall <= ceiling, f"recall {recall} above ceiling {ceiling}"


def test_effective_candidate_k_reflects_the_window_actually_scored():
    r = build([f"c{i}" for i in range(60)], {}, candidate_k=5)
    r.search("q", k=20)
    assert r.effective_candidate_k == 20  # k won, not candidate_k
    r.search("q", k=3)
    assert r.effective_candidate_k == 5  # candidate_k won


def test_effective_candidate_k_is_capped_by_corpus_size():
    r = build(["a", "b"], {}, candidate_k=50)
    r.search("q", k=10)
    assert r.effective_candidate_k == 2


def test_reranked_recall_never_exceeds_the_ceiling():
    """The invariant that catches a miswired rerank row, asserted end to end."""
    ranked = [f"c{i}" for i in range(20)]
    relevant = {"c0", "c15"}  # one inside a 10-candidate window, one outside
    r = build(ranked, {c: float(len(ranked) - i) for i, c in enumerate(ranked)}, candidate_k=10)
    base_hits = r.base.search("q", k=10)
    ceiling = recall_ceiling(base_hits, relevant, candidate_k=10)
    got = {h.chunk_id for h in r.search("q", k=10)}
    assert len(got & relevant) / len(relevant) <= ceiling
    assert ceiling == pytest.approx(0.5)
