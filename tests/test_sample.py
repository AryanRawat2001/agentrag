"""Tests for stratified sampling.

Regression tests for defects found in the Phase 0 verification gate: alphabetical
starvation in the round-robin, a pair budget that was double its documented size,
samplers that overshot their target, and cross-source RNG coupling.
"""

from __future__ import annotations

import random

from ragpipe.models import SourceDoc
from ragpipe.sample import _round_robin, sample_ctgov, sample_fda


def _fda_doc(i: int, center: str, status: str, year: int) -> SourceDoc:
    return SourceDoc(
        doc_id=f"fda-{i}",
        source="fda_guidance",
        title=f"Guidance {i}",
        url=f"https://example.test/media/{i}/download",
        status=status,
        center=center,
        issue_date=f"{year}-01-01",
    )


def _ctgov_doc(i: int, sponsor_class: str, size: int = 1000) -> SourceDoc:
    return SourceDoc(
        doc_id=f"ctgov-NCT{i:08d}-Prot_000",
        source="ctgov_protocol",
        title=f"Study {i}",
        url=f"https://example.test/{i}.pdf",
        docket=f"NCT{i:08d}",
        extra={"sponsor_class": sponsor_class, "declared_bytes": size},
    )


class TestRoundRobin:
    def test_no_stratum_is_systematically_starved(self):
        """When target < len(strata), the choice of who misses out must be
        unbiased rather than alphabetical.

        The original implementation iterated a sorted key list and broke mid-pass,
        so the alphabetically last strata always received zero documents. On the
        real data that meant `--fda-n 40` over 58 strata produced a
        CBER/CDER/CDRH-only corpus with nothing from Human Foods Program or ORA.
        Verified here by checking that the starved set actually varies with the
        seed instead of always being the tail.
        """
        strata = {f"s{i:02d}": [f"doc{i}-{j}" for j in range(5)] for i in range(20)}
        starved_sets = set()
        for seed in range(8):
            picked = _round_robin(strata, 12, random.Random(seed))
            assert len(picked) == 12
            contributing = frozenset(p.split("-")[0] for p in picked)
            starved_sets.add(frozenset(strata) - {f"s{c[3:]:0>2}" for c in contributing})
        # More than one distinct starvation pattern across seeds proves the
        # selection is not fixed to the alphabetical tail.
        assert len(starved_sets) > 1

    def test_respects_target_exactly(self):
        strata = {f"s{i}": list(range(5)) for i in range(4)}
        assert len(_round_robin(strata, 7, random.Random(0))) == 7

    def test_cannot_exceed_available_documents(self):
        strata = {"a": [1, 2], "b": [3]}
        assert len(_round_robin(strata, 99, random.Random(0))) == 3

    def test_empty_strata_returns_empty(self):
        assert _round_robin({}, 10, random.Random(0)) == []
        assert _round_robin({"a": []}, 10, random.Random(0)) == []


class TestSampleFda:
    def _pool(self, n: int = 400) -> list[SourceDoc]:
        docs = []
        centers = ["CDER", "CDRH", "CBER", "CVM"]
        for i in range(n):
            status = "Draft" if i % 2 else "Final"
            docs.append(_fda_doc(i, centers[i % 4], status, 1995 + (i % 30)))
        # Matched draft/final pairs share a normalised title.
        for i in range(n, n + 40):
            docs.append(_fda_doc(i, "CDER", "Draft" if i % 2 else "Final", 2020))
            docs[-1].title = f"Shared Topic {i // 2}"
        return docs

    def test_never_exceeds_target(self):
        """Small targets used to overshoot because of a `max(4, ...)` floor:
        --fda-n 4 silently returned 8 documents."""
        pool = self._pool()
        for target in (0, 1, 2, 3, 5, 17, 120):
            assert len(sample_fda(pool, target, seed=1)) <= target

    def test_zero_target_returns_empty(self):
        assert sample_fda(self._pool(), 0, seed=1) == []

    def test_pair_share_is_roughly_as_documented(self):
        """The budget is expressed in documents, not titles.

        It previously counted titles while the comment claimed a document share,
        so the real figure was double the documented one (~17% against "~8%").
        """
        selected = sample_fda(self._pool(), 120, seed=1)
        paired = sum(1 for d in selected if d.extra.get("sample_reason") == "draft_final_pair")
        assert paired % 2 == 0, "pairs must be complete"
        assert paired <= 120 * 0.20

    def test_is_deterministic_for_a_given_seed(self):
        pool = self._pool()
        first = [d.doc_id for d in sample_fda(pool, 60, seed=7)]
        second = [d.doc_id for d in sample_fda(self._pool(), 60, seed=7)]
        assert first == second

    def test_different_seeds_give_different_samples(self):
        pool = self._pool()
        a = [d.doc_id for d in sample_fda(pool, 60, seed=1)]
        b = [d.doc_id for d in sample_fda(self._pool(), 60, seed=2)]
        assert a != b

    def test_tolerates_malformed_issue_dates(self):
        """`_era` used to raise ValueError on a non-numeric date; ctgov supplies
        unvalidated raw dates, so the guard is load-bearing."""
        pool = self._pool(40)
        pool[0].issue_date = "N/A"
        pool[1].issue_date = None
        pool[2].issue_date = ""
        assert len(sample_fda(pool, 10, seed=1)) == 10


class TestSampleCtgov:
    def _pool(self, n: int = 200) -> list[SourceDoc]:
        classes = ["INDUSTRY", "NIH", "OTHER", "NETWORK"]
        return [_ctgov_doc(i, classes[i % 4], size=1000 * (i + 1)) for i in range(n)]

    def test_never_exceeds_target(self):
        pool = self._pool()
        for target in (0, 1, 2, 3, 40):
            assert len(sample_ctgov(pool, target, seed=1)) <= target

    def test_largest_documents_are_included(self):
        selected = sample_ctgov(self._pool(), 40, seed=1)
        reasons = {d.extra.get("sample_reason") for d in selected}
        assert "largest_available_document" in reasons

    def test_large_documents_are_not_excluded_from_the_remainder(self):
        """A previous version filtered every >=10MB document out of the remainder
        pool, removing candidates while the force-include contributed nothing it
        claimed to. With a pool that is entirely large, sampling must still work."""
        pool = [_ctgov_doc(i, "INDUSTRY", size=50_000_000) for i in range(30)]
        assert len(sample_ctgov(pool, 10, seed=1)) == 10

    def test_spreads_across_sponsor_class(self):
        selected = sample_ctgov(self._pool(), 40, seed=1)
        classes = {d.extra["sponsor_class"] for d in selected}
        assert len(classes) == 4


class TestCrossSourceIndependence:
    def test_fda_target_does_not_perturb_the_protocol_sample(self):
        """The two samplers shared one Random, so the protocol sample depended on
        how much entropy the FDA stage happened to consume. Holding the seed and
        changing only --fda-n changed the protocol sample, which quietly broke the
        reproducibility claim. Each sampler now seeds its own RNG.
        """
        fda_pool = [_fda_doc(i, "CDER", "Final", 2020) for i in range(300)]
        ctgov_pool = [_ctgov_doc(i, "INDUSTRY") for i in range(200)]

        sample_fda(fda_pool, 120, seed=42)
        first = [d.doc_id for d in sample_ctgov(ctgov_pool, 40, seed=42)]

        sample_fda(fda_pool, 200, seed=42)  # different FDA target
        second = [d.doc_id for d in sample_ctgov(ctgov_pool, 40, seed=42)]

        assert first == second
