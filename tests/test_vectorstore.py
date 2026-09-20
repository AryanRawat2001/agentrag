"""Tests for the Qdrant serving path.

Weighted toward the local-mode guard, because that is where this module already produced
a table of perfect numbers that measured nothing: local Qdrant brute-forces every query,
so an `hnsw_ef` sweep against it reported index recall 1.0000 at every setting from 4 to
256. The fix is that the sweep refuses to run, and these tests pin that.

Local mode is fine for everything *except* measuring approximation, so the retriever
itself is tested against it.
"""

from __future__ import annotations

import itertools
import re
import statistics

import numpy as np
import pytest

from ragpipe import vectorstore as vs

#: `qdrant-client` moved to the `vectorstore` extra in Phase 7. Guarded for the same
#: reason `test_contextual.py` guards `anthropic`: a core-only install is the one the
#: container uses, and there a skip says "not exercised here" where a failure would say
#: "broken". Without this, 14 tests errored and two failed *misleadingly* -- including a
#: `pytest.raises(ValueError)` that reported its guard broken when the real cause was
#: `from_dense` importing qdrant_client before reaching the ValueError.
pytest.importorskip("qdrant_client", reason="needs the `vectorstore` extra")


class FakeDense:
    """Stands in for `DenseRetriever`: unit vectors and the two attributes used."""

    name = "dense fake"

    def __init__(self, n=8, dim=4):
        # Seeded random, then unit-normalised, so every score is distinct. An earlier
        # fixture built vectors as `e[i % dim] + 0.1 * e[(i+1) % dim]`, which makes rows
        # 0/4, 1/5, 2/6 and 3/7 *identical* -- so every score tied, and the exact and
        # Qdrant paths broke the ties differently. The comparison test caught it, which is
        # what it is for; the non-determinism was in the fixture, not the code.
        rng = np.random.default_rng(20260819)
        m = (
            rng.standard_normal((n, dim)).astype(np.float32)
            if n
            else np.zeros((0, dim), np.float32)
        )
        if n:
            m /= np.linalg.norm(m, axis=1, keepdims=True)
        self.matrix = m
        self._chunk_ids = [f"c{i}" for i in range(n)]

    def _query_vector(self, query: str) -> np.ndarray:
        return self.matrix[0]

    def search(self, query: str, k: int = 10):
        from ragpipe.retrieval import Hit

        scores = self.matrix @ self._query_vector(query)
        order = np.argsort(-scores)[:k]
        return [
            Hit(chunk_id=self._chunk_ids[i], score=float(scores[i]), rank=r)
            for r, i in enumerate(order, 1)
        ]


@pytest.fixture
def store():
    dense = FakeDense()
    s = vs.QdrantStore.from_dense(dense, collection="test")
    yield s, dense
    s.close()


class TestLocalModeCannotMeasureApproximation:
    def test_local_mode_is_detected(self, store):
        s, _ = store
        assert s.is_local

    def test_sweep_refuses_rather_than_returning_perfect_numbers(self, store):
        """The defect: comparing exact search to exact search and reporting 1.0000."""
        s, dense = store
        with pytest.raises(RuntimeError, match="brute-force"):
            vs.sweep_recall(s, dense, ["anything"], k=2)

    def test_the_refusal_names_how_to_actually_measure_it(self, store):
        s, dense = store
        with pytest.raises(RuntimeError) as exc:
            vs.sweep_recall(s, dense, ["q"], k=2)
        assert "docker run" in str(exc.value)
        assert "url=" in str(exc.value)

    def test_detection_is_structural_not_remembered(self, store):
        """It inspects the client, so it stays right if a caller builds one itself."""
        from qdrant_client import QdrantClient

        s, _ = store
        s.client = QdrantClient(location=":memory:")
        assert s.is_local


class TestRetrieverContract:
    def test_returns_ranked_hits(self, store):
        s, dense = store
        hits = vs.QdrantRetriever(s, dense).search("q", k=4)
        assert len(hits) == 4
        assert [h.rank for h in hits] == [1, 2, 3, 4]
        assert all(isinstance(h.chunk_id, str) for h in hits)

    def test_scores_are_non_increasing(self, store):
        s, dense = store
        scores = [h.score for h in vs.QdrantRetriever(s, dense).search("q", k=8)]
        assert scores == sorted(scores, reverse=True)

    def test_k_is_clamped_to_the_collection(self, store):
        s, dense = store
        assert len(vs.QdrantRetriever(s, dense).search("q", k=999)) == len(s)

    def test_k_below_one_is_clamped_up(self, store):
        s, dense = store
        assert len(vs.QdrantRetriever(s, dense).search("q", k=0)) == 1

    def test_empty_collection_returns_nothing(self):
        dense = FakeDense(n=0)
        dense.matrix = np.zeros((0, 4), dtype=np.float32)
        dense._chunk_ids = []
        s = vs.QdrantStore.from_dense(dense, collection="empty")
        try:
            assert vs.QdrantRetriever(s, dense).search("q", k=5) == []
        finally:
            s.close()

    def test_it_satisfies_the_retriever_protocol(self, store):
        """So it drops into the existing eval harness as another row."""
        s, dense = store
        r = vs.QdrantRetriever(s, dense)
        assert isinstance(r.name, str) and callable(r.search) and len(r) == len(s)

    def test_the_fixture_has_no_tied_scores(self, store):
        """Precondition for the comparison below: ties make the two engines legitimately
        disagree, so a tied fixture would test tie-break parity rather than correctness."""
        s, dense = store
        scores = sorted(h.score for h in dense.search("q", k=len(s)))
        assert len(set(round(x, 6) for x in scores)) == len(scores)

    def test_local_mode_returns_the_same_top_k_as_exact(self, store):
        """Not a recall measurement -- a correctness one. Local mode brute-forces, so it
        *must* agree with exact search; disagreement would mean the vectors or the id
        mapping are wrong, which is the thing this test is for."""
        s, dense = store
        got = [h.chunk_id for h in vs.QdrantRetriever(s, dense).search("q", k=5)]
        want = [h.chunk_id for h in dense.search("q", k=5)]
        assert got == want


class TestStoreConstruction:
    def test_point_ids_derive_from_chunk_ids_not_position(self, store):
        """A re-upsert of the same corpus must land on the same points, so a partial
        rebuild cannot duplicate a chunk under a second id."""
        s, dense = store
        first = dict(s._ids)
        s2 = vs.QdrantStore.from_dense(dense, collection="test2")
        try:
            assert set(first.values()) == set(s2._ids.values())
            assert set(first.keys()) == set(s2._ids.keys())
        finally:
            s2.close()

    def test_vector_id_mismatch_is_rejected(self):
        dense = FakeDense()
        dense._chunk_ids = dense._chunk_ids[:-1]
        with pytest.raises(ValueError, match="vector/id mismatch"):
            vs.QdrantStore.from_dense(dense, collection="bad")

    def test_dimension_is_taken_from_the_matrix(self, store):
        s, dense = store
        assert s.dim == dense.matrix.shape[1]

    def test_the_collection_declares_both_vector_kinds(self, store):
        """The locked decision is dense *and* sparse in one store."""
        s, _ = store
        info = s.client.get_collection(s.collection)
        assert vs.DENSE_VECTOR in info.config.params.vectors
        assert vs.SPARSE_VECTOR in (info.config.params.sparse_vectors or {})

    def test_hnsw_build_params_are_the_documented_defaults(self):
        """The phase measures what a default configuration costs."""
        assert vs.DEFAULT_HNSW_M == 16
        assert vs.DEFAULT_HNSW_EF_CONSTRUCT == 100

    def test_the_ef_sweep_spans_below_and_above_typical_k(self):
        assert min(vs.DEFAULT_EF_SWEEP) < 10 < max(vs.DEFAULT_EF_SWEEP)


class TestRecallPoint:
    def test_serialises_with_rounded_figures(self):
        p = vs.RecallPoint(
            ef=16,
            k=10,
            n_queries=60,
            recall_at_k=0.8333333,
            exact_match_rate=0.5,
            p50_ms=1.23456,
            p95_ms=2.34567,
        )
        d = p.as_dict()
        assert d["recall_at_k"] == 0.8333
        assert d["p50_ms"] == 1.235
        assert d["ef"] == 16 and d["n_queries"] == 60


class TestAggregateBuilds:
    """HNSW construction is randomised, so one build is a sample, not the answer."""

    @staticmethod
    def _point(ef, recall, identical, p50=3.0, p95=5.0):
        return vs.RecallPoint(
            ef=ef,
            k=10,
            n_queries=60,
            recall_at_k=recall,
            exact_match_rate=identical,
            p50_ms=p50,
            p95_ms=p95,
        )

    def test_no_builds_is_no_rows(self):
        assert vs.aggregate_builds([]) == []

    def test_headline_is_the_median_and_the_range_is_the_extremes(self):
        builds = [
            [self._point(128, 0.99, 0.90)],
            [self._point(128, 1.00, 1.00)],
            [self._point(128, 0.98, 0.88)],
        ]
        (row,) = vs.aggregate_builds(builds)
        assert row["recall_at_k"] == 0.99  # median, not mean (0.99 either way here)
        assert row["recall_min"] == 0.98
        assert row["recall_max"] == 1.0
        assert row["exact_match_rate"] == 0.90
        assert row["identical_min"] == 0.88
        assert row["identical_max"] == 1.0
        assert row["n_builds"] == 3

    def test_the_median_is_not_the_mean(self):
        """An outlier build must not drag the headline. This is the case that matters:
        the original report's 1.0000 was one build."""
        builds = [[self._point(128, r, r)] for r in (0.90, 0.91, 1.00)]
        (row,) = vs.aggregate_builds(builds)
        assert row["recall_at_k"] == 0.91

    def test_latency_takes_the_median_p50_and_the_worst_p95(self):
        builds = [
            [self._point(64, 1.0, 1.0, p50=2.0, p95=4.0)],
            [self._point(64, 1.0, 1.0, p50=3.0, p95=9.0)],
            [self._point(64, 1.0, 1.0, p50=4.0, p95=5.0)],
        ]
        (row,) = vs.aggregate_builds(builds)
        assert row["p50_ms"] == 3.0
        assert row["p95_ms"] == 9.0

    def test_ef_order_is_preserved_including_the_default_row(self):
        builds = [[self._point(None, 1.0, 1.0), self._point(4, 0.8, 0.3)]] * 2
        rows = vs.aggregate_builds(builds)
        assert [r["ef"] for r in rows] == [None, 4]

    def test_an_ef_missing_from_one_build_is_aggregated_over_what_exists(self):
        builds = [
            [self._point(4, 0.80, 0.30), self._point(8, 0.85, 0.35)],
            [self._point(4, 0.90, 0.40)],
        ]
        rows = {r["ef"]: r for r in vs.aggregate_builds(builds)}
        assert rows[4]["n_builds"] == 2
        assert rows[8]["n_builds"] == 1


def _ann_payload(n_builds=3, **over):
    """A payload shaped like the one `ragpipe ann` writes."""
    sweep = [
        {
            "ef": ef,
            "k": 10,
            "n_queries": 60,
            "n_builds": n_builds,
            "recall_at_k": recall,
            "recall_min": recall - 0.005,
            "recall_max": recall + 0.005,
            "exact_match_rate": identical,
            "identical_min": identical - 0.01,
            "identical_max": identical + 0.01,
            "p50_ms": 3.0,
            "p95_ms": 5.0,
        }
        for ef, recall, identical in ((4, 0.8567, 0.35), (128, 0.995, 0.95))
    ]
    payload = {
        "qdrant_version": "1.19.0",
        "n_vectors": 13423,
        "indexed_vectors": 13423,
        "dim": 384,
        "m": 16,
        "ef_construct": 100,
        "full_scan_threshold": 10000,
        "k": 10,
        "n_queries": 60,
        "n_builds": n_builds,
        "within_build_identical": True,
        "exact_p50_ms": 0.45,
        "exact_p95_ms": 0.56,
        "bare_http_p50_ms": 0.85,
        "default": {
            "ef": None,
            "k": 10,
            "n_queries": 60,
            "n_builds": n_builds,
            "recall_at_k": 0.9883,
            "recall_min": 0.9867,
            "recall_max": 0.9917,
            "exact_match_rate": 0.8833,
            "identical_min": 0.8833,
            "identical_max": 0.9167,
            "p50_ms": 3.39,
            "p95_ms": 5.15,
        },
        "sweep": sweep,
    }
    payload.update(over)
    return payload


def _payload_floats(obj, out=None):
    out = set() if out is None else out
    if isinstance(obj, dict):
        for v in obj.values():
            _payload_floats(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _payload_floats(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(f"{float(obj):.4f}")
    return out


class TestRenderAnnReport:
    """The renderer had no test and had already shipped a stale literal: prose reading
    "at ef=16 recall is still 0.9567" above a table that said 0.9367."""

    def test_no_four_decimal_figure_appears_that_is_not_in_the_payload(self):
        """A structural guard against the whole class of defect. Any `d.dddd` in the
        output must be derivable from the payload -- 1.0000 is allowed because the
        exact baseline is 1 by definition, not by measurement."""
        report = vs.render_ann_report(_ann_payload())
        allowed = _payload_floats(_ann_payload()) | {"1.0000"}
        found = set(re.findall(r"\d+\.\d{4}", report))
        assert found <= allowed, f"figures not from the payload: {sorted(found - allowed)}"

    def test_changing_a_payload_figure_changes_the_prose(self):
        """The other half of the same guard: the first test passes trivially if the
        prose quotes nothing at all."""
        base = vs.render_ann_report(_ann_payload())
        moved = _ann_payload()
        moved["default"]["recall_at_k"] = 0.5000
        assert "0.5000" in vs.render_ann_report(moved)
        assert vs.render_ann_report(moved) != base

    def test_the_default_row_is_labelled_and_comes_first(self):
        report = vs.render_ann_report(_ann_payload())
        table = [ln for ln in report.splitlines() if ln.startswith("|") and "---" not in ln]
        assert "default (unset)" in table[1]

    def test_a_single_build_does_not_render_a_range(self):
        """`0.9883-0.9883` would read as a measured range that came out tight, which is
        the opposite of what one sample means."""
        payload = _ann_payload(n_builds=1)
        payload["default"]["n_builds"] = 1
        for row in payload["sweep"]:
            row["n_builds"] = 1
        report = vs.render_ann_report(payload)
        assert "1 independent index build**." in report
        table = [ln for ln in report.splitlines() if ln.startswith("|") and "---" not in ln]
        for row in table[1:]:
            assert "–" not in row, row
        assert "—" in table[1]

    def test_the_within_build_determinism_check_is_reported_from_the_payload(self):
        assert "confirmed two sweeps over one build agree" in vs.render_ann_report(_ann_payload())
        failed = vs.render_ann_report(_ann_payload(within_build_identical=False))
        assert "found them disagreeing" in failed

    def test_an_unrun_determinism_check_claims_neither_outcome(self):
        payload = _ann_payload(within_build_identical=None)
        report = vs.render_ann_report(payload)
        assert "agree exactly" not in report
        assert "found them disagreeing" not in report

    def test_the_illustrative_gap_row_is_never_the_default_row(self):
        """The default has no `ef` to name, and an earlier version could have picked it."""
        report = vs.render_ann_report(_ann_payload())
        assert "at `ef=None`" not in report
        assert "at `ef=4`" in report


#: Numeric tokens allowed to appear in the rendered report without coming from the
#: payload. Every entry needs a reason, because this list is the only way a hardcoded
#: figure can hide from `test_no_number_survives_payload_mutation`.
ALLOWED_LITERALS = {
    # The exact baseline is 1 by definition, not by measurement.
    "1.0000",
    # Qdrant's *documented* default `indexing_threshold` in KB, and this corpus's
    # approximate footprint in MB. Facts about Qdrant and about the corpus, not figures
    # produced by this run.
    "20,000",
    "20",
    # The `ef` bounds quoted in the narrative of the first, local-mode attempt ("recall
    # 1.0000 at every ef from 4 to 256"). Historical: it describes a discarded run.
    "4",
    "256",
    # Percentile labels in the column headers, not values.
    "50",
    "95",
    # Ordered-list numbering.
    "1",
    "2",
    "3",
}

_NUMERIC_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _perturb(obj, counter):
    """Replace every leaf with a distinct sentinel, preserving types."""
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int):
        return 700000 + next(counter)
    if isinstance(obj, float):
        return round(0.100001 * next(counter), 6)
    if isinstance(obj, str):
        return f"vX{next(counter)}"
    if isinstance(obj, list):
        return [_perturb(v, counter) for v in obj]
    if isinstance(obj, dict):
        return {k: _perturb(v, counter) for k, v in obj.items()}
    return obj


class TestNoHardcodedFigures:
    """A mutation guard on the renderer, replacing a regex that could not do the job.

    The previous guard collected `\\d+\\.\\d{4}` from the output and required each match to
    appear in the payload. Three holes, all of which the project's audit gate found:

    1. The regex cannot see a comma-separated integer (`13,423`), a percentage (`66.7%`),
       or a two-decimal millisecond figure (`3.39`) — and the historical defect it cited
       was *"0.9567 while only 66.7%"*, whose second half it would still miss. Two live
       literals (`13,423 vectors x 384 dims`, `well past 13k`) were sitting in the
       renderer while the test passed and the progress log claimed the test had caught
       exactly that.
    2. The allowed set was built from the *same* payload used to render, so a literal
       matching the current run passed **by construction** and only became detectable on
       the next run — precisely when it does damage.
    3. Requiring every figure to appear verbatim in the payload cannot work anyway:
       percentages, ratios and differences in the prose are legitimately *derived*.

    Mutation sidesteps all three. Render once from the real payload and once from a
    payload whose every leaf is a distinct sentinel; any numeric token appearing
    identically in both outputs is, by construction, not a function of the payload. No
    model of how figures are derived is needed, and it fails on the current run.
    """

    def _shared_tokens(self):
        real = _ann_payload()
        mutated = _perturb(_ann_payload(), itertools.count(1))
        a = set(_NUMERIC_TOKEN.findall(vs.render_ann_report(real)))
        b = set(_NUMERIC_TOKEN.findall(vs.render_ann_report(mutated)))
        return a & b

    def test_no_number_survives_payload_mutation(self):
        leaked = self._shared_tokens() - ALLOWED_LITERALS
        assert not leaked, (
            f"hardcoded in the renderer, not derived from the payload: {sorted(leaked)}"
        )

    def test_the_guard_catches_each_kind_the_old_regex_missed(self):
        """The four token shapes that made the previous guard ineffective. Each is
        injected into the rendered output and must be reported as leaked."""
        for literal in ("13,423", "66.7%", "3.39", "7.5"):
            token = _NUMERIC_TOKEN.findall(literal)[0]
            assert token not in ALLOWED_LITERALS, literal
            original = vs.render_ann_report

            def patched(payload, _lit=literal, _orig=original):
                return _orig(payload) + f"\n\nThe corpus has {_lit} of them.\n"

            vs.render_ann_report = patched
            try:
                leaked = self._shared_tokens() - ALLOWED_LITERALS
                assert token in leaked, f"guard missed {literal!r}"
            finally:
                vs.render_ann_report = original

    def test_the_allowlist_is_not_a_blanket(self):
        """An allowlist that covered a measured figure would silently re-open the hole."""
        payload = _ann_payload()
        measured = {
            f"{payload['default']['recall_at_k']:.4f}",
            f"{payload['default']['exact_match_rate']:.4f}",
            f"{payload['exact_p50_ms']:.2f}",
            f"{payload['n_vectors']:,}",
        }
        assert not (measured & ALLOWED_LITERALS)

    def test_every_allowed_literal_is_actually_needed(self):
        """A stale allowlist entry is a hole waiting for a coincidence. If a literal is
        no longer emitted, it must be deleted from the list rather than left behind."""
        unused = ALLOWED_LITERALS - self._shared_tokens()
        assert not unused, f"allowlist entries no longer emitted; delete them: {sorted(unused)}"


class TestPerBuildEvidence:
    """The payload must retain the evidence, not only the conclusion.

    Phase 6 shipped a faithfulness figure whose judge verdicts were never persisted, so
    the headline was not re-derivable. `aggregate_builds` did the same thing: it received
    five values per cell and wrote down three summary statistics.
    """

    @staticmethod
    def _rows(values):
        builds = [
            [
                vs.RecallPoint(
                    ef=128,
                    k=10,
                    n_queries=60,
                    recall_at_k=v,
                    exact_match_rate=v,
                    p50_ms=3.0 + i,
                    p95_ms=5.0 + i,
                )
            ]
            for i, v in enumerate(values)
        ]
        return vs.aggregate_builds(builds)

    def test_the_published_median_is_re_derivable_from_the_payload(self):
        (row,) = self._rows([0.98, 1.0, 0.99])
        assert row["recall_by_build"] == [0.98, 1.0, 0.99]
        assert row["recall_at_k"] == round(statistics.median(row["recall_by_build"]), 4)
        assert row["exact_match_rate"] == round(statistics.median(row["identical_by_build"]), 4)

    def test_the_range_is_re_derivable_too(self):
        (row,) = self._rows([0.98, 1.0, 0.99])
        assert row["recall_min"] == min(row["recall_by_build"])
        assert row["recall_max"] == max(row["recall_by_build"])

    def test_latency_evidence_is_kept_and_matches_its_aggregates(self):
        (row,) = self._rows([0.98, 0.99, 1.0])
        assert row["p50_by_build"] == [3.0, 4.0, 5.0]
        assert row["p50_ms"] == round(statistics.median(row["p50_by_build"]), 3)
        # p95 is the *worst* build, not the median. Asserted so the report's prose and
        # this convention cannot drift apart.
        assert row["p95_ms"] == max(row["p95_by_build"])

    def test_one_value_per_build_is_kept(self):
        (row,) = self._rows([0.98, 0.99, 1.0, 0.97, 0.96])
        assert len(row["recall_by_build"]) == row["n_builds"] == 5

    def test_an_even_build_count_reports_a_value_no_build_produced(self):
        """Documented, not worked around: `statistics.median` interpolates. The docstring
        used to disparage means while the code computed one at every even n."""
        (row,) = self._rows([0.98, 0.99])
        assert row["recall_at_k"] == 0.985
        assert row["recall_at_k"] not in row["recall_by_build"]
