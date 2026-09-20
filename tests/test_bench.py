"""Tests for the per-stage serving benchmark.

Weighted toward the two ways a latency artifact misleads.

First, **a stage that did not run must not report a number.** Same rule as
`service.Timings`, for the same reason: `0.0` reads as "instant", which is the most
flattering possible lie about a stage that was skipped.

Second, **a stored figure must not be presentable as a fresh measurement.** The generation
row is read from an earlier run's artifact by default, because the free tier is 20 requests
per day; the `source` column is the only thing separating "this build is fast" from "a
build was fast once", so it is asserted rather than trusted.

The renderer gets a mutation guard, like `vectorstore.render_ann_report` — that module
shipped a stale literal in its prose twice, and this one quotes more derived figures than
it does.
"""

from __future__ import annotations

import itertools
import json
import re

import pytest

from ragpipe import bench


class TestPercentiles:
    def test_an_empty_sample_reports_none_not_zero(self):
        row = bench.percentiles([])
        assert row == {"n": 0, "p50_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}

    def test_p50_is_interpolated_and_p95_is_nearest_rank(self):
        """The two conventions this project settled on across four call sites. Mixing
        them silently is the defect; using both knowingly is not."""
        row = bench.percentiles([1.0, 2.0, 3.0, 4.0])
        assert row["p50_ms"] == 2.5  # interpolated: not a member of the sample
        assert row["p95_ms"] == 4.0  # nearest rank
        assert (row["min_ms"], row["max_ms"]) == (1.0, 4.0)

    def test_order_does_not_matter(self):
        assert bench.percentiles([9.0, 1.0, 5.0])["p50_ms"] == 5.0

    def test_a_single_sample_reports_it_without_pretending_to_a_distribution(self):
        row = bench.percentiles([3.0])
        assert row["n"] == 1
        assert row["p50_ms"] == row["p95_ms"] == row["min_ms"] == row["max_ms"] == 3.0


class TestStageResult:
    def test_a_measured_stage_says_measured(self):
        r = bench.StageResult(stage="retrieve", samples=[1.0, 2.0])
        assert r.as_dict()["source"] == "measured"

    def test_a_stored_stage_says_stored(self):
        """The whole point of the column: a stored number is not evidence about this
        build, and nothing else in the table distinguishes the two."""
        payload = {
            "generator": "gemini-3.6-flash",
            "outcomes": [{"latency_s": 2.0, "attempts": 1}, {"latency_s": 4.0, "attempts": 1}],
        }
        row = bench.generate_from_stored(payload).as_dict()
        assert row["source"] == "stored"
        assert row["n"] == 2
        assert row["p50_ms"] == 3000.0
        assert "20 requests per day" in row["note"]

    def test_failed_outcomes_are_excluded_from_stored_latency(self):
        """A 429 that took 0.3s to be refused is not a generation latency sample."""
        payload = {
            "outcomes": [
                {"latency_s": 5.0, "attempts": 1},
                {"latency_s": 0.3, "error": "GenerationError: 429"},
            ]
        }
        row = bench.generate_from_stored(payload).as_dict()
        assert row["n"] == 1
        assert row["p50_ms"] == 5000.0

    def test_an_empty_stored_payload_yields_an_empty_row_not_a_crash(self):
        assert bench.generate_from_stored({}).as_dict()["n"] == 0


class TestCitationSetReconstruction:
    def test_stored_checks_become_verifier_inputs(self):
        gen = {
            "outcomes": [
                {"checks": [{"chunk_id": "c1", "quote": "hello there"}]},
                {"checks": []},
                {"refused": True},
            ]
        }
        index = {"c1": {"chunk_id": "c1", "doc_id": "d", "text": "well hello there friend"}}
        sets = bench.load_citation_sets(gen, index)
        assert len(sets) == 1
        claimed, chunks = sets[0]
        assert claimed == [{"chunk_id": "c1", "quote": "hello there"}]
        assert chunks[0]["chunk_id"] == "c1"

    def test_citations_naming_absent_chunks_are_dropped(self):
        """The chunk file can be regenerated under a different strategy, which changes
        every chunk id. A replay against ids that no longer exist would time the
        not-found path and report it as verification cost."""
        gen = {"outcomes": [{"checks": [{"chunk_id": "gone", "quote": "x" * 40}]}]}
        assert bench.load_citation_sets(gen, {}) == []

    def test_the_replay_is_the_real_verifier(self):
        """Not a stand-in: the timed call must be the function the service calls."""
        from ragpipe import citations as cit

        text = "The sponsor shall submit within 60 days."
        chunk = {"chunk_id": "c1", "doc_id": "d", "text": text}
        claimed = [{"chunk_id": "c1", "quote": text}]
        result = bench.bench_verify([(claimed, [chunk])], repeats=2)
        assert result.as_dict()["n"] == 2
        # And the same input really does verify, so the timing is of the success path.
        assert cit.verify_answer(claimed, [chunk]).n_verified == 1


class TestVerifyCostByMethod:
    def test_exact_and_normalized_are_reported_separately(self):
        """The finding that explained a 30x p95/p50 spread: `normalized` builds a
        whitespace-collapsed copy of the chunk plus an index map, `exact` does not."""
        exact_chunk = {"chunk_id": "e", "doc_id": "d", "text": "The sponsor shall submit a report."}
        norm_chunk = {
            "chunk_id": "n",
            "doc_id": "d",
            "text": "The sponsor shall\n  submit a report within sixty days of the date.",
        }
        sets = [
            ([{"chunk_id": "e", "quote": "The sponsor shall submit a report."}], [exact_chunk]),
            (
                [{"chunk_id": "n", "quote": "The sponsor shall submit a report within sixty"}],
                [norm_chunk],
            ),
        ]
        out = bench.verify_cost_by_method(sets, repeats=5)
        assert set(out) == {"exact", "normalized"}
        for label in out:
            assert out[label]["n_answers"] == 1
            assert out[label]["p50_ms"] > 0

    def test_an_answer_with_any_normalized_citation_counts_as_normalized(self):
        """It pays the normalisation cost regardless of how many exact hits accompany
        it, so averaging it into the exact bucket would understate both."""
        chunk = {
            "chunk_id": "c",
            "doc_id": "d",
            "text": "The sponsor shall\n  submit a report within sixty days of the date.",
        }
        claimed = [
            {"chunk_id": "c", "quote": "submit a report within sixty days of the date."},
            {"chunk_id": "c", "quote": "The sponsor shall submit a report within sixty"},
        ]
        out = bench.verify_cost_by_method([(claimed, [chunk])], repeats=5)
        assert "normalized" in out


def _payload(**over):
    stages = [
        {
            "stage": "retrieve",
            "source": "measured",
            "note": "BM25 over 13,423 passages, k=5, 3 passes",
            "n": 540,
            "p50_ms": 0.1667,
            "p95_ms": 0.2344,
            "min_ms": 0.0861,
            "max_ms": 0.9017,
        },
        {
            "stage": "verify",
            "source": "measured",
            "note": "8 real answers, 22 citations replayed, 3 passes",
            "n": 24,
            "p50_ms": 0.0069,
            "p95_ms": 0.2142,
            "min_ms": 0.003,
            "max_ms": 0.3108,
        },
        {
            "stage": "generate",
            "source": "stored",
            "note": "from reports/generation_eval.json (gemini-3.6-flash); not re-measured",
            "n": 16,
            "p50_ms": 12534.2146,
            "p95_ms": 27105.2335,
            "min_ms": 2160.0,
            "max_ms": 27105.2335,
        },
    ]
    out = {
        "generated_at": "2026-09-17T00:00:00+00:00",
        "conventions": "p50 = interpolated median; p95 = nearest rank",
        "stages": stages,
        "verify_by_method": {
            "exact": {"n_answers": 6, "p50_ms": 0.0049, "max_ms": 0.0125},
            "normalized": {"n_answers": 2, "p50_ms": 0.1352, "max_ms": 0.2032},
        },
        "n_queries": 180,
        "repeats": 3,
        "k": 5,
    }
    out.update(over)
    return out


#: Numbers the report states that are not measurements of this run.
#:
#: `6.3` is the one that matters: it is a *live observation* from a single request through
#: the container, quoted to show that a single draw contradicts the p50. It exists in no
#: payload and cannot be derived, and the prose labels it an anecdote. Allow-listed rather
#: than deleted, because the point it makes -- do not trust one draw -- is the section's
#: whole argument.
ALLOWED_LITERALS = {
    "6.3",  # the live anecdote, explicitly framed as one draw
    "0.8",  # the transport figure cited from ann_recall.md by name
    "8",  # "plan deviation #8"
    "95",  # the p95 column label
    "50",  # the p50 column label
    "20",  # free-tier requests per day, a fact about the provider
    "10",  # "orders of magnitude" / powers of ten in the prose
    "200",  # the repeat count named in the estimator disclosure
}


#: Names that merely contain digits. Stripped before extraction rather than allow-listed,
#: for the reason `test_failures.py` gives: admitting a bare digit to excuse "Phase 5"
#: also excuses a real count of 5.
_NAMES = re.compile(r"\b[Pp]hase \d+\b|\bBM25\b|\bdeviation #\d+\b")


def _numeric_tokens(text):
    return set(re.findall(r"\d[\d,]*(?:\.\d+)?", _NAMES.sub("", text)))


#: Stage keys whose *relative* magnitudes gate a whole section of the report. The
#: "wide distribution" block only renders when `max_ms / min_ms >= 3`, and a naive
#: perturbation gives consecutive values whose ratio is ~1 -- so the section disappeared
#: from the mutated rendering and every literal inside it went uncompared. Phase 8b's
#: headline ("2.16 s to 27.11 s, a 12.6x range") lived in exactly that block.
#: Chosen so no product can render as a figure the real payload also produces -- an
#: earlier set collided with the true generate p50 ("12.53 s") at one counter value and
#: reported it as a hardcoded literal. `test_the_sentinels_do_not_collide_with_real_output`
#: pins that.
_SPREAD_PRESERVING = {"min_ms": 3.0, "p50_ms": 41.0, "p95_ms": 89.0, "max_ms": 97.0}

#: Sentinel latencies are lifted into a band whose *rendered* form cannot coincide with a
#: real one. This is not tuning. At the original base a mutated `retrieve.min_ms` rendered
#: as `12.53 s`, which is exactly real `generate` p50 -- so the guard reported a real
#: measurement as a hardcoded literal. Hunting for multipliers that happen not to collide
#: is a birthday problem: the widened sweep below would always eventually find one. The
#: largest real token in the report is `13,423`, and the renderer switches to seconds at
#: 1000 ms, so a floor of 1e5 ms puts every sentinel latency at three integer digits of
#: seconds where no real figure reaches two. Disjoint by construction, not by luck.
_SENTINEL_BASE = 130001.0

#: Keys the renderer *branches on* rather than prints.
_DISCRIMINATORS = frozenset({"stage", "source"})

#: Per-stage scale, so the *ratios between* stages change even though each stage keeps its
#: internal shape. Uniform multipliers preserved `generate_p50 / retrieve_p50`, so the
#: rendered "5 orders of magnitude" came out identical in both renders and the guard
#: reported it as a hardcoded literal. Spread must be preserved within a stage and
#: destroyed across them.
_STAGE_SCALE = {"retrieve": 1.0, "verify": 130.0, "generate": 7.0}


def _perturb(obj, counter, key=None):
    if isinstance(obj, bool) or obj is None:
        return obj
    if key in _SPREAD_PRESERVING and isinstance(obj, (int, float)):
        # Keep the shape (min < p50 < p95 <= max, ratio well above 3) while changing every
        # value, so magnitude-gated sections still render and can be compared.
        return round(_SPREAD_PRESERVING[key] * (_SENTINEL_BASE + 7.0 * next(counter)), 4)
    if isinstance(obj, int):
        return 700000 + next(counter)
    if isinstance(obj, float):
        return round(0.100001 * next(counter), 6)
    if isinstance(obj, str):
        # Letters only. An earlier version returned f"vX{counter}", and `vX3` contains
        # the numeric token `3`, so the guard flagged its own scaffolding as a leak.
        n = next(counter)
        return "vX" + "".join(chr(ord("a") + int(d)) for d in str(n))
    if isinstance(obj, list):
        return [_perturb(v, counter) for v in obj]
    if isinstance(obj, dict):
        # A stage row carries its own scale, so cross-stage ratios do not survive.
        stage = obj.get("stage")
        if stage in _STAGE_SCALE:
            scale = _STAGE_SCALE[stage]
            return {
                k: (
                    obj[k]
                    if k in _DISCRIMINATORS
                    else (
                        round(
                            _SPREAD_PRESERVING[k] * scale * (_SENTINEL_BASE + 7.0 * next(counter)),
                            4,
                        )
                        if k in _SPREAD_PRESERVING and isinstance(obj[k], (int, float))
                        else _perturb(obj[k], counter, key=k)
                    )
                )
                for k in obj
            }
        # `stage` **and** `source` are preserved. Both are *discriminators*, not data: the
        # renderer branches on them, so perturbing one makes a whole section vanish from
        # the mutated render and silently exempts every figure inside it. That was fixed
        # for the ratio-gated section via `_SPREAD_PRESERVING` and left broken for the
        # string-gated one two functions below -- the same bug, in the same file, missed
        # because the fix targeted a symptom rather than the class.
        return {
            k: (obj[k] if k in _DISCRIMINATORS else _perturb(v, counter, key=k))
            for k, v in obj.items()
        }
    return obj


class TestRenderReport:
    def test_every_stage_appears_with_its_source(self):
        out = bench.render_report(_payload())
        for stage in ("retrieve", "verify", "generate"):
            assert f"`{stage}`" in out
        assert "stored" in out and "measured" in out

    def test_a_stage_that_did_not_run_is_marked_not_run_rather_than_zero(self):
        payload = _payload()
        payload["stages"] = [
            s if s["stage"] != "generate" else {**s, "n": 0, "p50_ms": None, "p95_ms": None}
            for s in payload["stages"]
        ]
        out = bench.render_report(payload)
        assert "not run" in out
        assert "0.0000 ms" not in out

    def test_the_ratio_is_stated_as_orders_of_magnitude_not_a_false_multiple(self):
        """The arithmetic ratio is ~75,000x, and printing that implies a precision the
        inputs cannot support: the numerator is a p50 of a 12x-wide distribution."""
        out = bench.render_report(_payload())
        assert "orders of magnitude" in out
        assert "75,000" not in out and "69,712" not in out

    def test_the_wide_generation_spread_is_called_out(self):
        payload = _payload()
        out = bench.render_report(payload)
        gen = next(s for s in payload["stages"] if s["stage"] == "generate")
        expected = f"{gen['max_ms'] / gen['min_ms']:.1f}x"
        assert "wide distribution" in out
        assert expected in out, f"expected the computed spread {expected} in the prose"

    def test_a_narrow_generation_spread_does_not_claim_a_wide_one(self):
        payload = _payload()
        payload["stages"] = [
            s if s["stage"] != "generate" else {**s, "min_ms": 12000.0, "max_ms": 13000.0}
            for s in payload["stages"]
        ]
        assert "wide distribution" not in bench.render_report(payload)

    def test_the_verify_breakdown_explains_the_p95(self):
        out = bench.render_report(_payload())
        assert "Why the verify row has a wide p95" in out
        assert "`normalized`" in out

    def test_the_stored_caveat_appears_only_when_a_row_is_stored(self):
        out = bench.render_report(_payload())
        assert "was not measured in this run" in out
        live = _payload()
        live["stages"] = [
            s if s["stage"] != "generate" else {**s, "source": "measured"} for s in live["stages"]
        ]
        assert "was not measured in this run" not in bench.render_report(live)

    def test_no_currency_figure_is_invented(self):
        out = bench.render_report(_payload())
        assert not re.search(r"\$\s*\d", out)
        assert "tokens, not dollars" in out

    def test_the_sentinels_do_not_collide_with_real_output(self):
        """A sentinel that happens to render as a real figure is reported as a leak, which
        sends a reader hunting for a hardcoded literal that does not exist."""
        real = set(_numeric_tokens(bench.render_report(_payload())))
        # Wide enough to reach where collisions actually lived. The previous sweep stopped
        # at 40 while the `12.53` collision reappeared at 390, 397 and 404 -- so the
        # earlier fix moved the collision outside the tested window rather than removing
        # it, under a comment claiming no product *can* collide. A guarantee is only ever
        # as wide as its sweep.
        for counter_start in range(1, 1200):
            mutated = bench.render_report(_perturb(_payload(), itertools.count(counter_start)))
            overlap = (set(_numeric_tokens(mutated)) & real) - ALLOWED_LITERALS
            assert not overlap, f"sentinel collision at counter={counter_start}: {overlap}"

    def test_the_mutation_preserves_magnitude_gated_sections(self):
        """Otherwise a whole section vanishes and its literals are never compared."""
        mutated = bench.render_report(_perturb(_payload(), itertools.count(1)))
        assert "wide distribution" in mutated, (
            "the generate-spread section did not render under mutation, so nothing inside "
            "it is being checked"
        )
        assert "Why the verify row has a wide p95" in mutated

    def test_no_number_survives_payload_mutation(self):
        """Mutation guard, same as `render_ann_report`: any numeric token identical in
        both renderings is by construction not derived from the payload.

        The allowlist is the powers of ten the prose uses structurally ("orders of
        magnitude", "factor of ten") plus the transport figure this report cites from
        `ann_recall.md` by name rather than recomputing.
        """
        allowed = ALLOWED_LITERALS
        real = bench.render_report(_payload())
        mutated = bench.render_report(_perturb(_payload(), itertools.count(1)))
        leaked = (_numeric_tokens(real) & _numeric_tokens(mutated)) - allowed
        assert not leaked, f"figures not derived from the payload: {sorted(leaked)}"

    def test_that_mutation_guard_would_catch_a_hardcoded_figure(self):
        original = bench.render_report

        def patched(payload):
            return original(payload) + "\n\nRetrieval takes 0.1667 ms.\n"

        bench.render_report = patched
        try:
            real = bench.render_report(_payload())
            mutated = bench.render_report(_perturb(_payload(), itertools.count(1)))
            leaked = _numeric_tokens(real) & _numeric_tokens(mutated)
            assert "0.1667" in leaked
        finally:
            bench.render_report = original


class TestShippedArtifact:
    """The artifact on disk, checked the way the audit gate would."""

    @staticmethod
    def _payload():
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "reports" / "serving_bench.json"
        if not path.exists():
            pytest.skip("needs reports/serving_bench.json — run `make bench`")
        return json.loads(path.read_text())

    def test_every_stage_row_is_internally_consistent(self):
        for row in self._payload()["stages"]:
            if not row["n"]:
                continue
            assert row["min_ms"] <= row["p50_ms"] <= row["max_ms"], row
            assert row["p50_ms"] <= row["p95_ms"] <= row["max_ms"], row

    def test_the_conventions_are_recorded_in_the_payload(self):
        assert "interpolated median" in self._payload()["conventions"]

    def test_the_markdown_matches_the_json_on_disk(self):
        from pathlib import Path

        payload = self._payload()
        md = Path(__file__).resolve().parents[1] / "reports" / "serving_bench.md"
        if not md.exists():
            pytest.skip("needs reports/serving_bench.md")
        assert md.read_text() == bench.render_report(payload), (
            "serving_bench.md is not what the renderer produces from serving_bench.json; "
            "one was edited by hand or they were generated at different times"
        )

    def test_local_stages_are_orders_of_magnitude_below_generation(self):
        """The headline claim of the report, asserted against the artifact."""
        rows = {r["stage"]: r for r in self._payload()["stages"]}
        if not (rows["generate"]["n"] and rows["retrieve"]["n"]):
            pytest.skip("needs both stages populated")
        assert rows["generate"]["p50_ms"] > 1000 * rows["retrieve"]["p50_ms"]

    def test_the_median_is_re_derivable_where_samples_are_kept(self):
        """`verify_by_method` stores per-answer medians, so its own p50 is checkable."""
        by_method = self._payload().get("verify_by_method") or {}
        for label, row in by_method.items():
            assert row["p50_ms"] <= row["max_ms"], (label, row)
            assert row["n_answers"] >= 1


class TestPublishedFiguresTrackTheArtifact:
    """The README and the case study quote bench figures by hand, and nothing checked them.

    `serving_bench.md` is generated and a test asserts it matches its JSON. `README.md` and
    `docs/case-study.md` are written by hand — and the `retrieve`/`verify` rows are
    wall-clock timings over samples the payload does **not** retain, so they move a few
    percent on every `make bench`. One re-run during a verification pass silently made the
    README's `0.173 ms` wrong, with no test to notice.

    Two responses, both here. The volatile rows are now published as *bounds*, and these
    tests assert the bounds still hold and that the reproducible figures still match
    exactly. A bound that stops holding is a real regression; three decimals that drift are
    not.
    """

    @staticmethod
    def _payload():
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "reports" / "serving_bench.json"
        if not path.exists():
            pytest.skip("needs reports/serving_bench.json — run `make bench`")
        return json.loads(path.read_text())

    @staticmethod
    def _docs():
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        out = {}
        for name in ("README.md", "docs/case-study.md"):
            path = root / name
            if path.exists():
                out[name] = path.read_text(encoding="utf-8")
        if not out:
            pytest.skip("needs README.md or docs/case-study.md")
        return out

    def test_the_published_bounds_on_local_stages_still_hold(self):
        """The bound is the claim, so the bound is what gets checked."""
        stages = {s["stage"]: s for s in self._payload()["stages"]}
        assert stages["retrieve"]["p50_ms"] < 0.2, (
            f"retrieve p50 {stages['retrieve']['p50_ms']:.4f} ms broke the published "
            "'< 0.2 ms' bound — update the README and case study rather than the bound"
        )
        assert stages["retrieve"]["p95_ms"] < 0.4
        assert stages["verify"]["p95_ms"] < 0.4

    #: Millisecond figures a published document may state as a point value. Everything
    #: else must be a bound. This is an **allowlist**, and that direction is the whole
    #: point: the first version blacklisted the three renderings that had already gone
    #: wrong — `0.173`, `0.167`, `0.1726`, `0.1668` — which by construction cannot catch
    #: the next one. An audit defeated it in a single `ragpipe bench` run: the case study
    #: published `0.179 ms` with all 971 tests green. You cannot enumerate future wrong
    #: values; you can enumerate the legitimate ones.
    STABLE_MS = frozenset(
        {
            "0.006",  # verify p50 — survives a re-run at three decimals
        }
    )

    #: The *only* backticked figures exempt from the bound rule, named one by one.
    #:
    #: Replaces `if f"`{value}" in line: continue`, which exempted **any** figure wrapped
    #: in backticks. The intent was to let a note name the value it retracted; the effect
    #: was a one-character bypass of the whole guard -- appending "retrieval p50 is
    #: `0.179 ms`" to the README passed with every test green, which is the same escape
    #: this allowlist replaced a denylist to close. Enumerating the two real citations
    #: costs a line each and cannot be widened by accident.
    RETRACTED_CITATIONS = frozenset(
        {
            # README's own explanation of why these rows are bounds has to be able to
            # name the reading that went stale.
            ("README.md", "0.173"),
            # The case study citing a figure that measured the judging unit, not latency.
            ("docs/case-study.md", "0.136"),
        }
    )

    #: Every file that publishes a latency figure, not just the two the first version
    #: scanned. `architecture.svg` and `demo-script.md` both quoted `0.17 ms` and neither
    #: was checked by anything.
    PUBLISHED = (
        "README.md",
        "docs/case-study.md",
        "docs/demo-script.md",
        "docs/architecture.svg",
    )

    def test_no_document_states_a_volatile_timing_as_a_point_figure(self):
        """Volatile stages must be published as bounds, in every document that mentions one.

        `retrieve` p50 has rendered as 0.157, 0.159, 0.163, 0.167, 0.173 and 0.179 across
        six runs — a 14% spread on a figure that was being printed to three decimals as
        though it were a constant. So any three-decimal millisecond figure is treated as a
        defect unless it is on `STABLE_MS`.

        Two-decimal figures are left alone deliberately: the ANN table's 2.98 / 3.39 /
        0.46 ms come from `ann_recall.json`, which persists per-build arrays, so they are
        reproducible rather than volatile. The distinguishing property is whether the
        payload retained its samples, and published precision is the proxy for it.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        offenders = []
        for name in self.PUBLISHED:
            path = root / name
            if not path.exists():
                continue
            offenders += self.scan(name, path.read_text(encoding="utf-8"))
        assert not offenders, (
            "volatile latency published as a point figure — use a bound:\n  "
            + "\n  ".join(offenders)
        )

    @classmethod
    def scan(cls, name, text):
        """Offending point figures in `text`, attributed to `name`.

        Extracted from the test body so the guard can be exercised on text that is *not*
        a committed file. Both escape hatches this guard has had were found by hand-editing
        the README and re-running -- a check that can only be aimed at four real files can
        only be break-tested destructively, and a destructive break-test does not stay
        run.

        Three or more decimals is what cleanly separates the two cases. `ann_recall.json`
        stores per-build arrays, so its figures (2.98 ms, 0.46 ms) are reproducible and
        quoted at two decimals; `serving_bench` stores no samples for `retrieve`/`verify`,
        and those were the ones being published to three. A rule over *all* millisecond
        figures flagged the reproducible ANN table and the bounds themselves — precision,
        not magnitude, is the signal.
        """
        offenders = []
        for match in re.finditer(r"(\d+\.\d{3,})\s*ms", text):
            value = match.group(1)
            if value in cls.STABLE_MS:
                continue
            line_start = text.rfind("\n", 0, match.start()) + 1
            end = text.find("\n", match.end())
            line = text[line_start:] if end == -1 else text[line_start:end]
            # A retraction citation is exempt only if it is one of the two named above.
            # `f"`{value}" in line` on its own exempted anything backticked.
            if (name, value) in cls.RETRACTED_CITATIONS and f"`{value}" in line:
                continue
            offenders.append(f"{name}: {line.strip()[:88]}")
        return offenders

    def test_a_backticked_figure_is_no_longer_exempt_by_default(self):
        """The escape hatch, pinned closed. `if f"`{value}" in line` exempted anything in
        backticks, so appending "retrieval p50 is `0.179 ms`" to the README passed with
        every test green -- the same escape the allowlist replaced a denylist to close."""
        assert self.scan("README.md", "retrieval p50 is `0.179 ms` at rest")

    def test_a_bare_figure_is_rejected(self):
        assert self.scan("README.md", "retrieval p50 is 0.179 ms at rest")

    def test_the_two_enumerated_retraction_citations_are_exempt(self):
        assert not self.scan("README.md", "an earlier table published `0.173 ms`, which")
        assert not self.scan("docs/case-study.md", "a figure of `0.136 ms` measured the")

    def test_an_exemption_does_not_transfer_between_files(self):
        """`0.136` is a legitimate citation in the case study and a fresh unbounded figure
        in the README. The exemption is keyed on both, so it cannot leak."""
        assert self.scan("README.md", "verify p50 is `0.136 ms` at rest")
        assert self.scan("docs/demo-script.md", "an earlier table published `0.173 ms`")

    def test_two_decimal_figures_are_left_alone(self):
        """The ANN table's figures come from stored per-build arrays and are reproducible."""
        assert not self.scan("README.md", "exact search took 2.98 ms and ANN 0.46 ms")

    def test_a_bound_is_not_flagged_as_a_point_figure(self):
        assert not self.scan("README.md", "retrieval is under 0.2 ms at p50")

    def test_a_figure_on_the_last_line_is_reported_without_truncation(self):
        """`text.find("\\n", ...)` returns -1 at end of file, so `text[start:end]` becomes
        `text[start:-1]` and drops the final character. It does not change *whether* the
        figure is flagged -- only the message, which would read `0.179 m`. That still
        matters: this failure message is the only thing telling an author which figure to
        fix, and a mangled one sends them looking for a value that is not in the file."""
        offenders = self.scan("README.md", "intro\nretrieval p50 is 0.179 ms")
        assert offenders
        assert "0.179 ms" in offenders[0]

    def test_that_allowlist_rejects_a_freshly_measured_reading(self):
        """Break-test. The guard is only worth having if today's true value still fails it,
        because today's true value is tomorrow's stale one."""
        stages = {s["stage"]: s for s in self._payload()["stages"]}
        rendered = f"{stages['retrieve']['p50_ms']:.3f}"
        assert rendered not in self.STABLE_MS, (
            f"{rendered} is on the allowlist, so publishing today's reading would pass — "
            "which is exactly how 0.179 ms got committed"
        )

    def test_the_reproducible_figures_do_match_exactly(self):
        """`generate` is read from 16 stored per-call latencies, so it *is* exact and the
        documents must agree with it to the digit."""
        stages = {s["stage"]: s for s in self._payload()["stages"]}
        expected = f"{stages['generate']['p50_ms'] / 1000:.1f} s"
        for name, text in self._docs().items():
            assert expected in text, f"{name} disagrees with the stored generate p50"

    def test_verify_p50_is_stable_at_the_precision_published(self):
        """0.006 ms survives re-runs at three decimals; that is why it is quoted and
        `retrieve` is not."""
        stages = {s["stage"]: s for s in self._payload()["stages"]}
        rendered = f"{stages['verify']['p50_ms']:.3f} ms"
        assert rendered == "0.006 ms", (
            f"verify p50 now renders {rendered}; it was published as a stable figure, so "
            "either it regressed or it needs bounding like retrieve"
        )
        for name, text in self._docs().items():
            assert rendered in text, f"{name} disagrees with the stored verify p50"


class TestUntestedPaths:
    """Two functions the Phase 8 code-review gate found had no test at all."""

    def test_live_generation_is_capped_and_the_cap_is_enforced_not_promised(self):
        """`--live-generate` spends daily quota, and a rejected request spends one too.
        The help text promises a cap of 5; this asserts the code enforces it."""
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            ["uv", "run", "ragpipe", "bench", "--live-generate", "50"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 1, "an over-cap request must refuse, not proceed"
        assert "capped at 5" in proc.stderr
        # And it must refuse *before* spending anything.
        assert "measuring generation live" not in proc.stderr

    def test_live_generation_stops_at_its_limit(self):
        """The default `limit=3` was never exercised. A fake generator counts calls."""
        from ragpipe.generation import GenerationResult

        class Counting:
            name = "fake"

            def __init__(self):
                self.calls = 0

            def generate(self, prompt, **kw):
                self.calls += 1
                return GenerationResult(
                    text=json.dumps({"answer": "x", "refused": False, "citations": []}),
                    model="fake",
                    latency_s=0.1,
                )

        chunk = {
            "chunk_id": "c1",
            "doc_id": "d",
            "text": "annual report within 60 days",
            "embed_text": "annual report within 60 days",
            "section_heading": "",
            "identifiers": [],
        }

        class Retr:
            name = "r"

            def search(self, q, k=10):
                from ragpipe.retrieval import Hit

                return [Hit(chunk_id="c1", score=1.0, rank=1)]

        gen = Counting()
        result = bench.bench_generate_live(
            ["q1", "q2", "q3", "q4", "q5"], {"c1": chunk}, Retr(), generator=gen, limit=2
        )
        assert gen.calls == 2, f"limit ignored: {gen.calls} calls"
        assert result.as_dict()["n"] == 2
        assert result.as_dict()["source"] == "measured"

    def test_a_generation_failure_is_counted_not_swallowed(self):
        from ragpipe.generation import GenerationError

        class Failing:
            name = "fake"

            def generate(self, prompt, **kw):
                raise GenerationError("429 quota")

        chunk = {
            "chunk_id": "c1",
            "doc_id": "d",
            "text": "text",
            "embed_text": "text",
            "section_heading": "",
            "identifiers": [],
        }

        class Retr:
            name = "r"

            def search(self, q, k=10):
                from ragpipe.retrieval import Hit

                return [Hit(chunk_id="c1", score=1.0, rank=1)]

        out = bench.bench_generate_live(
            ["q1"], {"c1": chunk}, Retr(), generator=Failing(), limit=1
        ).as_dict()
        assert out["n"] == 0
        assert "failed" in out["note"]

    def test_a_missing_artifact_is_recorded_rather_than_crashing(self):
        """`load_artifacts`' missing-artifact branch was untested; the existing test
        injected `missing_artifacts` straight into the renderer."""
        from pathlib import Path

        from ragpipe import failures

        with __import__("tempfile").TemporaryDirectory() as d:
            out = failures.load_artifacts(Path(d), Path(d))
        assert set(out["missing"]) >= {"retrieval", "generation", "verdicts"}
        assert "retrieval" not in out
