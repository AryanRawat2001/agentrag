"""Sweep-report tests.

The sweep exists to make a decision, so its report must not mislead. The specific
hazard it defends against is the one the sweep itself uncovered: `recall@10` rises
with chunk size for mechanical reasons (fewer chunks per section shrinks the
denominator), so a reader who takes the recall column at face value concludes
"bigger is better" and ships the wrong default.
"""

from __future__ import annotations

import pytest

from ragpipe import eval_report


def row(size: int, retriever: str, slice_name: str, **metrics) -> dict:
    base = {"hit@1": 0.0, "hit@10": 0.0, "recall@10": 0.0, "precision@10": 0.0}
    base.update(metrics)
    return {
        "target_chars": size,
        "overlap_chars": int(size * 0.15),
        "strategy": "structural",
        "retriever": retriever,
        "slice_name": slice_name,
        "metrics": base,
        "separability": {},
    }


def payload(rows: list[dict], shape: dict | None = None) -> dict:
    sizes = sorted({r["target_chars"] for r in rows})
    return {
        "generated_at": "2026-08-17T00:00:00+00:00",
        "strategy": "structural",
        "embed_model": "bge-small",
        "overlap_ratio": 0.15,
        "shape": shape
        or {
            str(s): {
                "n_chunks": 1000 * (5 - i),
                "median_chars": s // 2,
                "index_tokens": 3_000_000,
                "pct_truncated": [0.0, 0.6, 10.4, 31.1][i],
            }
            for i, s in enumerate(sizes)
        },
        "rows": rows,
    }


def test_renders_every_size_as_a_column():
    rows = [row(s, "bm25", "section_lookup", **{"recall@10": 0.9}) for s in (512, 1024, 2048)]
    out = eval_report.render_sweep(payload(rows))
    for s in ("512", "1,024", "2,048"):
        assert s in out


def test_truncation_is_surfaced_as_a_function_of_size():
    """The interaction most easily missed: the embedding window is a fixed budget, so
    the share of chunks it clips is decided by chunk size, not by the model."""
    rows = [row(s, "bm25", "section_lookup", **{"recall@10": 0.9}) for s in (512, 4096)]
    shape = {
        "512": {
            "n_chunks": 22365,
            "median_chars": 564,
            "index_tokens": 3_349_453,
            "pct_truncated": 0.0,
        },
        "4096": {
            "n_chunks": 6624,
            "median_chars": 1225,
            "index_tokens": 3_067_335,
            "pct_truncated": 31.1,
        },
    }
    out = eval_report.render_sweep(payload(rows, shape))
    assert "truncat" in out.lower()
    assert "0.0%" in out and "31.1%" in out


def test_names_the_precision_versus_fragmentation_tradeoff():
    """A reader who does not know why the numbers move cannot use the table."""
    rows = [row(s, "bm25", "section_lookup", **{"recall@10": 0.9}) for s in (512, 2048)]
    out = eval_report.render_sweep(payload(rows)).lower()
    assert "fragment" in out
    assert "precise" in out or "precision" in out


def test_best_size_is_identified_per_slice_not_globally():
    """The sweep's finding is that the optimum differs by slice. A single global
    winner would erase exactly that."""
    rows = [
        row(512, "bm25", "exact_identifier", **{"recall@10": 0.95}),
        row(4096, "bm25", "exact_identifier", **{"recall@10": 0.80}),
        row(512, "bm25", "section_lookup", **{"recall@10": 0.70}),
        row(4096, "bm25", "section_lookup", **{"recall@10": 0.93}),
    ]
    out = eval_report.render_sweep(payload(rows))
    ident = out.split("`exact_identifier`")[1].split("###")[0]
    section = out.split("`section_lookup`")[1].split("###")[0]
    assert "512" in ident.split("Best on this slice")[1]
    assert "4,096" in section.split("Best on this slice")[1]


def test_primary_metric_is_stated_per_slice():
    """Each slice names its own metric, and the name is rendered rather than implied.

    `title_lookup` uses `hit@1`: `recall@10` there would read as failure (every chunk of
    the document is relevant, so it is bounded far below 1 by construction), and `hit@10`
    -- the original choice, which this assertion used to pin -- is saturated on this
    corpus because the queries are verbatim titles.
    """
    rows = [
        row(1024, "bm25", "title_lookup", **{"hit@1": 0.95}),
        row(1024, "bm25", "section_lookup", **{"recall@10": 0.90}),
    ]
    out = eval_report.render_sweep(payload(rows))
    assert "primary metric **hit@1**" in out
    assert "primary metric **recall@10**" in out
    assert "primary metric **hit@10**" not in out


def test_unanswerable_is_not_rendered_as_a_retrieval_slice():
    """Retrieval accuracy is undefined on an empty relevant set."""
    rows = [
        row(1024, "bm25", "section_lookup", **{"recall@10": 0.9}),
        row(1024, "bm25", "unanswerable"),
    ]
    out = eval_report.render_sweep(payload(rows))
    assert "`unanswerable` — primary metric" not in out


def test_every_retriever_gets_a_row_even_where_a_size_is_missing():
    """A gap must render as 0.000 rather than shifting the columns, which would
    silently misalign every later size."""
    rows = [
        row(512, "bm25", "section_lookup", **{"recall@10": 0.8}),
        row(1024, "bm25", "section_lookup", **{"recall@10": 0.9}),
        row(512, "dense", "section_lookup", **{"recall@10": 0.7}),
        # dense at 1024 deliberately absent
    ]
    out = eval_report.render_sweep(payload(rows))
    dense_line = next(line for line in out.splitlines() if line.startswith("| `dense`"))
    assert dense_line.count("|") == 4  # retriever + 2 sizes + trailing
    assert "0.000" in dense_line


def test_write_sweep_emits_both_artifacts(tmp_path):
    rows = [row(1024, "bm25", "section_lookup", **{"recall@10": 0.9})]
    md, js = tmp_path / "s.md", tmp_path / "s.json"
    eval_report.write_sweep(payload(rows), md, js)
    assert md.exists() and js.exists()
    assert "Chunk-size sweep" in md.read_text()
    import json as _json

    assert _json.loads(js.read_text())["rows"][0]["target_chars"] == 1024


def test_overlap_is_documented_as_held_constant():
    """If overlap floated with size, the sweep would vary two things and attribute
    the result to one."""
    rows = [row(s, "bm25", "section_lookup", **{"recall@10": 0.9}) for s in (512, 2048)]
    out = eval_report.render_sweep(payload(rows))
    assert "15%" in out
    assert "only variable" in out


class TestPrimaryMetricChoice:
    """The per-slice primary metric is a reporting decision with teeth.

    `title_lookup` shipped with `hit@10`, which is saturated on this corpus: the queries
    are verbatim document titles, so 47 of 48 configurations scored exactly 1.0000 and the
    project's own notes concluded the *slice* was useless and should be hardened or
    dropped. It measures fine on `hit@1` (0.8500-0.9667, 3.0 binomial standard errors at
    n=60). These tests pin the corrected choice and, more importantly, pin the *reason*,
    so a future edit cannot quietly reintroduce a saturated headline.
    """

    def test_title_lookup_is_judged_on_a_rank_sensitive_metric(self):
        from ragpipe.eval_report import PRIMARY_METRIC

        assert PRIMARY_METRIC["title_lookup"] == "hit@1"

    def test_every_slice_has_a_primary_metric(self):
        import json
        from pathlib import Path

        from ragpipe.eval_report import PRIMARY_METRIC

        evalset = Path(__file__).resolve().parents[1] / "corpus" / "evalset.jsonl"
        if not evalset.exists():
            pytest.skip("needs corpus/evalset.jsonl")
        slices = {json.loads(line)["slice_name"] for line in evalset.open() if line.strip()}
        missing = slices - set(PRIMARY_METRIC)
        assert not missing, f"slices with no stated primary metric: {sorted(missing)}"

    def test_no_slices_primary_metric_is_saturated_in_the_shipped_report(self):
        """Calls `eval_report.is_saturated` — the same function the renderer calls.

        Both the guard and the report used to carry their own copy of this rule, and they
        disagreed: the test said `modal_share > 0.9` while the renderer said
        `len(values) == 1`. Correcting one left the other stale.
        """
        import json
        from pathlib import Path

        from ragpipe.eval_report import PRIMARY_METRIC, is_saturated

        report = Path(__file__).resolve().parents[1] / "reports" / "retrieval_eval.json"
        if not report.exists():
            pytest.skip("needs reports/retrieval_eval.json")
        runs = json.loads(report.read_text())["runs"]
        offenders = []
        for slice_name in {r["slice_name"] for r in runs}:
            metric = PRIMARY_METRIC.get(slice_name)
            if metric is None or metric == "separability":
                continue
            values = [r["metrics"][metric] for r in runs if r["slice_name"] == slice_name]
            if is_saturated(values):
                offenders.append((slice_name, metric))
        assert not offenders, (
            f"primary metric cannot support the orderings the report prints: {offenders}"
        )

    def test_the_saturation_criterion_rejects_the_metric_it_exists_to_reject(self):
        """Break-test, calling the real function.

        The previous version defined a local `fires()` duplicating the criterion, so
        weakening the real one left it green — Phase 8c's "a check that recomputes a rule
        the module already implements will not notice when the rule is corrected",
        re-introduced inside the test written to prevent it.
        """
        import json
        from pathlib import Path

        from ragpipe.eval_report import is_saturated

        report = Path(__file__).resolve().parents[1] / "reports" / "retrieval_eval.json"
        if not report.exists():
            pytest.skip("needs reports/retrieval_eval.json")
        runs = [
            r for r in json.loads(report.read_text())["runs"] if r["slice_name"] == "title_lookup"
        ]
        assert is_saturated([r["metrics"]["hit@10"] for r in runs]), (
            "the criterion no longer rejects hit@10, the metric Phase 8a replaced"
        )
        assert not is_saturated([r["metrics"]["hit@1"] for r in runs]), (
            "the criterion rejects hit@1, which discriminates over 8 distinct values"
        )

    def test_both_renderers_warn_when_a_slice_is_saturated(self):
        """The reader-facing half, and there are two readers.

        The ablation banner carried its own stale rule (`len(values) == 1`) and printed a
        48-row ordering under a metric the module's docstring calls saturated.
        `render_sweep` had no banner at all, despite printing the same kind of per-slice
        ordering over chunk sizes.
        """
        rows = [
            row(1024, f"r{i}", "title_lookup", **{"hit@1": 1.0 if i else 0.98}) for i in range(12)
        ]
        out = eval_report.render_sweep(payload(rows))
        assert "Saturated:" in out
        assert "no ordering" in out.lower()

    def test_the_renderer_stays_quiet_when_a_slice_discriminates(self):
        rows = [row(1024, f"r{i}", "title_lookup", **{"hit@1": 0.85 + i / 100}) for i in range(12)]
        assert "Saturated:" not in eval_report.render_sweep(payload(rows))
