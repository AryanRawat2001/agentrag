"""The retrieval evaluation report — the ablation table this project is organised around.

Phase 2 built the ruler and the sparse baseline; Phase 3 added dense retrieval, the
three fusion methods, and cross-encoder reranking; Phase 4 adds contextual retrieval.
The renderer adapts its framing to whichever rows are present, so the artifact cannot
describe rows it does not contain — or promise as future work rows that sit in the
table below the promise.

Each slice has a **primary metric**, stated in the table rather than left implicit,
because using one metric everywhere would misreport at least one slice:

  `exact_identifier`  recall@10  — the relevant set is small and complete retrieval
                                   is the point
  `section_lookup`    recall@10  — likewise
  `title_lookup`      hit@1      — every chunk of a document is relevant, so recall@10
                                   is bounded far below 1 by construction and would
                                   read as failure. `hit@10` was the original choice
                                   and it **saturates**: the queries are verbatim
                                   document titles, so finding the right document
                                   somewhere in the top 10 is trivial and 47 of 48
                                   configurations scored exactly 1.0000. Ranking it
                                   *first* is not trivial — see the note below
  `unanswerable`      separability — retrieval accuracy is undefined on an empty
                                   relevant set; what matters is whether a score
                                   threshold can tell it apart from the rest
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The metric each slice is judged on. Stated rather than implicit, because one metric
#: everywhere would misreport at least one slice -- see the module docstring.
#:
#: `title_lookup` moved from `hit@10` to `hit@1` in Phase 8, and the reasoning is worth
#: keeping because the conclusion that prompted it was wrong. The carried-forward note
#: read "title_lookup is saturated and no longer discriminates -- every one of the 16
#: configurations scores exactly 1.000, so it should be made harder or dropped". Measured
#: against the report rather than re-read:
#:
#:   hit@10   0.9833-1.0000   2 distinct values over 48 runs   <- saturated
#:   hit@1    0.8500-0.9667   8 distinct                       <- 3.0 binomial se, real
#:   mrr@10   0.9218-0.9833  18 distinct
#:   ndcg@10  0.7460-0.7960  39 distinct
#:
#: The *slice* discriminates perfectly well; the *metric* was saturated, at that one
#: depth. Dropping the slice would have thrown away a working measurement because one
#: summary of it was badly chosen. `hit@1` over `mrr@10` because the two rank all 48
#: configurations near-identically (Spearman rho 0.948, same top five) and "was the right
#: document ranked first" is a sentence a reader can check.
#: A primary metric is unusable when one value covers nearly every configuration: it
#: cannot support any ordering a report prints beneath it.
#:
#: `> 0.9` rather than "only one distinct value". `hit@10` on `title_lookup` takes exactly
#: **two** values -- 47 of 48 runs at 1.0000 and one at 0.9833 -- so a
#: strict-inequality-on-two test passed the very metric this rule exists to reject.
MAX_MODAL_SHARE = 0.9


def is_saturated(values: list[float], *, places: int = 4) -> bool:
    """True when `values` cannot separate the configurations they describe.

    One implementation, called by both the report renderer and its test. The renderer
    previously carried its own copy (`len(values) == 1`), so correcting the rule in the
    test left the reader-facing banner stale -- it printed a 48-row ordering under a
    metric the module's own docstring calls saturated. A checker and its subject must not
    each own a version of the rule.
    """
    if len(values) < 2:
        return False
    rounded = [round(v, places) for v in values]
    distinct = set(rounded)
    if len(distinct) < 2:
        return True
    return max(rounded.count(v) for v in distinct) / len(rounded) > MAX_MODAL_SHARE


PRIMARY_METRIC = {
    "exact_identifier": "recall@10",
    "section_lookup": "recall@10",
    "title_lookup": "hit@1",
    "unanswerable": "separability",
}

REPORTED_METRICS = ("hit@1", "hit@10", "recall@10", "precision@10", "mrr@10", "ndcg@10")


def _table(rows: list[tuple[Any, ...]], headers: tuple[str, ...]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        *["| " + " | ".join(str(c) for c in row) + " |" for row in rows],
    ]


def _ceiling_range(rows: list[dict[str, Any]]) -> str:
    """The observed span of rerank recall ceilings, formatted for prose.

    Computed rather than written down: a literal range in this file disagreed with the
    JSON in the same report by ~0.13.
    """
    vals = [r["recall_ceiling"] for r in rows if "recall_ceiling" in r]
    if not vals:
        return "n/a"
    return f"{min(vals):.3f}" if len(vals) == 1 else f"{min(vals):.3f}-{max(vals):.3f}"


def build(
    runs: list[dict[str, Any]],
    evalset_summary: dict[str, Any],
    truncation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "evalset": evalset_summary,
        "runs": runs,
        "primary_metric": PRIMARY_METRIC,
        "truncation": truncation or {},
    }


def render_markdown(payload: dict[str, Any]) -> str:
    runs = payload["runs"]
    es = payload["evalset"]

    # The title tracks what the table actually contains rather than the phase that
    # created the harness. It read "Phase 2" after Phase 3 had filled in dense,
    # fusion, and rerank rows, and the intro still promised those rows as future work
    # while they sat in the table below it.
    has_dense = any("dense" in r["retriever"] or "hybrid" in r["retriever"] for r in runs)
    has_rerank = any("rerank" in r["retriever"] for r in runs)

    lines = [
        "# Retrieval evaluation — the ablation table",
        "",
        f"_Generated {payload['generated_at']}_",
        "",
    ]
    if has_dense:
        lines += [
            "Sparse, dense, three rank-fusion methods across a weight sweep"
            + (", and cross-encoder reranking." if has_rerank else "."),
            "Built on the Phase 2 harness, which established the ruler and the sparse",
            "baseline every later row is measured against.",
            "",
            "Retrieval metrics need only `question -> relevant chunk ids`, so every row here",
            "is free to recompute once embeddings are cached — which is why this table gates",
            "changes instead of being produced once at the end.",
            "",
        ]
    else:
        lines += [
            "The ruler, and the sparse baseline it measures. Later phases add rows for dense",
            "retrieval, rank fusion, reranking, and contextual retrieval; this establishes",
            "what they are measured against.",
            "",
            "Runs on extracted text with no model calls, so it is free and fast enough to",
            "gate every change — which is the point of building it before the retriever.",
            "",
        ]
    lines += [
        "## Evaluation set",
        "",
        "Queries are **programmatic probes with exact ground truth**, derived from",
        "structure already in the corpus, not natural-language questions. They test",
        "retrieval mechanics precisely and cost nothing to run. Natural-language",
        "questions need an LLM to draft and a human to curate; those arrive with the",
        "generation eval in Phase 6 and answer a different question.",
        "",
        *_table(
            [
                (f"`{name}`", count, PRIMARY_METRIC.get(name, "—"), note)
                for name, count, note in es["slices"]
            ],
            ("Slice", "Queries", "Primary metric", "Ground truth"),
        ),
        "",
        f"Total: **{es['n_queries']} queries** ({es['n_answerable']} answerable, "
        f"{es['n_unanswerable']} unanswerable). Seed `{es['seed']}`.",
        "",
    ]

    # Reconcile the eval-set size against what was actually scored. A query whose
    # relevant set resolves empty against the current chunks is dropped by
    # `metrics.aggregate` — correct behaviour, but it made the summary say 60 while
    # every result row said 58, with nothing anywhere to explain the difference.
    unscorable = es.get("n_unscorable", 0)
    if unscorable:
        per_slice = ", ".join(
            f"`{name}` {n}" for name, n in sorted(es.get("n_scored_by_slice", {}).items())
        )
        lines += [
            f"**{unscorable} of these queries are not scorable and are excluded from every",
            f"row below**, so the per-slice `n` in the results tables is {per_slice} rather",
            "than the counts above. They target a document that was quarantined for having a",
            "character-spaced text layer (see `chunking_report.md`), so their ground-truth",
            "spans resolve to no chunk at all. Dropping them is correct — scoring a query",
            "whose answer is not in the index would measure nothing — but the golden set",
            "still lists them and should be regenerated against the post-quarantine corpus.",
            "",
        ]

    if payload.get("truncation"):
        lines += [
            "## Context-window truncation",
            "",
            "Text past a model's context window is silently discarded, and text that was",
            "never embedded can never be retrieved. That makes truncation a *retrieval*",
            "defect that shows up as mediocre metrics rather than as an error, so it is",
            "counted here rather than assumed away.",
            "",
            "Truncation has two independent causes, and this table measures only the",
            "second. **Across** chunk sizes the rate is set by the target — see",
            "`chunk_size_sweep.md`, where it runs 0.0% at 512 characters to 31.1% at 4,096.",
            "**Within** a fixed size, which particular chunks get clipped is set by",
            "tokenization density rather than by chunks running over target: regulatory text",
            "runs ~4.5 chars/token at the median but far lower at the floor, and the floor is",
            "table-of-contents dot leaders, where a run of `....................` costs",
            "roughly one token per character. Collapsing those runs is what this table's",
            "figures are measured after.",
            "",
            "Figures below are **after** normalization (leader and rule runs collapsed),",
            "since that is what the model receives. Normalization is applied to `embed_text`",
            "only, never to `text`, so citation offsets are untouched.",
            "",
            *_table(
                [
                    (
                        f"`{key}`",
                        f"{t['n_texts']:,}",
                        f"{t['n_truncated']:,}",
                        f"**{t['pct_truncated']:.1f}%**",
                        f"{t['pct_tokens_dropped']:.1f}%",
                    )
                    for key, t in sorted(payload["truncation"].items())
                ],
                (
                    "Chunking / model",
                    "Chunks",
                    "Truncated",
                    "% chunks truncated",
                    "% tokens dropped",
                ),
            ),
            "",
        ]

    lines += [
        "## Results",
        "",
        "Each row is one (chunking strategy, retriever) pair. Metrics are means over the",
        "answerable queries in that slice.",
        "",
        "**Comparing `recall@k` across chunking strategies overstates the gap.** Span-based",
        "ground truth removes the chunk-id remapping artifact, but recall still has a",
        "per-strategy denominator: one section resolves to fewer chunks under `structural`",
        "than under `fixed`, so `fixed` must retrieve more chunks to reach the same recall.",
        "The exact ratio moves with the chunk target, so it is not quoted here — the point",
        "is the direction. `hit@k` is denominator-free and is the figure to use for",
        "cross-strategy claims. The same trap applies across chunk *sizes*, and there it is",
        "large enough to invert a conclusion: see `chunk_size_sweep.md`.",
        "",
    ]

    for slice_name in sorted({r["slice_name"] for r in runs if r["slice_name"] != "unanswerable"}):
        subset = [r for r in runs if r["slice_name"] == slice_name]
        primary = PRIMARY_METRIC.get(slice_name, "recall@10")
        observed = [r["metrics"].get(primary, 0.0) for r in subset]
        values = {round(v, 4) for v in observed}
        saturated = is_saturated(observed)
        lines += [
            f"### `{slice_name}` — primary metric **{primary}**",
            "",
        ]
        if saturated:
            modal = max(values, key=lambda v: [round(x, 4) for x in observed].count(v))
            share = [round(x, 4) for x in observed].count(modal) / len(observed)
            lines += [
                f"> **Saturated: {share:.0%} of configurations score {modal:.4f} on",
                f"> `{primary}`.** This slice cannot discriminate between the design",
                "> decisions under test, so no ordering below should be read as evidence.",
                "> Row order here reflects non-primary columns only.",
                "",
            ]
        lines += [
            *_table(
                [
                    (
                        f"`{r['chunking']}`",
                        f"`{r['retriever']}`",
                        int(r["metrics"].get("n_queries", 0)),
                        *[
                            (
                                f"**{r['metrics'].get(m, 0):.3f}**"
                                if m == primary
                                else f"{r['metrics'].get(m, 0):.3f}"
                            )
                            for m in REPORTED_METRICS
                        ],
                        # Blank for non-rerank rows: a first-stage retriever has no
                        # candidate list to be bounded by, so "1.000" there would
                        # imply a constraint that does not apply to it. Blank also
                        # when the slice's primary metric is not recall@10 — the
                        # stored ceiling *is* a recall ceiling, and printing it beside
                        # a hit@k figure invites exactly the comparison it does not
                        # support. See the note below the table.
                        f"{r['recall_ceiling']:.3f}"
                        if "recall_ceiling" in r and primary == "recall@10"
                        else "—",
                    )
                    for r in sorted(subset, key=lambda x: -x["metrics"].get(primary, 0.0))
                ],
                ("Chunking", "Retriever", "n", *REPORTED_METRICS, "recall@10 ceiling"),
            ),
            "",
        ]
        if any("recall_ceiling" in r for r in subset) and primary != "recall@10":
            lines += [
                "The `recall@10 ceiling` column is blank here on purpose. This slice's",
                f"primary metric is **{primary}**, and the ceiling that was measured is a",
                "*recall* ceiling — the fraction of relevant chunks the candidate list",
                "contained. On this slice every chunk of the named document is relevant, so a",
                "50-candidate window covers only a small fraction of them and the recall",
                # Computed, not a literal. A hardcoded range here said 0.606-0.740 while the
                # JSON it points readers to said 0.491-0.592 — prose contradicting its own
                # artifact, and the exact class of defect this note is warning about.
                f"ceiling measured {_ceiling_range(subset)} here, while `{primary}` needs just",
                "one of them and reads near 1.0. Printing the two side by side makes a correct",
                "row look like it beat its own ceiling, which the prose below calls a wiring",
                "bug. The stored figure is in `retrieval_eval.json` for anyone who wants it.",
                "",
            ]
        if any("recall_ceiling" in r for r in subset) and primary == "recall@10":
            lines += [
                "A reranked row can only permute the candidate list its first stage",
                "returned, so `rerank ceiling` is the best recall@10 that list allowed. Read",
                "those rows against the ceiling, not against 1.000 — a row at its ceiling has",
                "a first-stage recall problem, not a reranking one. A row *above* its own",
                "ceiling is a wiring bug.",
                "",
            ]

    unans = [r for r in runs if r["slice_name"] == "unanswerable"]
    if unans:
        lines += [
            "### `unanswerable` — score separability",
            "",
            "Retrieval accuracy is undefined on an empty relevant set. What matters is",
            "whether the retriever's top score distinguishes a query whose answer exists",
            "from one whose answer does not — that is the evidence for whether Phase 5 can",
            "refuse on a threshold instead of guessing one.",
            "",
            "Computed over `exact_identifier` versus `unanswerable` **only**: those two are",
            "phrased identically and differ solely in whether the cited regulation exists,",
            "which is the comparison that means something. An earlier version pooled all",
            "answerable queries, so section headings and document titles — long, high-IDF",
            "strings that score highly for unrelated reasons — made up two thirds of the",
            "pool. That pushed the class balance to 180/60, so the headline accuracy sat",
            "against a 0.750 majority baseline, and it reversed which chunking strategy",
            "looked better. The baseline is now reported alongside, because an accuracy",
            "without one is unreadable.",
            "",
            *_table(
                [
                    (
                        f"`{r['chunking']}`",
                        f"`{r['retriever']}`",
                        f"{r['separability'].get('answerable_mean_top1', 0):.2f}",
                        f"{r['separability'].get('unanswerable_mean_top1', 0):.2f}",
                        f"{r['separability'].get('best_threshold', 0):.2f}",
                        f"**{r['separability'].get('best_accuracy', 0):.3f}**",
                        f"{r['separability'].get('majority_baseline', 0):.3f}",
                        f"{r['separability'].get('gain_over_baseline', 0):+.3f}",
                    )
                    for r in sorted(unans, key=lambda x: -x["separability"].get("best_accuracy", 0))
                ],
                (
                    "Chunking",
                    "Retriever",
                    "Mean top-1 (answerable)",
                    "Mean top-1 (unanswerable)",
                    "Best threshold",
                    "Best accuracy",
                    "Baseline",
                    "Gain",
                ),
            ),
            "",
        ]

    return "\n".join(lines) + "\n"


def write(payload: dict[str, Any], report_path: Path, json_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def render_sweep(payload: dict[str, Any]) -> str:
    """The chunk-size sweep report.

    Separate from the ablation table because it answers a different question. The
    ablation asks "which retriever wins at a fixed chunk size"; the sweep asks
    "does the answer depend on the size" — and if the optimum differs by slice, a
    single global chunk size is a compromise rather than a choice.
    """
    rows = payload["rows"]
    sizes = sorted({r["target_chars"] for r in rows})

    lines = [
        "# Chunk-size sweep",
        "",
        f"_Generated {payload['generated_at']}_",
        "",
        f"`{payload['strategy']}` chunking at {len(sizes)} target sizes, scored with the",
        "same golden set and the same retrievers as the ablation table. Overlap is held",
        # Read from the payload, not hardcoded: `--overlap-ratio` is configurable, and a
        # literal here would let the artifact assert a control it did not apply.
        f"at {payload['overlap_ratio']:.0%} of target throughout, so size is the only variable.",
        "",
        "Chunk size is the parameter most RAG tutorials pick by feel and never revisit.",
        "It trades two effects against each other: smaller chunks are more precise (less",
        "irrelevant text around a match) but fragment sections, so a query answered by a",
        "whole section needs more of them retrieved to score the same recall.",
        "",
        "## Corpus shape by size",
        "",
        *_table(
            [
                (
                    f"**{s:,}**",
                    f"{payload['shape'][str(s)]['n_chunks']:,}",
                    f"{payload['shape'][str(s)]['median_chars']:,.0f}",
                    f"{payload['shape'][str(s)]['index_tokens']:,}",
                    f"{payload['shape'][str(s)]['pct_truncated']:.1f}%",
                )
                for s in sizes
            ],
            (
                "Target chars",
                "Chunks",
                "Median chars",
                # Labelled as an estimate because it is one: chars/4, not a real
                # tokenizer pass. The truncation column beside it *does* use the real
                # tokenizer, so two different token definitions sit in one row. The
                # estimate runs ~14% high on this corpus.
                "Index tokens (est, chars/4)",
                "% truncated at 512 tok (real)",
            ),
        ),
        "",
        "The truncation column is the interaction worth noticing: the embedding model's",
        "512-token window is a fixed budget, so the share of chunks it clips is a",
        "*function of chunk size*. A size chosen purely for retrieval precision can be",
        "silently discarding text before the vector is ever computed.",
        "",
        "## Retrieval by size",
        "",
    ]

    for slice_name in sorted({r["slice_name"] for r in rows if r["slice_name"] != "unanswerable"}):
        primary = PRIMARY_METRIC.get(slice_name, "recall@10")
        subset = [r for r in rows if r["slice_name"] == slice_name]
        retrievers = sorted({r["retriever"] for r in subset})
        best = max(subset, key=lambda r: r["metrics"].get(primary, 0.0))
        lines += [
            f"### `{slice_name}` — primary metric **{primary}**",
            "",
            *(
                [
                    "> **Saturated: this metric cannot separate the sizes below.**",
                    "> No ordering here should be read as evidence.",
                    "",
                ]
                if is_saturated([r["metrics"].get(primary, 0.0) for r in subset])
                else []
            ),
            *_table(
                [
                    (
                        f"`{ret}`",
                        *[
                            (
                                lambda m: (
                                    f"**{m:.3f}**"
                                    if (m and abs(m - best["metrics"].get(primary, 0.0)) < 1e-9)
                                    else f"{m:.3f}"
                                )
                            )(
                                next(
                                    (
                                        r["metrics"].get(primary, 0.0)
                                        for r in subset
                                        if r["retriever"] == ret and r["target_chars"] == s
                                    ),
                                    0.0,
                                )
                            )
                            for s in sizes
                        ],
                    )
                    for ret in retrievers
                ],
                ("Retriever", *[f"{s:,}" for s in sizes]),
            ),
            "",
            f"Best on this slice: **{best['target_chars']:,} chars** with "
            f"`{best['retriever']}` at {best['metrics'].get(primary, 0.0):.3f}.",
            "",
        ]

    return "\n".join(lines) + "\n"


def write_sweep(payload: dict[str, Any], report_path: Path, json_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_sweep(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
