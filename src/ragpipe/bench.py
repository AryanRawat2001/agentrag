"""Per-stage serving latency, as a re-runnable artifact.

Plan deviation #8 asks for per-stage p50/p95 and cost per query. Phase 7 delivered them on
`/stats`, computed over whatever the live service happened to have served, and quoted three
single observations in the README and the progress log:

    retrieve 0.44 ms | generate 6257 ms | verify 0.012 ms

Those were honestly labelled `n=1` after a review, but they lived in **no** `reports/*.json`
— which makes them exactly what this project's own Phase 7 entry names twice as a recurring
defect: *a measurement that runs somewhere the project cannot re-run it is a measurement the
project does not have.* The ANN report had the same problem and lost its headline to it.

This module closes that. `ragpipe bench` writes `reports/serving_bench.{md,json}`.

## The quota problem, and what it forces

The three stages have wildly different costs, so they cannot all be sampled the same way:

- **retrieve** is local, free, and deterministic-ish. Sample it properly: hundreds of real
  queries from the eval set, repeated.
- **verify** is local, free, and fully deterministic. But it needs *citations* to check, and
  producing citations means generating. So it is measured by **replaying the real citations
  already stored** in `reports/generation_eval.json` — the same `{chunk_id, quote}` pairs a
  live answer produces, against the same chunks. No quota, real inputs.
- **generate** is a network call to a third party with a **20-requests-per-day** free tier.
  Measuring it properly here would burn the day's budget to produce a latency table. So by
  default it is **reported from the `latency_s` already stored** for every outcome of the
  Phase 5 run, and labelled as such. `--live-generate N` measures it fresh when a budget
  exists.

The result is a table whose stages carry *different* sample sizes, which is why `n` is a
column rather than a footnote. A single `n` for the whole table would be a lie about at
least two of the rows.

## Conventions, stated because this project has mixed them before

`p50` is an interpolated median; `p95` is nearest-rank. The same pair `service.Timings`,
`aggregate_builds`, and `cmd_ann` use — four call sites that now agree, after a review found
`sorted[n // 2]` (the upper-middle value, not a median at even n) in two of them.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Repetitions of the whole query set for the local stages. Retrieval is sub-millisecond,
#: so a single pass is dominated by whatever else the machine is doing; repeating gives the
#: percentiles something to describe. Cheap: 240 queries x 3 is under a second.
DEFAULT_REPEATS = 3

#: Stages in the order a request visits them. `total` is deliberately absent: this module
#: measures stages in isolation, and summing isolated medians is not a request latency.
#: `/stats` reports real end-to-end totals from served traffic; that is the right source.
STAGES = ("retrieve", "verify", "generate")


def percentiles(samples: list[float]) -> dict[str, Any]:
    """p50/p95/min/max over `samples`, or an all-`None` row when empty.

    `None` rather than `0.0` for an empty sample, for the reason the service tests spell
    out: a zero reads as "instant", which is the most flattering possible lie about a
    stage that did not run.
    """
    if not samples:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "p50_ms": round(statistics.median(ordered), 4),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
        "min_ms": round(ordered[0], 4),
        "max_ms": round(ordered[-1], 4),
    }


@dataclass
class StageResult:
    """One stage's latency distribution, and where its numbers came from."""

    stage: str
    samples: list[float] = field(default_factory=list, repr=False)
    #: "measured" (timed in this run) or "stored" (read from an earlier run's artifact).
    #: Printed in the table, because a stored figure is not evidence this build is fast.
    source: str = "measured"
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "source": self.source,
            "note": self.note,
            **percentiles(self.samples),
        }


def bench_retrieve(
    retriever: Any, queries: list[str], *, k: int = 5, repeats: int = DEFAULT_REPEATS
) -> StageResult:
    """Time the search stage over real queries.

    Warms up with one untimed pass. `bm25s` builds lookup structures lazily and the first
    query of a process pays for them, so an unwarmed first sample is several times the
    steady-state figure and drags a p95 with it.
    """
    for q in queries[: min(len(queries), 20)]:
        retriever.search(q, k=k)

    samples: list[float] = []
    for _ in range(max(1, repeats)):
        for q in queries:
            started = time.perf_counter()
            retriever.search(q, k=k)
            samples.append((time.perf_counter() - started) * 1000.0)
    return StageResult(
        stage="retrieve",
        samples=samples,
        note=f"BM25 over {len(retriever):,} passages, k={k}, {repeats} passes",
    )


def bench_verify(
    citation_sets: list[tuple[list[dict[str, str]], list[dict[str, Any]]]],
    *,
    repeats: int = DEFAULT_REPEATS,
) -> StageResult:
    """Time the verifier by replaying real citations against their real chunks.

    `citation_sets` pairs each answer's claimed citations with the context it was shown.
    These are the genuine `{chunk_id, quote}` pairs a live answer produced, so the work is
    identical to the serving path -- only the source of the input differs, and no quota is
    spent to obtain it.
    """
    from ragpipe import citations as cit

    # Warm up, as `bench_retrieve` does. The first `verify_answer` of a process compiles
    # the module's normalisation regexes, and at this sample size one cold call lands in
    # the p95 and reported a 37x spread on a deterministic function.
    for claimed, chunks in citation_sets[: min(len(citation_sets), 8)]:
        cit.verify_answer(claimed, chunks)

    samples: list[float] = []
    for _ in range(max(1, repeats)):
        for claimed, chunks in citation_sets:
            started = time.perf_counter()
            cit.verify_answer(claimed, chunks)
            samples.append((time.perf_counter() - started) * 1000.0)
    n_cites = sum(len(c) for c, _ in citation_sets)
    return StageResult(
        stage="verify",
        samples=samples,
        note=(f"{len(citation_sets)} real answers, {n_cites} citations replayed, {repeats} passes"),
    )


def verify_cost_by_method(
    citation_sets: list[tuple[list[dict[str, str]], list[dict[str, Any]]]],
    *,
    repeats: int = 200,
) -> dict[str, Any]:
    """Break verification cost down by the verdict it reaches.

    Exists because the verify row's p95 was ~30x its p50 and neither cold-start nor
    citation count explained it. The driver is the *method*: `exact` is a substring
    search, while `normalized` has to build a whitespace-collapsed copy of the chunk and
    an index map back to the original offsets -- O(chunk length) per citation. Measured,
    six `exact` citations cost less than one `normalized` one.

    Worth reporting rather than smoothing away, because it says which way real traffic
    skews: `normalized` is the *common* case in text extracted from PDFs, so a served
    workload sits at the expensive end of this distribution, not the cheap one.
    """
    from ragpipe import citations as cit

    for claimed, chunks in citation_sets:
        cit.verify_answer(claimed, chunks)

    buckets: dict[str, list[float]] = {}
    for claimed, chunks in citation_sets:
        report = cit.verify_answer(claimed, chunks)
        methods = {check.method for check in report.checks}
        # Labelled by the most expensive verdict present: a set containing one
        # `normalized` pays the normalisation cost regardless of how many `exact` hits
        # accompany it, so averaging it into an "exact" bucket would understate both.
        label = "normalized" if cit.NORMALIZED in methods else "exact"
        if not methods <= {cit.EXACT, cit.NORMALIZED}:
            label = "other"
        timings = []
        for _ in range(max(1, repeats)):
            started = time.perf_counter()
            cit.verify_answer(claimed, chunks)
            timings.append((time.perf_counter() - started) * 1000.0)
        buckets.setdefault(label, []).append(statistics.median(timings))

    return {
        label: {
            "n_answers": len(vals),
            "p50_ms": round(statistics.median(vals), 4),
            "max_ms": round(max(vals), 4),
        }
        for label, vals in sorted(buckets.items())
    }


def generate_from_stored(payload: dict[str, Any]) -> StageResult:
    """Read generation latency out of a `generation_eval.json` payload.

    Not measured here, and the table says so. Generation is a third-party network call
    against a 20-per-day free tier; spending that budget to produce a latency row would
    cost the project a day of evaluation for a number it already has on disk.
    """
    samples = [
        float(o["latency_s"]) * 1000.0
        for o in payload.get("outcomes", [])
        if not o.get("error") and o.get("latency_s")
    ]
    return StageResult(
        stage="generate",
        samples=samples,
        source="stored",
        note=(
            f"from reports/generation_eval.json ({payload.get('generator', 'unknown model')}); "
            "not re-measured, because the free tier is 20 requests per day"
        ),
    )


def bench_generate_live(
    queries: list[str],
    chunk_index: dict[str, dict[str, Any]],
    retriever: Any,
    *,
    generator: Any,
    k: int = 5,
    limit: int = 3,
) -> StageResult:
    """Measure generation for real, on a deliberately small sample.

    Off by default. Every call is a quota unit, and a rejected call still spends one, so
    this refuses to be talked into a large `limit` -- see the CLI's own cap.
    """
    from ragpipe import answer as ans
    from ragpipe.generation import GenerationError

    samples: list[float] = []
    errors = 0
    for query in queries[:limit]:
        hits = retriever.search(query, k=k)
        context = [chunk_index[h.chunk_id] for h in hits if h.chunk_id in chunk_index]
        if not context:
            continue
        started = time.perf_counter()
        try:
            ans.answer_query(query, context, generator=generator)
        except GenerationError:
            errors += 1
            continue
        samples.append((time.perf_counter() - started) * 1000.0)
    note = f"{len(samples)} live calls at k={k}"
    if errors:
        note += f", {errors} failed (quota or upstream)"
    return StageResult(stage="generate", samples=samples, note=note)


def build_payload(results: list[StageResult], *, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": meta.get("generated_at"),
        # Named precisely. `percentiles` uses `ordered[min(n-1, int(0.95*n))]`, a *floor*
        # index, which sits one rank above textbook nearest-rank whenever 0.95n is
        # integral -- at n=540 it returns the 514th of 540 rather than the 513th. All five
        # percentile call sites in this project agree on that formula, so the numbers are
        # consistent; it was the *label* that was imprecise.
        "conventions": (
            "p50 = interpolated median; p95 = floor-index order statistic at 0.95n "
            "(one rank above textbook nearest-rank when 0.95n is integral)"
        ),
        "stages": [r.as_dict() for r in results],
        **{k: v for k, v in meta.items() if k != "generated_at"},
    }


def render_report(payload: dict[str, Any]) -> str:
    """Render the serving-latency artifact. Every figure comes from `payload`."""
    lines: list[str] = []
    a = lines.append
    rows = {r["stage"]: r for r in payload["stages"]}

    a("# Serving path — per-stage latency")
    a("")
    a(
        "Plan deviation #8 asks for per-stage p50/p95 and cost per query. This is the "
        "re-runnable half: `make bench` regenerates it, and the figures below are the "
        "artifact rather than a line in a chat log. `/stats` on a running service reports "
        "the same stages over real served traffic, including end-to-end totals."
    )
    a("")
    a("| stage | n | p50 | p95 | min | max | source |")
    a("|---|---:|---:|---:|---:|---:|---|")
    for stage in STAGES:
        row = rows.get(stage)
        if row is None:
            continue
        if row["n"] == 0:
            a(f"| `{stage}` | 0 | — | — | — | — | not run |")
            continue
        a(
            f"| `{stage}` | {row['n']:,} | **{_ms(row['p50_ms'])}** | {_ms(row['p95_ms'])} "
            f"| {_ms(row['min_ms'])} | {_ms(row['max_ms'])} | {row['source']} |"
        )
    a("")
    a("Notes per stage, because the sample sizes are not comparable:")
    a("")
    for stage in STAGES:
        row = rows.get(stage)
        if row and row.get("note"):
            a(f"- **`{stage}`** — {row['note']}")
    a("")

    retrieve, verify, generate = rows.get("retrieve"), rows.get("verify"), rows.get("generate")
    if retrieve and generate and retrieve["n"] and generate["n"]:
        # Orders of magnitude, not a multiplier. The arithmetic ratio here is ~70,000x,
        # and printing that implies a precision the inputs cannot support: the numerator
        # is a p50 of a distribution that spans 12x, so the "real" ratio moves by an
        # order of magnitude depending on which draw you take. Rounding to powers of ten
        # is the honest resolution.
        decades = math.log10(generate["p50_ms"] / retrieve["p50_ms"])
        a("## The shape is the finding")
        a("")
        a(
            f"**Generation costs roughly {decades:.0f} orders of magnitude more than "
            f"retrieval** — {_ms(generate['p50_ms'])} against {_ms(retrieve['p50_ms'])} at "
            "p50. Stated as a power of ten rather than a multiplier on purpose: the "
            "numerator is the median of a distribution that spans more than a factor of "
            "ten (below), so a precise-looking multiple would be false precision."
        )
        a("")
        a(
            "Every local stage in this pipeline is free by comparison, which is why the "
            "stages are timed separately. A single end-to-end number would move only when "
            "the third-party API moved, and would hide the fact that **nothing this "
            "project controls is the bottleneck** — no amount of retrieval or verifier "
            "optimisation would change a served request's latency measurably."
        )
        a("")
    if verify and verify["n"] and retrieve and retrieve["n"]:
        a(
            f'**"The verifier is free" is a measurement, not a claim.** Verification runs in '
            f"{_ms(verify['p50_ms'])} at p50 over {verify['n']:,} replays — the citation "
            "check that is this project's entire point costs less than the search that "
            "feeds it. It is timed rather than assumed because a verification step that "
            "was expensive would be a reason not to ship one, and that deserved a number."
        )
        a("")

    # The generation row's spread is the story, when there is one. Computed here rather
    # than written down: an earlier report in this project hardcoded a ratio into its
    # renderer and reprinted it as current for two runs.
    if generate and generate["n"] >= 5 and generate["min_ms"]:
        spread = generate["max_ms"] / generate["min_ms"]
        if spread >= 3:
            a("## The generation figure is a wide distribution, not a number")
            a("")
            a(
                f"Generation ranged **{_ms(generate['min_ms'])} to {_ms(generate['max_ms'])}** "
                f"across {generate['n']} calls — a {spread:.1f}x spread on the same model, "
                f"with p50 {_ms(generate['p50_ms'])} and p95 {_ms(generate['p95_ms'])}. "
                "Output size explains little of it: the correlation between tokens produced "
                "and time taken is weak, so this is variance in somebody else's "
                "infrastructure rather than variance in the work."
            )
            a("")
            a(
                "Practical consequence: **a single generation latency figure from this "
                "project should not be trusted to two significant figures.** An earlier "
                "live request through the container measured ~6.3 s, which looks like a "
                "contradiction of the p50 above and is simply one draw from this "
                "distribution. Quote the range, or quote p50 *with* p95 — and treat any "
                "single observation as an anecdote."
            )
            a("")

    by_method = payload.get("verify_by_method")
    if by_method and len(by_method) > 1:
        a("## Why the verify row has a wide p95")
        a("")
        a(
            "Not noise, and not cold start. Verification cost is driven by the **verdict "
            "it reaches**, because the two verified methods do different amounts of work: "
            "`exact` is a substring search, while `normalized` has to build a "
            "whitespace-collapsed copy of the chunk plus an index map back to the original "
            "character offsets."
        )
        a("")
        cheapest = min((r["p50_ms"] for r in by_method.values() if r.get("p50_ms")), default=None)
        a(
            "**Different estimator from the stage table above, deliberately.** Those rows "
            "are raw single-shot timings; these are each answer's *median over many "
            "repeats*"
            + (
                f", because an operation costing {_ms(cheapest)} is otherwise measuring "
                "scheduler noise"
                if cheapest
                else ""
            )
            + ". That is why the slowest figure here can sit *below* the stage row's max — "
            "the two describe different populations, and comparing them across tables "
            "would be an error."
        )
        a("")
        a("| answers containing | answers | median over repeats | slowest answer |")
        a("|---|---:|---:|---:|")
        for label in sorted(by_method):
            row = by_method[label]
            a(f"| `{label}` | {row['n_answers']} | {_ms(row['p50_ms'])} | {_ms(row['max_ms'])} |")
        a("")
        exact, norm = by_method.get("exact"), by_method.get("normalized")
        if exact and norm and exact["p50_ms"]:
            a(
                f"An answer whose citations all matched exactly verifies in "
                f"{_ms(exact['p50_ms'])}; one containing a normalised match takes "
                f"{_ms(norm['p50_ms'])} — **{norm['p50_ms'] / exact['p50_ms']:.0f}x** more. "
                "The direction matters more than the size: `normalized` is the *common* "
                "case in text extracted from PDFs, where line breaks land mid-sentence, so "
                "real traffic sits at the expensive end of this range rather than the "
                f"cheap end this sample is weighted toward.{_vs_generation(generate, norm)}"
            )
            a("")

    a("## What these numbers are not")
    a("")
    a(
        "**Not a capacity forecast.** Single process, single client, no concurrency, and "
        "the local stages are measured in-process rather than over HTTP — so they exclude "
        "the transport that `ann_recall.md` measures separately at ~0.8 ms. They are good "
        "for comparing stages against each other and for the ratio above."
    )
    a("")
    if generate and generate.get("source") == "stored":
        a(
            "**The generation row was not measured in this run.** It is read from the "
            "stored Phase 5 outcomes, because the free tier is 20 requests per day and "
            "spending that budget on a latency row would cost a day of evaluation for a "
            "figure already on disk. `ragpipe bench --live-generate N` measures it fresh. "
            "The `source` column carries this so a stored figure is never mistaken for "
            "evidence about the current build."
        )
        a("")
    a(
        "**Cost per query is reported in tokens, not dollars**, in "
        "`reports/generation_eval.json` and on `/stats`. The free tier bills nothing and no "
        "confirmed paid per-token rate for these models was obtained; a currency figure "
        "here would be invented, and an earlier phase of this project was caught doing "
        "exactly that four times."
    )
    a("")
    return "\n".join(lines)


def _vs_generation(generate: dict[str, Any] | None, other: dict[str, Any]) -> str:
    """ " ... N orders of magnitude below generation", or nothing if generation did not run.

    Separate and guarded because the first version divided `generate["p50_ms"]` inline,
    which is `None` for a stage that was skipped -- so rendering a retrieval-only bench
    raised `TypeError` inside the report generator.
    """
    if not generate or not generate.get("p50_ms") or not other.get("p50_ms"):
        return ""
    decades = math.log10(generate["p50_ms"] / other["p50_ms"])
    return f" Even there it is {decades:.0f} orders of magnitude below generation."


def _ms(value: float | None) -> str:
    """Format a millisecond figure at a precision that suits its magnitude.

    A sub-millisecond stage and a multi-second one share this column, and `0.01 ms`
    against `6257 ms` needs different precision to stay readable — printing four decimals
    on both would imply the seconds figure is accurate to a microsecond.
    """
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value / 1000:,.2f} s"
    if value >= 10:
        return f"{value:,.1f} ms"
    if value >= 1:
        return f"{value:.2f} ms"
    return f"{value:.3f} ms"


def load_citation_sets(
    gen_eval: dict[str, Any], chunk_index: dict[str, dict[str, Any]]
) -> list[tuple[list[dict[str, str]], list[dict[str, Any]]]]:
    """Rebuild verifier inputs from stored outcomes.

    Each stored outcome carries its claimed citations; the context is reconstructed from
    the chunk ids those citations name. That is narrower than the k chunks the model
    actually saw, and it is the honest choice for a *timing* replay: `verify_answer`
    searches the cited chunk first and only falls back to the rest of the document, so
    padding the context with chunks the citations never mention would time a code path
    the real request did not take.
    """
    sets: list[tuple[list[dict[str, str]], list[dict[str, Any]]]] = []
    for outcome in gen_eval.get("outcomes", []):
        checks = outcome.get("checks") or []
        if not checks:
            continue
        claimed = [
            {"chunk_id": str(c.get("chunk_id", "")), "quote": str(c.get("quote", ""))}
            for c in checks
        ]
        chunks = [chunk_index[c["chunk_id"]] for c in claimed if c["chunk_id"] in chunk_index]
        if chunks:
            sets.append((claimed, chunks))
    return sets


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
