"""Qdrant-backed serving path: approximate dense search, sparse vectors, and fusion.

The locked decision names Qdrant for named dense *and* sparse vectors with server-side
fusion, so hybrid lives in one store rather than a hand-joined second index. This module
is where that decision finally gets exercised — and, more to the point, measured.

## Sparse vectors: true of the code as of Phase 9

For most of this project's life the collection *declared* `sparse_vectors_config` and
nothing was ever written to it, so "hybrid lives in one store" described the plan and not
the repo. `from_dense(..., sparse=bm25)` now writes named sparse vectors alongside the
dense ones, and `QdrantRetriever(mode="hybrid")` fuses them **server-side** with RRF.

The load-bearing property is not that sparse search returns something. It is that the
vectors reproduce *this repo's own* `BM25Retriever`, weight for weight. The values are
read out of `bm25s`' own index rather than recomputed from k1/b, and the query vector
carries term **counts**, so a dot product equals the score `retrieve()` would give. On
3,000 real chunks the top-10 is identical to `BM25Retriever` for every query tried, with
a maximum score delta of 0.0 — checked, because a differently-tokenised sparse index
would show up in the ablation as a Qdrant effect when it was really a retriever effect.

Two failure modes are refused rather than tolerated, both of which return plausible
empty rankings instead of errors: querying a collection whose sparse vectors were never
written, and loading sparse weights whose corpus order differs from the dense side.

RRF is offered because it is measurable, not because it is better. Phase 3 found naive
RRF *worse than sparse alone* on this corpus — fusion rewards agreement, so a retriever
with no signal does not abstain, it votes.

## Why the eval harness kept using exact search

`dense.py` searches exactly: a full matrix product against every chunk vector. Its
docstring explains the reasoning and predicts this module:

    An ANN index has its own recall curve, so an approximate dense row conflates two
    effects — how good the embedding model is, and how much the index gave up to be fast.

Phase 3 therefore measured the ceiling. This module is checked *against* that ceiling, so
**the recall lost to approximation is its own reportable number** rather than being folded
invisibly into every dense and hybrid row of the ablation table.

Two things make that comparison clean:

- **The same vectors.** `from_dense` loads Qdrant from an existing `DenseRetriever`'s
  matrix, which came from the shared content-hash cache. No re-embedding, so nothing can
  drift between the two sides, and the delta is attributable to the index alone.
- **The same interface.** `QdrantRetriever` satisfies the `Retriever` protocol, so it
  drops into the existing eval harness as another row instead of needing a parallel one.

## Recall loss is a curve, not a number

HNSW trades recall for latency through `hnsw_ef` at query time, so reporting a single
configuration's recall would be picking a point off a curve and calling it the answer.
`sweep_recall` walks `ef` and reports recall@k *and* latency at each setting, against the
exact baseline's own result on the same queries. That table is the honest artifact: it
says what approximation costs and what it buys.

`ef` below `k` cannot return `k` good neighbours, so the low end of the curve is expected
to be bad; it is included because a curve with a visible knee is more informative than one
that starts after it.

## Local mode cannot measure approximation, and says so loudly

`QdrantClient(location=":memory:")` and `QdrantClient(path=...)` run in-process with no
server, which is what lets the tests run anywhere. **But local mode does not build or use
HNSW — it brute-forces every query**, and it warns as much:

    UserWarning: Local mode performs exact (brute-force) search, so `search_params` has
    no effect, with the exception of `idf`

The first run of `sweep_recall` against local mode duly reported **index recall of 1.0000
at every `ef` from 4 to 256**, because it was comparing exact search to exact search. Seven
rows of perfect numbers measuring nothing — the precise failure mode this project keeps
logging, and it nearly got written into a report.

So `sweep_recall` now **refuses to run** against a local client rather than returning that
table. Measuring what approximation costs requires a real server:

    docker run -p 6333:6333 qdrant/qdrant

and then `QdrantStore.from_dense(..., url="http://localhost:6333")`. An earlier version of
this docstring claimed local mode "uses the same indexing code as the server", which is
true of storage and false of search.

Latency is a separate caveat that still applies to both modes: in-process search has no
network hop, no serialisation and no concurrent load, so absolute timings are a floor
rather than a forecast, and are only useful for comparing settings against each other.
"""

from __future__ import annotations

import contextlib
import statistics
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ragpipe.retrieval import Hit

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

#: HNSW build parameters. `m` is edges per node, `ef_construct` the build-time candidate
#: list. Qdrant's defaults (16 / 100) are deliberately kept: this phase is measuring what
#: a default configuration costs, because that is what someone deploying it would get.
#: Tuning the build is a separate experiment and would confound the `ef` sweep below.
DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF_CONSTRUCT = 100

#: Force HNSW construction regardless of collection size.
#:
#: Qdrant does not index a small segment by default: `indexing_threshold` is 20,000 KB of
#: vectors, and below it a segment stays in plain brute-force mode because that is genuinely
#: faster. This corpus is 13,423 x 384 x 4 bytes = about 20 MB spread over several segments,
#: so a default collection would answer every query exactly -- and an `hnsw_ef` sweep
#: against it would report recall 1.0000 at every setting.
#:
#: That is the *second* way this measurement can silently succeed at measuring nothing (the
#: first was local mode). Both produce a table of perfect numbers, so `assert_indexed`
#: below checks the index was actually built rather than trusting this setting to have
#: worked.
FORCE_INDEXING_THRESHOLD_KB = 1

#: Query-time candidate list sizes to sweep. Spans below `k` (where recall must suffer)
#: through well above it (where the curve should flatten onto the exact baseline).
DEFAULT_EF_SWEEP = (4, 8, 16, 32, 64, 128, 256)


@dataclass
class RecallPoint:
    """Recall and latency at one `hnsw_ef` setting, against the exact baseline."""

    #: `None` means Qdrant's own default search settings -- the row the headline claim
    #: rests on. It has to be a distinct value rather than a guessed integer, because
    #: reusing whichever swept row happens to match would be asserting what that default
    #: is instead of measuring what an untuned user actually gets.
    ef: int | None
    k: int
    n_queries: int
    #: Mean share of the exact top-k that the approximate search also returned. This is
    #: *index* recall against the exact result, not retrieval recall against ground
    #: truth -- it isolates what approximation costs and says nothing about whether the
    #: embedding model is any good.
    recall_at_k: float
    #: Share of queries where the approximate and exact top-k agree exactly, in content.
    exact_match_rate: float
    p50_ms: float
    p95_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "ef": self.ef,
            "k": self.k,
            "n_queries": self.n_queries,
            "recall_at_k": round(self.recall_at_k, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
        }


def aggregate_builds(points_by_build: Sequence[Sequence[RecallPoint]]) -> list[dict[str, Any]]:
    """Collapse the same sweep run over several independent index builds into one row per
    `ef`, carrying the spread rather than a single build's number.

    This exists because of a specific failure. The first version of this measurement ran
    one build and reported "at Qdrant's default settings approximation costs nothing:
    recall@k and top-k agreement are both 1.0000". Re-running it from a shipped command
    gave 0.9883 and 0.9000 on the same 60 queries. A controlled follow-up separated the
    two candidate causes: **two sweeps over one built index are bit-identical**, and
    across builds `ef=128` recall ranged 0.9900-1.0000 with top-k agreement 0.9000-1.0000.
    HNSW graph construction is randomised, so a single build's recall is a sample, and
    "costs nothing" was that sample landing on its best value.

    The median is reported as the headline and the range alongside it, because the honest
    claim is a band.

    **The per-build values are persisted, not just the aggregate.** Keeping only
    median/min/max means two of five values per cell are unrecoverable and nobody -- not a
    reader, not this project's own audit gate -- can check that the published median is the
    median of anything. That is the same defect as Phase 6 shipping a faithfulness figure
    whose judge verdicts were never written down: the artifact retains the conclusion and
    discards the evidence. `*_by_build` is that evidence, and a test asserts the published
    median equals the median of it.

    On even build counts `statistics.median` interpolates, so the headline is then the mean
    of the two middle builds and is **not an observed value** -- 2 builds of 0.98 and 0.99
    report 0.985. Stated rather than worked around: the interpolated median is the standard
    convention and the persisted array makes the substitution visible. An earlier version
    of this docstring disparaged means while the code computed one at every even `n`.
    """
    if not points_by_build:
        return []
    by_ef: dict[Any, list[RecallPoint]] = {}
    order: list[Any] = []
    for build in points_by_build:
        for point in build:
            if point.ef not in by_ef:
                by_ef[point.ef] = []
                order.append(point.ef)
            by_ef[point.ef].append(point)

    rows: list[dict[str, Any]] = []
    for ef in order:
        points = by_ef[ef]
        recalls = [p.recall_at_k for p in points]
        identicals = [p.exact_match_rate for p in points]
        rows.append(
            {
                "ef": ef,
                "k": points[0].k,
                "n_queries": points[0].n_queries,
                "n_builds": len(points),
                "recall_at_k": round(statistics.median(recalls), 4),
                "recall_min": round(min(recalls), 4),
                "recall_max": round(max(recalls), 4),
                "exact_match_rate": round(statistics.median(identicals), 4),
                "identical_min": round(min(identicals), 4),
                "identical_max": round(max(identicals), 4),
                # The evidence behind every aggregate above, in build order.
                "recall_by_build": [round(v, 4) for v in recalls],
                "identical_by_build": [round(v, 4) for v in identicals],
                "p50_by_build": [round(p.p50_ms, 3) for p in points],
                "p95_by_build": [round(p.p95_ms, 3) for p in points],
                # Latency is a property of the query path, not of the graph, so the
                # median across builds is the summary and the worst p95 is the tail.
                "p50_ms": round(statistics.median([p.p50_ms for p in points]), 3),
                "p95_ms": round(max(p.p95_ms for p in points), 3),
            }
        )
    return rows


def _report_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The default row first, then the sweep. Both come from the payload."""
    rows: list[dict[str, Any]] = []
    if payload.get("default"):
        rows.append(payload["default"])
    rows.extend(payload["sweep"])
    return rows


def _span(row: dict[str, Any], field_name: str) -> str:
    """Render an observed min-max range, or a dash when there is only one build.

    A single build renders as a dash rather than as `0.9883-0.9883`, which would read as
    a measured range that happened to be tight -- the opposite of what one sample means.
    """
    low = row.get(f"{field_name}_min")
    high = row.get(f"{field_name}_max")
    if low is None or high is None or row.get("n_builds", 1) < 2:
        return "—"
    if low == high:
        return f"{low:.4f} (all builds)"
    return f"{low:.4f}–{high:.4f}"


def render_ann_report(payload: dict[str, Any]) -> str:
    """Render the ANN recall/latency artifact. Every figure comes from `payload`."""
    lines: list[str] = []
    a = lines.append
    a("# Serving path — what approximate search costs on this corpus")
    a("")
    n_builds = payload.get("n_builds", 1)
    a(
        f"Qdrant {payload.get('qdrant_version', '?')} over HTTP, HNSW `m={payload['m']}` "
        f"`ef_construct={payload['ef_construct']}`, {payload['n_vectors']:,} vectors x "
        f"{payload['dim']} dims, {payload['n_queries']} queries at k={payload['k']}, "
        f"over **{n_builds} independent index build"
        f"{'s' if n_builds != 1 else ''}**. "
        f"{payload['indexed_vectors']:,} vectors confirmed in the index before measuring."
    )
    a("")
    a(
        "**Recall here is *index* recall against exact search**, not retrieval recall "
        "against ground truth. Phase 3 deliberately shipped exact cosine search so that "
        "this number could be isolated — an approximate dense row in the ablation table "
        "would otherwise conflate how good the embedding model is with how much the index "
        "gave up to be fast."
    )
    a("")
    a(
        "Recall and agreement are the **median across builds**, with the observed range "
        "beside them. `p50` is the median of the per-build p50s; **`p95` is the worst "
        "build's**, not a median, and neither latency column carries a range. Every "
        "per-build value is in `ann_recall.json` under `*_by_build`, so each aggregate "
        "here is re-derivable rather than merely asserted. See the reproducibility "
        "section: a single build's number is a sample."
    )
    a("")
    a("| `hnsw_ef` | index recall@k | range | top-k identical | range | p50 ms | p95 ms |")
    a("|---:|---:|:--|---:|:--|---:|---:|")
    for row in _report_rows(payload):
        label = "**default (unset)**" if row["ef"] is None else str(row["ef"])
        a(
            f"| {label} | {row['recall_at_k']:.4f} | {_span(row, 'recall')} "
            f"| {row['exact_match_rate']:.4f} | {_span(row, 'identical')} "
            f"| {row['p50_ms']:.2f} | {row['p95_ms']:.2f} |"
        )
    a(
        f"| **exact (numpy, in-process)** | 1.0000 | — | 1.0000 | — | "
        f"**{payload['exact_p50_ms']:.2f}** | {payload['exact_p95_ms']:.2f} |"
    )
    a("")
    a("## The headline is a negative result")
    a("")
    # Pick the illustrative row *from the payload*. An earlier version of this renderer
    # hardcoded "at ef=16 recall is still 0.9567 while only 66.7% ..." into the prose, and
    # the next run's table said 0.9367 / 0.5167 — a stale literal inside a report
    # generator, reprinted as current, which is precisely the defect Phase 4's audit found
    # four times. Nothing quantitative in this prose is written down any more.
    gap = max(
        (r for r in payload["sweep"] if r["ef"] is not None and r["recall_at_k"] < 1.0),
        key=lambda r: r["recall_at_k"] - r["exact_match_rate"],
        default=None,
    )
    default = payload.get("default") or {}
    a(
        f"**At Qdrant's default search settings, approximation costs close to nothing on "
        f"recall — median {default.get('recall_at_k', float('nan')):.4f} "
        f"({_span(default, 'recall')}) — but it reorders the top-k more often than that "
        f"suggests: only {default.get('exact_match_rate', float('nan')):.1%} "
        f"({_span(default, 'identical')}) of queries come back with the identical "
        f"top-{payload['k']}.**"
    )
    a("")
    a(
        "An earlier single-build version of this report gave both figures as exactly 1.0000 "
        "and concluded that approximation costs nothing. Repeated rebuilds over the identical "
        "query sample do not reproduce either number, and 1.0000 sits *outside* the range "
        "above rather than at the top of it. Build randomisation (below) accounts for part of "
        "the spread but not for a value the rebuilds never reach, and the script that produced "
        "the original figure was ad-hoc and is not recoverable — which is the reason this "
        "measurement is now a command. The unexplained residual is recorded rather than "
        "explained away; the claim above is the one repeated builds support."
    )
    a("")
    a(
        "Below the default the curve degrades as expected, and the `top-k identical` column "
        "degrades much faster than recall"
        + (
            f" — at `ef={gap['ef']}` recall is still {gap['recall_at_k']:.4f} while only "
            f"{gap['exact_match_rate']:.1%} of queries return the identical top-{payload['k']}"
            if gap
            else ""
        )
        + ". That second column is the one a RAG system should care about, because rank "
        "order is what feeds fusion and reranking, and both are order-sensitive."
    )
    a("")
    default_p50 = default.get("p50_ms", float("nan"))
    a(
        f"**And the vector database is slower than the thing it replaces.** Exact search is "
        f"{payload['exact_p50_ms']:.2f} ms p50 as an in-process numpy matrix product; Qdrant "
        f"at its default is ~{default_p50:.2f} ms. But the gap is *not* HNSW being slow "
        f"— a bare HTTP round trip to the same container, doing no search at all, "
        f"measures **{payload['bare_http_p50_ms']:.2f} ms p50**. So the decomposition is "
        f"roughly {payload['bare_http_p50_ms']:.2f} ms of transport and "
        f"~{max(0.0, default_p50 - payload['bare_http_p50_ms']):.2f} ms of "
        f"search, against {payload['exact_p50_ms']:.2f} ms of doing it locally."
    )
    a("")
    a(
        f"The conclusion is about scale, not about Qdrant: **{payload['n_vectors']:,} vectors "
        f"x {payload['dim']} dims is a trivial matrix product**, so there is no "
        "approximation worth making and no index "
        "worth consulting over a network. This is the same shape of finding as Phase 3's "
        '"hybrid always wins is false" — the received best practice is measured and, on '
        "this corpus, it does not pay. It also vindicates the Phase 3 decision to keep the "
        "eval harness on exact search: had the ablation table used ANN, every dense and "
        "hybrid row would have carried an invisible index error for nothing."
    )
    a("")
    a(
        "**Read that decomposition as a bound, not a split.** The transport baseline is a "
        "`GET /` returning Qdrant's version JSON, measured on an idle server before any "
        "collection exists. The timed query is a `POST` carrying a "
        f"{payload['dim']}-float vector and parsing {payload['k']} scored points against a "
        "live index. Request and response serialisation that genuinely belongs to "
        'transport is therefore counted in the "search" remainder — so the transport '
        "figure is a **floor** and the search figure a **ceiling**. The conclusion does "
        "not depend on the split: either way the round trip costs more than the matrix "
        "product it replaces."
    )
    a("")
    a("## What would change the answer")
    a("")
    a(
        "Qdrant earns its place at a size this corpus does not reach. The crossover is "
        f"where the matrix product stops being free — around a million vectors at "
        f"{payload['dim']} dims, well past {payload['n_vectors']:,}. Not extrapolated here, "
        "because a number invented for a corpus that does not exist is worth less than "
        "saying so."
    )
    a("")
    a("## Two ways this measurement silently measured nothing")
    a("")
    a(
        "Both are worth recording because both produced a full table of perfect numbers, "
        "and each looked like a successful run."
    )
    a("")
    a(
        "1. **Qdrant local mode brute-forces every query.** The first sweep, against "
        '`location=":memory:"`, reported index recall **1.0000 at every `ef` from 4 to '
        "256** — it was comparing exact search against exact search. Qdrant warns about "
        "this; the warning was nearly scrolled past. `sweep_recall` now refuses to run "
        "against a local client."
    )
    a(
        "2. **A server will not index a small collection.** `indexing_threshold` defaults "
        "to 20,000 KB, and this corpus is about 20 MB across several segments, so a "
        "default collection would also have answered every query exactly. The store now "
        "forces indexing *and* `assert_indexed` blocks until Qdrant confirms the vector "
        "count, because the setting working is a claim and the count is evidence."
    )
    a("")
    # `r["ef"] is not None` for the same reason line 272 filters: a `None` ef inside
    # `sweep` (rather than in `default`) would make this `max` raise a TypeError.
    swept = [r for r in payload["sweep"] if r["ef"] is not None]
    top = max(swept, key=lambda r: r["ef"]) if swept else None
    a(
        f"One residual caveat, stated rather than resolved: the collection's "
        f"`full_scan_threshold` is {payload.get('full_scan_threshold', '?')} KB, so Qdrant "
        "may still full-scan individual segments below that size, and segment layout is "
        "decided per build. `hnsw_ef` demonstrably changes recall, which proves HNSW is "
        "being consulted"
        + (
            f", and full recall is *not* reached at the default — it takes `ef={top['ef']}` "
            f"to reach {top['recall_at_k']:.4f}"
            if top is not None
            else ""
        )
        + ". A larger corpus, above the threshold in every segment, would remove this as a "
        "variable."
    )
    a("")
    a("## Reproducibility: the variance is in the build, not the search")
    a("")
    a(
        "Worth stating precisely, because the first version of this report got the cause "
        "wrong and drew the wrong boundary around it. It said search order varies run to "
        "run and that therefore only the *low end* of the curve should be read loosely. "
        "Both halves were wrong."
    )
    a("")
    widest = max(
        _report_rows(payload),
        key=lambda r: (
            (r.get("n_builds", 1) > 1) * (r.get("recall_max", 0) - r.get("recall_min", 0))
        ),
        default=None,
    )
    a(
        "A controlled run separated the two candidates: sweep twice over one built index, "
        "then rebuild and sweep again. **Two sweeps over the same index are bit-identical** "
        "at every `ef` — search is deterministic. Rebuilding the same "
        f"{payload['n_vectors']:,} vectors is not"
        + (
            f", and the spread is not confined to low `ef`: the widest recall band in the "
            f"table above is {_span(widest, 'recall')} at "
            + ("the default" if widest["ef"] is None else f"`ef={widest['ef']}`")
            if widest is not None and widest.get("n_builds", 1) > 1
            else ""
        )
        + ". HNSW graph construction is randomised; the search over it is not."
    )
    a("")
    a(
        "So the variance is not noise to be averaged away at the low end — it reaches the "
        'top of the curve, and it is what produced the original "costs nothing" headline. '
        f"Recall and agreement above are medians over {n_builds} independent build"
        f"{'s' if n_builds != 1 else ''} with the range shown, which is the smallest honest "
        "way to report a number that depends on a random graph. (An earlier version of "
        "this sentence claimed *every* column was a median with a range; two of them are "
        "not, and the caveat above now says which.)"
    )
    if payload.get("within_build_identical") is not None:
        a("")
        a(
            "The determinism check is re-run every time this report is generated rather "
            "than remembered: this run "
            + (
                "confirmed two sweeps over one build agree exactly."
                if payload["within_build_identical"]
                else "**found them disagreeing**, which would mean search is a second "
                "source of variance and the range above understates the spread."
            )
        )
    a("")
    a(
        "Latency figures are a floor, not a forecast: single client, no concurrency, "
        "container on the same host. They are good for comparing settings against each "
        "other and for the transport-versus-search decomposition above, and not for "
        "capacity planning."
    )
    a("")
    return "\n".join(lines)


@dataclass
class QdrantStore:
    """A Qdrant collection holding this corpus, with the retrievers that query it."""

    client: Any
    collection: str
    chunk_ids: list[str] = field(default_factory=list)
    dim: int = 0
    #: Whether named sparse vectors were actually written. The collection always declares
    #: `sparse_vectors_config`, so its presence proves nothing -- for most of this
    #: project's life the config was declared and never populated, which is why the
    #: locked decision's "hybrid lives in one store" was not yet true of the code. A
    #: sparse query against an unpopulated collection returns an empty list rather than
    #: an error, so this flag is what lets the retriever refuse instead of silently
    #: reporting that the lexical side found nothing.
    has_sparse: bool = False
    #: point id -> chunk_id. Qdrant point ids must be int or UUID, and chunk ids are
    #: neither, so the mapping is explicit rather than inferred from a payload lookup.
    _ids: dict[str, str] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.chunk_ids)

    @property
    def is_local(self) -> bool:
        """True when the engine is in-process, and therefore brute-forcing every query.

        Checked structurally rather than by remembering how the client was constructed, so
        it stays right if a caller builds the client itself.
        """
        inner = getattr(self.client, "_client", None)
        return type(inner).__name__ == "QdrantLocal"

    @classmethod
    def from_dense(
        cls,
        dense: Any,
        *,
        collection: str = "ragpipe",
        location: str = ":memory:",
        path: str | None = None,
        url: str | None = None,
        m: int = DEFAULT_HNSW_M,
        ef_construct: int = DEFAULT_HNSW_EF_CONSTRUCT,
        batch_size: int = 512,
        sparse: Any | None = None,
    ) -> QdrantStore:
        """Load a collection from an existing `DenseRetriever`'s vectors.

        Deliberately takes the retriever rather than re-embedding. The vectors came from
        the shared content-hash cache, so both sides of the recall comparison see
        bit-identical input and any difference is the index. Re-embedding here would have
        made the delta a sum of two effects, which is the exact confound `dense.py` chose
        exact search to avoid.
        """
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:  # pragma: no cover - exercised in tests/test_packaging.py
            # `qdrant-client` is an optional extra: the served API does not query Qdrant
            # (see this module's report), so it is not a core dependency. Without this,
            # `ragpipe ann` died on a bare ModuleNotFoundError raised here -- after
            # loading chunks, embedding the corpus and checking the server was up -- and
            # nothing in the traceback named the extra to install.
            raise ImportError(
                "the `vectorstore` extra is required for the Qdrant serving path: "
                "`uv sync --extra vectorstore` (or `pip install 'ragpipe[vectorstore]'`)."
            ) from exc

        matrix = np.ascontiguousarray(dense.matrix, dtype=np.float32)
        chunk_ids = list(dense._chunk_ids)
        if matrix.shape[0] != len(chunk_ids):
            raise ValueError(f"vector/id mismatch: {matrix.shape[0]} vectors, {len(chunk_ids)} ids")
        dim = int(matrix.shape[1]) if matrix.size else 0

        # `url` is a real server and the only mode that exercises HNSW; `path`/`location`
        # are in-process and brute-force. See the module docstring.
        if url:
            client = QdrantClient(url=url)
        elif path:
            client = QdrantClient(path=path)
        else:
            client = QdrantClient(location=location)
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection_name=collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=dim or 1,
                    # Vectors are unit-norm from the provider, so cosine and dot agree.
                    # COSINE is declared anyway: it makes the collection self-describing
                    # and survives a future provider that does not normalise.
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
                )
            },
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=FORCE_INDEXING_THRESHOLD_KB
            ),
        )

        # A sparse retriever's corpus order must match the dense one, or every weight
        # lands on the wrong chunk. Same length is not the same order, so this compares
        # the ids themselves and refuses rather than upserting a silently shuffled index.
        sparse_docs: list[tuple[list[int], list[float]]] | None = None
        if sparse is not None:
            sparse_ids = list(getattr(sparse, "chunk_ids", []))
            if sparse_ids != list(chunk_ids):
                raise ValueError(
                    "the sparse retriever's corpus order does not match the dense one, so "
                    "its weights would be upserted against the wrong chunks "
                    f"({len(sparse_ids)} vs {len(chunk_ids)} ids). Build both from the "
                    "same chunk list."
                )
            sparse_docs = sparse.sparse_document_vectors()

        store = cls(
            client=client,
            collection=collection,
            chunk_ids=chunk_ids,
            dim=dim,
            has_sparse=sparse_docs is not None,
        )
        # Point ids are derived from the chunk id, not from enumeration order, so a
        # re-upsert of the same corpus lands on the same points and a partial rebuild
        # cannot silently duplicate a chunk under a second id.
        for start in range(0, len(chunk_ids), batch_size):
            stop = min(start + batch_size, len(chunk_ids))
            points = []
            for i in range(start, stop):
                cid = chunk_ids[i]
                pid = str(uuid.uuid5(uuid.NAMESPACE_URL, cid))
                store._ids[pid] = cid
                vectors: dict[str, Any] = {DENSE_VECTOR: matrix[i].tolist()}
                if sparse_docs is not None:
                    idxs, vals = sparse_docs[i]
                    vectors[SPARSE_VECTOR] = models.SparseVector(indices=idxs, values=vals)
                points.append(models.PointStruct(id=pid, vector=vectors, payload={"chunk_id": cid}))
            if points:
                client.upsert(collection_name=collection, points=points, wait=True)
        return store

    def indexed_vectors(self) -> int:
        """How many vectors Qdrant has actually put in an HNSW index."""
        info = self.client.get_collection(self.collection)
        return int(getattr(info, "indexed_vectors_count", 0) or 0)

    def assert_indexed(self, timeout_s: float = 120.0) -> int:
        """Block until HNSW exists, then return its size. Raise if it never appears.

        Indexing is asynchronous, so a sweep started immediately after upsert can race it
        and measure brute-force search on a collection that is *about* to be indexed. And
        if `indexing_threshold` were ever wrong, the sweep would silently measure exact
        search again. Checking the count closes both, and it is the only evidence that the
        recall numbers mean anything.
        """
        if self.is_local:
            raise RuntimeError("local mode never builds HNSW; see the module docstring")
        deadline = time.monotonic() + timeout_s
        n = self.indexed_vectors()
        while n < len(self) and time.monotonic() < deadline:
            time.sleep(1.0)
            n = self.indexed_vectors()
        if n == 0:
            raise RuntimeError(
                f"Qdrant indexed 0 of {len(self)} vectors after {timeout_s:.0f}s, so an "
                "hnsw_ef sweep would measure brute-force search and report recall 1.0000 "
                "at every setting. Check `indexing_threshold`."
            )
        return n

    def close(self) -> None:
        # Closing a local client must never be the thing that fails a run.
        with contextlib.suppress(Exception):
            self.client.close()


class QdrantRetriever:
    """Approximate dense retrieval through Qdrant, satisfying the `Retriever` protocol.

    Query encoding is delegated to the same `DenseRetriever` the store was built from, so
    the query side is identical too -- including its per-instance query-vector memoisation.
    Only the *search* differs, which is the whole point.
    """

    def __init__(
        self,
        store: QdrantStore,
        dense: Any = None,
        *,
        sparse: Any = None,
        mode: str = "dense",
        ef: int | None = None,
        name: str | None = None,
    ) -> None:
        if mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError(f"mode must be dense|sparse|hybrid, got {mode!r}")
        if mode in {"dense", "hybrid"} and dense is None:
            raise ValueError(f"mode={mode!r} needs a dense retriever to encode the query")
        if mode in {"sparse", "hybrid"}:
            if sparse is None:
                raise ValueError(f"mode={mode!r} needs the BM25 retriever that built the vectors")
            # The collection declares `sparse_vectors_config` whether or not anything was
            # written to it, and Qdrant answers a query against an unpopulated sparse
            # vector with an empty list, not an error. Without this check `mode="sparse"`
            # would return nothing and read as "the corpus has no lexical match".
            if not store.has_sparse:
                raise ValueError(
                    "this collection has no sparse vectors -- rebuild with "
                    "`QdrantStore.from_dense(..., sparse=bm25)`. A sparse query would "
                    "otherwise return an empty ranking that looks like a real miss."
                )
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.mode = mode
        self.ef = ef
        base = {
            "dense": f"qdrant {getattr(dense, 'name', 'dense')}",
            "sparse": f"qdrant-sparse {getattr(sparse, 'name', 'bm25')}",
            "hybrid": "qdrant-hybrid rrf",
        }[mode]
        self.name = name or base + (f" ef={ef}" if ef is not None and mode != "sparse" else "")

    def __len__(self) -> int:
        return len(self.store)

    def _search_with_sparse(self, query: str, k: int = 10) -> list[Hit]:
        """Sparse-only or server-side-fused retrieval.

        `hybrid` is the reason this module exists: Qdrant holds both named vectors and
        does the fusion itself, so the two candidate lists never leave the store. That is
        what the locked decision meant by "hybrid lives in one store", and until sparse
        vectors were actually written it was true of the plan and not of the code.

        The fusion is **RRF**, and Phase 3 measured naive RRF as *worse than sparse alone*
        on this corpus -- fusion rewards agreement, so a retriever with no signal does not
        abstain, it votes. This path is therefore built to be measured, not assumed
        better; `qdrant-hybrid rrf` is another row in the ablation, not a default.
        """
        from qdrant_client import models

        k = max(1, min(k, len(self.store)))
        idxs, vals = self.sparse.sparse_query_vector(query)
        # No in-vocabulary term means no lexical evidence. Returning an empty ranking is
        # the same choice `BM25Retriever.search` makes and for the same reason: a
        # constant-score ranking handed to fusion becomes noise consumed as ranks.
        if not idxs:
            if self.mode == "sparse":
                return []
            sparse_vec = None
        else:
            sparse_vec = models.SparseVector(indices=idxs, values=vals)

        if self.mode == "sparse":
            result = self.store.client.query_points(
                collection_name=self.store.collection,
                query=sparse_vec,
                using=SPARSE_VECTOR,
                limit=k,
                with_payload=True,
            )
        else:
            params = models.SearchParams(hnsw_ef=self.ef) if self.ef is not None else None
            qv = self.dense._query_vector(query)
            # Each arm over-fetches so fusion has something to fuse: taking only k from
            # each means a document ranked k+1 by both can never surface, which defeats
            # the point of combining them.
            prefetch = [
                models.Prefetch(query=qv.tolist(), using=DENSE_VECTOR, limit=k * 4, params=params)
            ]
            if sparse_vec is not None:
                prefetch.append(models.Prefetch(query=sparse_vec, using=SPARSE_VECTOR, limit=k * 4))
            result = self.store.client.query_points(
                collection_name=self.store.collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
                with_payload=True,
            )

        hits: list[Hit] = []
        for rank, point in enumerate(result.points, 1):
            cid = (point.payload or {}).get("chunk_id") or self.store._ids.get(str(point.id))
            if cid is None:
                continue
            hits.append(Hit(chunk_id=cid, score=float(point.score), rank=rank))
        return hits

    def search(self, query: str, k: int = 10) -> list[Hit]:
        from qdrant_client import models

        if not len(self.store):
            return []
        if self.mode in {"sparse", "hybrid"}:
            return self._search_with_sparse(query, k)
        qv = self.dense._query_vector(query)
        k = max(1, min(k, len(self.store)))
        params = models.SearchParams(hnsw_ef=self.ef) if self.ef is not None else None
        result = self.store.client.query_points(
            collection_name=self.store.collection,
            query=qv.tolist(),
            using=DENSE_VECTOR,
            limit=k,
            search_params=params,
            with_payload=True,
        )
        hits: list[Hit] = []
        for rank, point in enumerate(result.points, 1):
            cid = (point.payload or {}).get("chunk_id")
            if cid is None:
                continue
            hits.append(Hit(chunk_id=str(cid), score=float(point.score), rank=rank))
        return hits


def topk_lists(
    store: QdrantStore,
    dense: Any,
    queries: Sequence[str],
    *,
    k: int = 10,
    ef: int | None = None,
) -> list[list[str]]:
    """The ranked chunk ids for each query -- the raw result, not a summary of it.

    Exists so the within-build determinism check can compare what search actually
    returned. The first version of that check compared each `ef`'s mean recall and
    exact-match *count* between two sweeps and reported agreement as "two sweeps over the
    same index are bit-identical at every `ef`". Those two scalars are invariant under
    reordering and under swapping which queries succeeded, so two genuinely different sets
    of results could satisfy it -- and rank order is precisely what this report's own
    headline says matters, because it is what feeds fusion and reranking.
    """
    retriever = QdrantRetriever(store, dense, ef=ef)
    return [[hit.chunk_id for hit in retriever.search(q, k=k)] for q in queries]


def sweep_recall(
    store: QdrantStore,
    dense: Any,
    queries: Sequence[str],
    *,
    k: int = 10,
    efs: Sequence[int | None] = DEFAULT_EF_SWEEP,
) -> list[RecallPoint]:
    """Measure index recall and latency across `hnsw_ef`, against exact search.

    Recall here is **index recall**: the share of the *exact* top-k that approximate
    search also returned. It is not retrieval recall against ground truth, and conflating
    the two would be the same mistake `dense.py` avoided -- this number says what the
    index gave up, nothing about the embedding model.

    Exact results are computed once and reused across `ef` settings, so the comparison is
    against a fixed reference and the sweep costs one exact pass rather than one per point.

    An `ef` of `None` leaves Qdrant's search parameters unset, measuring the default.
    """
    # Both silent-success paths are closed here rather than trusted: local mode never
    # builds HNSW, and a server can leave a small collection unindexed.
    if not store.is_local:
        store.assert_indexed()
    if store.is_local:
        raise RuntimeError(
            "Qdrant local mode performs brute-force search, so an hnsw_ef sweep measures "
            "nothing: the first attempt reported index recall 1.0000 at every ef from 4 "
            "to 256 because it was comparing exact search against exact search. Start a "
            "server (`docker run -p 6333:6333 qdrant/qdrant`) and build the store with "
            "`url=...` to measure what approximation actually costs."
        )
    exact = {q: [h.chunk_id for h in dense.search(q, k=k)] for q in queries}
    points: list[RecallPoint] = []
    for ef in efs:
        retriever = QdrantRetriever(store, dense, ef=ef)
        recalls: list[float] = []
        matches = 0
        latencies: list[float] = []
        for q in queries:
            reference = exact[q]
            # Skipped before timing, not after. Appending the latency first meant a query
            # with no exact reference contributed to the p50/p95 denominator but not to
            # `n_queries`, so the two figures in the same row described different
            # populations. Unreachable on a non-empty corpus, which is exactly how it
            # would have survived to one that is not.
            if not reference:
                continue
            started = time.perf_counter()
            got = [h.chunk_id for h in retriever.search(q, k=k)]
            latencies.append((time.perf_counter() - started) * 1000.0)
            overlap = len(set(got) & set(reference)) / len(reference)
            recalls.append(overlap)
            if got == reference:
                matches += 1
        latencies.sort()
        points.append(
            RecallPoint(
                ef=ef,
                k=k,
                n_queries=len(recalls),
                recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
                exact_match_rate=matches / len(recalls) if recalls else 0.0,
                # Interpolated median, matching `aggregate_builds`, `cmd_ann` and
                # `service.Timings`. Was `sorted[len//2]`, the upper-middle value.
                p50_ms=statistics.median(latencies) if latencies else 0.0,
                # Nearest-rank, consistent with `gen_eval`, and at these sample sizes it
                # is the slowest observation rather than an interpolated percentile.
                p95_ms=latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
                if latencies
                else 0.0,
            )
        )
    return points
