"""FastAPI serving path: retrieve, answer, verify — with per-stage timing and cost.

Plan deviation #8 asks for per-stage p50/p95 latency and cost per query, on the grounds
that "production-grade" claims need them. This module is where they come from, and they are
measured on **served requests** rather than a synthetic benchmark: `/stats` reports the
distribution of what the service has actually done.

## Three stages, timed separately

`retrieve`, `generate`, `verify`. Reporting one end-to-end number would hide the only fact
that matters for tuning: retrieval is milliseconds of local matrix work and generation is
seconds of somebody else's network. A p95 that moved could be either, and the fix differs.

Verification is timed too even though it is deterministic and fast, because it is the
project's headline feature and "the verifier is free" should be a measurement rather than
an assumption.

## Cost is reported in tokens, not dollars

Same decision as `generation.py` and the Phase 5 report: the free tier bills nothing, and
confirmed paid per-token rates for these models were never obtained. A dollar figure
computed from a guessed rate is worse than no dollar figure — Phase 4's audit found four
such invented literals. `price_per_mtok` on the generator registry is `None`, and this
service reports the token counts that a real rate would multiply.

Set `RAGPIPE_PRICE_IN` / `RAGPIPE_PRICE_OUT` (dollars per million tokens) to have the
service compute cost. Unset, `cost_usd` is `null` and the field says why.

## The index is loaded once, at startup

Not per request. Building the BM25 index over 13,423 chunks takes seconds, and doing it
inside a handler would make the latency table a measurement of index construction.
`AppState` holds it.

No embedder is loaded: the served retriever is BM25, which is why the container needs
none of torch. An earlier version of this note cited embedder load time as a reason for
startup loading, which was a reason borrowed from a dense serving path this service does
not have.

`/health` reports whether the load succeeded. Note that in the shipped `ragpipe serve`
path `ready: false` is unreachable, because `load_state` runs to completion before the
port is bound -- the honest statement is that a slow start is visible as a refused
connection, and `/health` is what distinguishes "up and indexed" from "up and empty" for
a state built any other way.

Generation is optional. Without a key the service still serves retrieval and returns a
`generator_available: false` health field, because the retrieval half of this project works
offline and a demo that cannot start without a third-party credential is a worse demo.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ragpipe import answer as ans
from ragpipe import citations as cit

#: Per-stage samples retained for the percentile table. A bounded deque rather than an
#: unbounded list: a long-running service would otherwise grow without limit, and recent
#: latency is what anyone asks about. 1,024 is far more than a demo will serve and small
#: enough to be free.
MAX_SAMPLES = 1024

#: Characters of chunk text returned per retrieved hit. Enough to judge a hit at a
#: glance, bounded so a k=50 request cannot return a megabyte. The full chunk is not the
#: service's job to serve -- `n_chars` and `truncated` say what was withheld.
PREVIEW_CHARS = 900

STAGES = ("retrieve", "generate", "verify", "total")


@dataclass
class Timings:
    """Per-stage latency samples, and the percentiles over them."""

    samples: dict[str, deque[float]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=MAX_SAMPLES))
    )

    def record(self, stage: str, ms: float) -> None:
        # Rejected rather than accepted-and-ignored. `percentiles()` only reports
        # `STAGES`, so a typo'd stage name previously accumulated samples forever that
        # nothing would ever show -- a measurement being taken and silently discarded,
        # which is indistinguishable from a stage that is never slow.
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; known: {STAGES}")
        self.samples[stage].append(ms)

    def percentiles(self) -> dict[str, dict[str, float | int | None]]:
        out: dict[str, dict[str, float | int | None]] = {}
        for stage in STAGES:
            vals = sorted(self.samples.get(stage, ()))
            if not vals:
                out[stage] = {"n": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
                continue
            out[stage] = {
                "n": len(vals),
                # Interpolated median, nearest-rank p95 -- the same two conventions
                # `gen_eval` uses, named here for the same reason: mixing them silently is
                # the defect, using them knowingly is not.
                "p50_ms": round(statistics.median(vals), 3),
                "p95_ms": round(vals[min(len(vals) - 1, int(0.95 * len(vals)))], 3),
                "max_ms": round(vals[-1], 3),
            }
        return out


@dataclass
class Usage:
    """Cumulative token counts, and cost only if real rates were supplied."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    n_generated: int = 0

    def add(self, result: Any) -> None:
        if result is None:
            return
        self.prompt_tokens += result.prompt_tokens
        self.output_tokens += result.output_tokens
        self.thinking_tokens += result.thinking_tokens
        self.n_generated += 1

    def as_dict(self) -> dict[str, Any]:
        price_in = _float_env("RAGPIPE_PRICE_IN")
        price_out = _float_env("RAGPIPE_PRICE_OUT")
        cost = None
        if price_in is not None and price_out is not None:
            # Thinking tokens bill as output. They are counted separately everywhere in
            # this project because they are invisible in the response and were the single
            # largest lever on Phase 5 spend.
            cost = (
                self.prompt_tokens * price_in
                + (self.output_tokens + self.thinking_tokens) * price_out
            ) / 1_000_000
        return {
            "n_generated": self.n_generated,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "tokens_per_query": (
                round(
                    (self.prompt_tokens + self.output_tokens + self.thinking_tokens)
                    / self.n_generated,
                    1,
                )
                if self.n_generated
                else None
            ),
            "cost_usd": None if cost is None else round(cost, 6),
            "cost_usd_per_query": (
                None if cost is None or not self.n_generated else round(cost / self.n_generated, 6)
            ),
            "cost_note": (
                "computed from RAGPIPE_PRICE_IN / RAGPIPE_PRICE_OUT"
                if cost is not None
                else "null by design: the free tier bills nothing and no confirmed paid "
                "rate was obtained. Set RAGPIPE_PRICE_IN and RAGPIPE_PRICE_OUT (dollars "
                "per million tokens) to have this computed."
            ),
        }


def _float_env(name: str) -> float | None:
    """A non-negative, finite price, or None.

    Rejects negatives and non-finite values rather than passing them through. A negative
    rate produced a negative `cost_usd`, and `nan` produced a `cost_usd` of NaN -- which
    is not valid JSON, so `/stats` would emit a body no strict parser accepts. Both are
    nonsense prices; refusing them keeps `cost_usd: null` meaning "no rate supplied".
    """
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value < 0 or value != value or value in (float("inf"), float("-inf")):
        return None
    return value


class QueryRequest(BaseModel):
    """The `/query` request body.

    Module level, not nested inside `create_app`, and that is load-bearing rather than
    stylistic. This module uses `from __future__ import annotations`, so every annotation
    is a string that FastAPI resolves against the *module* globals. A model defined in a
    function body is not in those globals: FastAPI could not resolve `req: QueryRequest`,
    fell back to treating `req` as a scalar query parameter, and returned 422 `Field
    required` for every well-formed body. The route had never accepted a request, and
    nothing caught it because nothing had called it -- the failure is invisible to import,
    to lint, and to type checking, and only a request reveals it.
    """

    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    #: Skip generation and return retrieved chunks only. The retrieval half is free
    #: and local, so it is worth being able to exercise it without spending quota.
    retrieve_only: bool = False


@dataclass
class AppState:
    """Everything loaded once at startup."""

    chunks: list[dict[str, Any]] = field(default_factory=list)
    chunk_index: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Distinct source documents behind the index. Counted once at load rather than per
    #: request, and exposed so a caller describing the corpus does not have to hardcode
    #: it. `bin/build-demo` narrated "a hundred and sixty documents" -- the number
    #: *fetched*, where 152 have a usable text layer and are the only ones answerable.
    n_docs: int = 0
    retriever: Any = None
    generator: Any = None
    generator_error: str = ""
    timings: Timings = field(default_factory=Timings)
    usage: Usage = field(default_factory=Usage)
    chunking: str = "structural"

    @property
    def ready(self) -> bool:
        return self.retriever is not None

    @property
    def generator_available(self) -> bool:
        return self.generator is not None


def load_state(
    chunk_path: Any,
    *,
    chunking: str = "structural",
    heading_mode: str = "prepend",
    generator_spec: str | None = None,
) -> AppState:
    """Build the index and, if a key exists, the generator. Called once.

    A missing generator is not a startup failure: retrieval is fully local, so the service
    degrades to retrieval-only rather than refusing to start. The reason is kept on the
    state and surfaced by `/health`, so "no key" is distinguishable from "broken".
    """
    from ragpipe import retrieval

    state = AppState(chunking=chunking)
    state.chunks = retrieval.load_chunks(chunk_path)
    state.chunk_index = {c["chunk_id"]: c for c in state.chunks}
    state.n_docs = len({c.get("doc_id") for c in state.chunks if c.get("doc_id")})
    state.retriever = retrieval.BM25Retriever(state.chunks, heading_mode=heading_mode)

    if generator_spec:
        from ragpipe.generation import GenerationError, make_generator

        try:
            state.generator = make_generator(generator_spec)
        except (GenerationError, ValueError) as exc:
            state.generator_error = str(exc)
    return state


def create_app(state: AppState) -> Any:
    """Build the FastAPI app around an already-loaded state.

    State is injected rather than loaded inside a startup hook so tests can construct a
    tiny corpus and exercise every route without loading a model or touching the network.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(
        # The user-facing name is `agentrag` -- what you type to run it, and what the
        # dashboard is titled. `ragpipe` remains the Python package and import namespace,
        # so this is the one place the two are deliberately different rather than a typo.
        title="agentrag",
        summary="Hybrid retrieval with citations verified against the source text.",
    )
    # Attached for introspection by anything holding only the app (a test harness, an
    # ASGI middleware). The routes deliberately use the closure instead, so there is one
    # source of truth at request time rather than two that could diverge.
    app.state.ragpipe = state

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> str:
        """The query page. One in-memory string, no build step, no static directory.

        Served from `/` so the container needs no second process and no volume for
        assets: `docker run -p 8000:8000` gives you the API and the page that drives it.
        """
        from ragpipe.dashboard import DASHBOARD_HTML

        return DASHBOARD_HTML

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ready": state.ready,
            "chunks": len(state.chunks),
            "docs": state.n_docs,
            "chunking": state.chunking,
            "retriever": getattr(state.retriever, "name", None),
            "generator_available": state.generator_available,
            "generator": getattr(state.generator, "name", None),
            # Present only when generation is unavailable, and it says *why* -- a missing
            # key and a broken install are different problems.
            "generator_error": state.generator_error or None,
        }

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        return {"latency": state.timings.percentiles(), "usage": state.usage.as_dict()}

    @app.post("/query")
    def query(req: QueryRequest) -> dict[str, Any]:
        if not state.ready:
            raise HTTPException(status_code=503, detail="index not loaded")

        t_total = time.perf_counter()
        t0 = time.perf_counter()
        hits = state.retriever.search(req.query, k=req.k)
        context = [state.chunk_index[h.chunk_id] for h in hits if h.chunk_id in state.chunk_index]
        retrieve_ms = (time.perf_counter() - t0) * 1000.0
        state.timings.record("retrieve", retrieve_ms)

        def _retrieved_row(hit: Any) -> dict[str, Any]:
            chunk = state.chunk_index.get(hit.chunk_id, {})
            # The chunk's own text, truncated. Included because a retrieval demo that
            # shows only doc ids and scores cannot be judged: "was this a good hit?" is
            # a question about the text, and without it the ranking is unfalsifiable.
            # `text`, never `embed_text` -- the latter carries a prepended heading that
            # the model never sees as body content, and showing it would misrepresent
            # what was retrieved.
            body = chunk.get("text") or ""
            return {
                "chunk_id": hit.chunk_id,
                "score": round(hit.score, 4),
                "rank": hit.rank,
                "doc_id": chunk.get("doc_id"),
                "section_heading": chunk.get("section_heading"),
                "text": body[:PREVIEW_CHARS],
                "n_chars": len(body),
                "truncated": len(body) > PREVIEW_CHARS,
            }

        retrieved = [_retrieved_row(h) for h in hits]

        # Nothing to ground on: refuse before generating. Both a zero-hit retrieval and a
        # ranking whose every chunk_id is missing from the index land here.
        #
        # Without this the empty context went to the model, which spent a daily quota unit
        # and *answered anyway* -- one fabricated citation, verified `unverified`. So the
        # bug cost quota and produced ungrounded output, which is the failure this whole
        # project is built to make impossible. `test_retrieve_only_never_calls_the_generator`
        # already guards the same cost on the deliberate path; this is the accidental one.
        if not context:
            total_ms = (time.perf_counter() - t_total) * 1000.0
            state.timings.record("total", total_ms)
            return {
                "query": req.query,
                "answered": False,
                "refused": True,
                "refusal_source": "no_context",
                "reason": "retrieval returned no usable context; refused before generating",
                "retrieved": retrieved,
                "timings_ms": {"retrieve": round(retrieve_ms, 3), "total": round(total_ms, 3)},
            }

        if req.retrieve_only or not state.generator_available:
            total_ms = (time.perf_counter() - t_total) * 1000.0
            state.timings.record("total", total_ms)
            return {
                "query": req.query,
                "answered": False,
                "reason": "retrieve_only" if req.retrieve_only else "no generator configured",
                "retrieved": retrieved,
                "timings_ms": {"retrieve": round(retrieve_ms, 3), "total": round(total_ms, 3)},
            }

        from ragpipe.generation import GenerationError

        t0 = time.perf_counter()
        try:
            graded = ans.answer_query(req.query, context, generator=state.generator)
        except GenerationError as exc:
            # A generation failure is the upstream's problem, not a 500. Retrieval already
            # succeeded and is worth returning, so the client gets the chunks and a reason.
            state.timings.record("generate", (time.perf_counter() - t0) * 1000.0)
            total_ms = (time.perf_counter() - t_total) * 1000.0
            state.timings.record("total", total_ms)
            # `refusal_source` is the structured half. A client that has to tell "come
            # back tomorrow" apart from "retry in a moment" cannot do it from the prose:
            # a per-minute 429's body also says "you exceeded your current quota", so a
            # substring scan gives the two opposite conditions the same answer.
            return {
                "query": req.query,
                "answered": False,
                "refusal_source": (
                    "daily_quota" if getattr(exc, "daily_quota", False) else "generation_error"
                ),
                "reason": f"generation failed: {exc}"[:300],
                "retrieved": retrieved,
                "timings_ms": {"retrieve": round(retrieve_ms, 3), "total": round(total_ms, 3)},
            }
        generate_ms = (time.perf_counter() - t0) * 1000.0
        state.timings.record("generate", generate_ms)
        state.usage.add(graded.usage)

        # Verification already ran inside `answer_query`; timed here as its own stage by
        # re-running the deterministic check over the same inputs. It is microseconds, and
        # "the verifier is free" should be a number rather than a claim.
        t0 = time.perf_counter()
        report = cit.verify_answer(graded.citations, context)
        verify_ms = (time.perf_counter() - t0) * 1000.0
        state.timings.record("verify", verify_ms)

        total_ms = (time.perf_counter() - t_total) * 1000.0
        state.timings.record("total", total_ms)

        # Which retrieved chunks the answer actually leaned on. Known only after
        # verification, so it is stamped on here rather than built above. This is the
        # link between the two halves of the pipeline: retrieval ranked ten things and
        # the answer used two, and that ratio is visible nowhere else.
        cited_ids = {check.chunk_id for check in report.checks}
        for row in retrieved:
            row["cited"] = row["chunk_id"] in cited_ids

        return {
            "query": req.query,
            "answered": not graded.refused,
            "refused": graded.refused,
            "refusal_source": graded.refusal_source,
            "answer": graded.answer,
            "citations": [
                {
                    "chunk_id": check.chunk_id,
                    "quote": check.quote,
                    "verified": check.verified,
                    "method": check.method,
                    "start": check.start,
                    "end": check.end,
                    # What the document actually says, which under whitespace
                    # normalisation differs from what the model wrote.
                    "source_text": check.matched_text,
                    "doc_id": state.chunk_index.get(check.chunk_id, {}).get("doc_id"),
                }
                for check in report.checks
            ],
            "verification": {
                "claimed": report.n_claimed,
                "verified": report.n_verified,
                "precision": report.citation_precision,
                "buckets": {k: v for k, v in report.counts.items() if v},
                "fully_grounded": report.fully_grounded,
            },
            "retrieved": retrieved,
            "timings_ms": {
                "retrieve": round(retrieve_ms, 3),
                "generate": round(generate_ms, 3),
                "verify": round(verify_ms, 3),
                "total": round(total_ms, 3),
            },
            "tokens": (
                {
                    "prompt": graded.usage.prompt_tokens,
                    "output": graded.usage.output_tokens,
                    "thinking": graded.usage.thinking_tokens,
                }
                if graded.usage
                else None
            ),
        }

    return app
