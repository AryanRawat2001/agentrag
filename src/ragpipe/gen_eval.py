"""Phase 5 generation eval: run grounded answering over the golden set and score it.

Retrieval eval (Phase 2) needed no model and cost nothing, so it could gate every
change. This one calls an API per query, so it is a sample rather than a sweep, and
the sample is stratified by slice -- an unstratified draw of 80 from 240 would give
the `unanswerable` slice a random count, and refusal rate is the single number this
phase exists to measure.

## What is scored, and over which population

Three families, kept apart because they answer different questions and have
different denominators:

1. **Citation verification** -- `citations.py`'s buckets. Precision is averaged over
   answers that claimed at least one citation. Refusals claim none, and folding
   their undefined precision in as 0.0 would mean a model that refuses more looks
   less faithful, which is backwards.
2. **Refusal** -- measured separately on the `unanswerable` slice (where refusing is
   correct) and on the answerable slices (where it is not). Reporting one blended
   "refusal rate" would let a model that refuses everything score well on half the
   table.
3. **Cost and latency** -- tokens by kind, with thinking counted separately because
   it is billed and invisible.

## The label caveat that shapes this report

The `exact_identifier` slice cannot be read as an answerability label. Its queries
are phrased "What are the requirements of <identifier>?" and a chunk is marked
relevant when the identifier *occurs* there -- correct for retrieval, wrong for
generation. `ident-0000` is the worked example: the gold chunk is retrieved at rank
1, and the span is `'21 CFR 1.980(k)'` inside a footnote listing citations for
"post administrative detention recordkeeping". The document mentions the regulation
and never states its requirements, so refusing is right.

Consequently `refusal_rate` on `exact_identifier` is **not** an over-refusal rate,
and this module does not present it as one. The blended answerable figure is
reported with and without that slice, because the two readings differ and only one
of them means what the name suggests. Establishing true answerability is Phase 6's
golden-answer work.

## Refusal threshold calibration

Phase 3 measured refusal separability as **best-threshold accuracy** (60 answerable,
60 unanswerable, majority baseline 0.5): BM25 alone 0.983 on `structural`, min-max
fusion 0.667 on `structural` (0.725 is its best anywhere, on `semantic`), rank fusion
plus reranking 0.942. So a score gate is viable but
retriever-specific. This module measures separability on whatever retriever it is
given, as pairwise AUC -- the share of (answerable, unanswerable) query pairs where
the answerable query's top retrieval score is higher. 1.000 means a threshold exists
that separates them perfectly. **This is not a recomputation of the Phase 3 figure:**
AUC and best-threshold accuracy are different statistics, and Phase 3 pooled only
`exact_identifier` against `unanswerable` where the default here pools every
answerable slice. Both pools are reported, because pooling all answerable queries is
what `reports/retrieval_eval.md` records as having reversed a conclusion once. The
sweep that follows reports what each candidate threshold would actually do, because
a perfect AUC still leaves the choice of cut point open.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragpipe import answer as ans
from ragpipe import citations as cit

UNANSWERABLE_SLICE = "unanswerable"

#: Slices whose "relevant span" means the identifier occurs there, not that the
#: chunk answers the question. See the module docstring.
MENTION_LABELLED_SLICES = frozenset({"exact_identifier"})

#: How much of each claimed quote to persist in the report JSON. Generous on purpose:
#: a truncated quote cannot be re-verified by an independent auditor, only
#: prefix-checked. Bounded so a pathological response cannot bloat the artifact.
MAX_STORED_QUOTE_CHARS = 2000


@dataclass(slots=True)
class QueryOutcome:
    query_id: str
    slice_name: str
    query: str
    refused: bool
    refusal_source: str | None
    n_claimed: int
    n_verified: int
    buckets: dict[str, int]
    citation_precision: float | None
    top_score: float | None
    prompt_tokens: int
    output_tokens: int
    thinking_tokens: int
    latency_s: float
    attempts: int
    # Per-citation detail. Kept because a report that says "unverified" without
    # showing the text is not diagnosable -- the first real run had one query fail all
    # seven of its citations and there was no way to tell paraphrase from an extraction
    # artifact without spending quota to re-run it.
    #
    # Stored up to `MAX_STORED_QUOTE_CHARS`, not 160. At 160, six of twenty-one
    # citations in the first shipped run hit the cap, and the audit gate could then
    # only confirm *prefix* consistency for them rather than re-verify the full quote.
    # A quote is bounded by its chunk (~1,024 chars), so the generous cap costs
    # nothing and makes the artifact independently auditable, which is the point.
    checks: list[dict[str, str]] = field(default_factory=list)
    #: The generated answer. Needed by tier 2, which judges the answer against each
    #: located quote; storing it also makes a faithfulness figure re-derivable from the
    #: report without re-running generation.
    answer_text: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass
class GenEvalResult:
    outcomes: list[QueryOutcome] = field(default_factory=list)
    generator: str = ""
    retriever: str = ""
    k: int = 0

    # ---- aggregation helpers -------------------------------------------------

    def scored(self) -> list[QueryOutcome]:
        """Outcomes that produced an answer. Errors are counted, never averaged in."""
        return [o for o in self.outcomes if o.error is None]

    def by_slice(self) -> dict[str, list[QueryOutcome]]:
        out: dict[str, list[QueryOutcome]] = defaultdict(list)
        for o in self.scored():
            out[o.slice_name].append(o)
        return dict(out)

    def bucket_totals(self) -> dict[str, int]:
        totals = dict.fromkeys(
            (
                cit.EXACT,
                cit.NORMALIZED,
                cit.LINE_NUMBER_AMBIGUOUS,
                cit.WRONG_CHUNK,
                cit.UNVERIFIED,
                cit.TOO_SHORT,
            ),
            0,
        )
        for o in self.scored():
            for key, n in o.buckets.items():
                totals[key] = totals.get(key, 0) + n
        return totals

    def mean_precision(self, outcomes: Sequence[QueryOutcome] | None = None) -> float | None:
        """Mean citation precision over answers that claimed at least one citation."""
        pool = self.scored() if outcomes is None else outcomes
        vals = [o.citation_precision for o in pool if o.citation_precision is not None]
        return statistics.fmean(vals) if vals else None

    def refusal_rate(self, slice_name: str) -> float | None:
        pool = self.by_slice().get(slice_name, [])
        return (sum(1 for o in pool if o.refused) / len(pool)) if pool else None

    def answerable_refusal_rate(self, *, exclude_mention_labelled: bool) -> float | None:
        pool = [
            o
            for o in self.scored()
            if o.slice_name != UNANSWERABLE_SLICE
            and not (exclude_mention_labelled and o.slice_name in MENTION_LABELLED_SLICES)
        ]
        return (sum(1 for o in pool if o.refused) / len(pool)) if pool else None

    def separability_auc(self, *, positives: str = "all") -> tuple[float | None, int]:
        """Pairwise AUC of top retrieval score, answerable vs unanswerable.

        `positives` selects the answerable pool, and the choice is **not cosmetic**:

        - `"all"` pools every answerable slice.
        - `"identifier"` uses `exact_identifier` only, which is the population Phase 3
          measured separability over.

        Phase 3 restricted its pool deliberately, and `reports/retrieval_eval.md`
        records why: an earlier version pooled all answerable queries, and section
        headings and document titles -- long, high-IDF strings that score highly for
        unrelated reasons -- made up two thirds of that pool and *reversed which
        chunking strategy looked better*. Reintroducing the pooled version while
        citing the Phase 3 number as comparable would repeat exactly the mistake that
        report documents, so both are computed and both are reported.

        Ties count as 0.5, which is the standard treatment and matters here because
        BM25 scores can tie exactly. Returns `(auc, n_pairs)` so a small-sample AUC
        cannot be read as though it were measured over the full set.
        """
        if positives not in {"all", "identifier"}:
            raise ValueError(f"positives must be 'all' or 'identifier', got {positives!r}")
        keep = (
            (lambda name: name != UNANSWERABLE_SLICE)
            if positives == "all"
            else (lambda name: name in MENTION_LABELLED_SLICES)
        )
        pos = [o.top_score for o in self.scored() if keep(o.slice_name)]
        neg = [o.top_score for o in self.scored() if o.slice_name == UNANSWERABLE_SLICE]
        pos = [s for s in pos if s is not None]
        neg = [s for s in neg if s is not None]
        if not pos or not neg:
            return None, 0
        wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
        return wins / (len(pos) * len(neg)), len(pos) * len(neg)

    def threshold_sweep(self, n_steps: int = 12) -> list[dict[str, Any]]:
        """What each candidate score gate would do to this sample.

        A perfect AUC says a separating threshold exists; it does not say which one
        to ship. This reports, per candidate, how many unanswerable queries would be
        refused for free and how many answerable ones would be wrongly gated.
        """
        scored = [o for o in self.scored() if o.top_score is not None]
        if not scored:
            return []
        lo = min(o.top_score for o in scored)
        hi = max(o.top_score for o in scored)
        if hi <= lo:
            return []
        pos = [o for o in scored if o.slice_name != UNANSWERABLE_SLICE]
        neg = [o for o in scored if o.slice_name == UNANSWERABLE_SLICE]
        rows = []
        for i in range(n_steps + 1):
            t = lo + (hi - lo) * i / n_steps
            gated_neg = sum(1 for o in neg if o.top_score < t)
            gated_pos = sum(1 for o in pos if o.top_score < t)
            rows.append(
                {
                    "threshold": round(t, 4),
                    "unanswerable_gated": gated_neg,
                    "unanswerable_total": len(neg),
                    "answerable_gated": gated_pos,
                    "answerable_total": len(pos),
                }
            )
        return rows

    def token_totals(self) -> dict[str, int]:
        s = self.scored()
        return {
            "prompt": sum(o.prompt_tokens for o in s),
            "output": sum(o.output_tokens for o in s),
            "thinking": sum(o.thinking_tokens for o in s),
        }

    def latency_percentiles(self) -> dict[str, float | None]:
        vals = sorted(o.latency_s for o in self.scored())
        if not vals:
            return {"p50": None, "p95": None}
        return {
            "p50": round(statistics.median(vals), 3),
            "p95": round(vals[min(len(vals) - 1, int(0.95 * len(vals)))], 3),
        }

    def to_payload(self) -> dict[str, Any]:
        by_slice = self.by_slice()
        auc, n_pairs = self.separability_auc()
        auc_id, n_pairs_id = self.separability_auc(positives="identifier")
        return {
            "generator": self.generator,
            "retriever": self.retriever,
            "k": self.k,
            "n_queries": len(self.outcomes),
            "n_scored": len(self.scored()),
            "n_errors": sum(1 for o in self.outcomes if o.error is not None),
            "slices": {
                name: {
                    "n": len(pool),
                    "refusal_rate": round(sum(1 for o in pool if o.refused) / len(pool), 4),
                    "mean_citation_precision": (
                        None
                        if self.mean_precision(pool) is None
                        else round(self.mean_precision(pool), 4)
                    ),
                    "n_claimed": sum(o.n_claimed for o in pool),
                    "n_verified": sum(o.n_verified for o in pool),
                    "mention_labelled": name in MENTION_LABELLED_SLICES,
                }
                for name, pool in sorted(by_slice.items())
            },
            "buckets": self.bucket_totals(),
            "mean_citation_precision": (
                None if self.mean_precision() is None else round(self.mean_precision(), 4)
            ),
            "refusal": {
                "unanswerable": self.refusal_rate(UNANSWERABLE_SLICE),
                "answerable_all": self.answerable_refusal_rate(exclude_mention_labelled=False),
                "answerable_excl_mention_labelled": self.answerable_refusal_rate(
                    exclude_mention_labelled=True
                ),
                "mention_labelled_slices": sorted(MENTION_LABELLED_SLICES),
            },
            "separability_auc": None if auc is None else round(auc, 4),
            "separability_pairs": n_pairs,
            "separability_auc_identifier_only": None if auc_id is None else round(auc_id, 4),
            "separability_pairs_identifier_only": n_pairs_id,
            "threshold_sweep": self.threshold_sweep(),
            "tokens": self.token_totals(),
            "latency_s": self.latency_percentiles(),
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


def stratified_sample(rows: Sequence[Mapping[str, Any]], per_slice: int) -> list[Mapping[str, Any]]:
    """Take the first `per_slice` queries of each slice.

    Deterministic by construction -- the evalset is already in a fixed order and its
    generation was seeded, so taking a prefix is reproducible without carrying a
    second seed. It is not a random sample, and the report says so.
    """
    seen: Counter[str] = Counter()
    picked: list[Mapping[str, Any]] = []
    for row in rows:
        name = row.get("slice_name", "")
        if seen[name] < per_slice:
            seen[name] += 1
            picked.append(row)
    return picked


def run_gen_eval(
    rows: Iterable[Mapping[str, Any]],
    retriever: Any,
    chunk_index: Mapping[str, Mapping[str, Any]],
    *,
    generator: Any,
    k: int = 5,
    refusal_threshold: float | None = None,
    progress: Any = None,
) -> GenEvalResult:
    """Retrieve, answer, and verify each query. One API failure never aborts a run."""
    result = GenEvalResult(
        generator=getattr(generator, "name", "?"),
        retriever=getattr(retriever, "name", "?"),
        k=k,
    )
    for i, row in enumerate(rows, 1):
        query = row["query"]
        hits = retriever.search(query, k=k)
        context = [chunk_index[h.chunk_id] for h in hits if h.chunk_id in chunk_index]
        top = hits[0].score if hits else None
        started = time.monotonic()
        try:
            graded = ans.answer_query(
                query,
                context,
                generator=generator,
                top_score=top,
                refusal_threshold=refusal_threshold,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad query must not kill the run
            failed = QueryOutcome(
                query_id=row.get("query_id", ""),
                slice_name=row.get("slice_name", ""),
                query=query,
                refused=False,
                refusal_source=None,
                n_claimed=0,
                n_verified=0,
                buckets={},
                citation_precision=None,
                top_score=top,
                prompt_tokens=0,
                output_tokens=0,
                thinking_tokens=0,
                latency_s=time.monotonic() - started,
                attempts=0,
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
            result.outcomes.append(failed)
            # The failed outcome is handed to `progress`, not None. The first version
            # passed None, so the caller could only print "ERROR" -- which hid a
            # quota message that named the exact limit being exceeded.
            if progress:
                progress(i, row, failed)
            continue

        usage = graded.usage
        outcome = QueryOutcome(
            query_id=row.get("query_id", ""),
            slice_name=row.get("slice_name", ""),
            query=query,
            refused=graded.refused,
            refusal_source=graded.refusal_source,
            n_claimed=graded.verification.n_claimed,
            n_verified=graded.verification.n_verified,
            buckets={key: n for key, n in graded.verification.counts.items() if n},
            checks=[
                {
                    "method": c.method,
                    "chunk_id": c.chunk_id,
                    "quote": c.quote[:MAX_STORED_QUOTE_CHARS],
                    "matched_text": (c.matched_text or "")[:MAX_STORED_QUOTE_CHARS],
                }
                for c in graded.verification.checks
            ],
            citation_precision=graded.citation_precision,
            top_score=top,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            thinking_tokens=usage.thinking_tokens if usage else 0,
            latency_s=time.monotonic() - started,
            attempts=usage.attempts if usage else 0,
            answer_text=graded.answer,
        )
        result.outcomes.append(outcome)
        if progress:
            progress(i, row, outcome)
    return result


# A run that scored less than this fraction of its queries is not evidence, and the
# report says so at the top instead of printing means over the survivors.
#
# This guard exists because the generator did not have it. The first Phase 5 run lost
# 77 of 80 queries to a rate-limit retry storm and still rendered a clean-looking
# report whose headline read "mean citation precision 1.0" -- computed from a single
# answer. Every figure in it was arithmetically correct and the whole artifact was
# misleading, which is the exact failure mode this project keeps finding: the defect
# is never the arithmetic, it is the claim.
MIN_SCORED_FRACTION = 0.8


@dataclass
class Faithfulness:
    """Tier 1 and tier 2 composed: is the quote real, and does it support the answer?

    The two tiers answer different questions and neither implies the other, so all three
    numbers are reported. A citation can be:

    - **located and supporting** — the only case that is actually evidence.
    - **located but not supporting** — the quote is real and does not back the claim.
      Phase 6 found this in the golden set at roughly a quarter of pairs, which is why
      tier 1 alone was never going to be enough.
    - **not located** — tier 1 already rejected it; tier 2 is not asked, because judging
      whether a fabricated quote supports a claim is a category error.

    `supported_of_located` is the headline: of the citations that survive the
    deterministic check, how many survive reading. Dividing by *all* claimed citations
    instead would fold tier-1 failures into a tier-2 number and make the two
    indistinguishable.
    """

    #: What the verdict counts below are counts *of*: "answer" or "citation". Without it
    #: the ratios silently mix units -- under answer-level judging `n_supported` counts
    #: answers while `n_located` counts citations, and the first report divided one by
    #: the other and printed 0.318 as "the share of citations that survive reading". It
    #: is neither.
    unit: str = "answer"
    n_claimed: int = 0
    n_located: int = 0
    n_supported: int = 0
    n_partial: int = 0
    n_unsupported: int = 0
    n_unjudged: int = 0
    answers_fully_supported: int = 0
    answers_with_citations: int = 0

    @property
    def supported_of_located(self) -> float | None:
        """Citation-level only. **None under answer-level judging**, where the numerator
        counts answers and the denominator citations -- a ratio with no meaning."""
        if self.unit != "citation" or not self.n_located:
            return None
        return self.n_supported / self.n_located

    @property
    def supported_of_claimed(self) -> float | None:
        """End to end, both tiers, over everything claimed. Citation-level only, for the
        same reason as above."""
        if self.unit != "citation" or not self.n_claimed:
            return None
        return self.n_supported / self.n_claimed

    @property
    def answer_level_rate(self) -> float | None:
        """Share of citing answers whose every citation located *and* supported.

        Stricter than the citation-level rate and closer to what a user experiences: one
        bad citation in an answer of four is a bad answer.
        """
        if not self.answers_with_citations:
            return None
        return self.answers_fully_supported / self.answers_with_citations

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "n_claimed": self.n_claimed,
            "n_located": self.n_located,
            "n_supported": self.n_supported,
            "n_partial": self.n_partial,
            "n_unsupported": self.n_unsupported,
            "n_unjudged": self.n_unjudged,
            "supported_of_located": (
                None if self.supported_of_located is None else round(self.supported_of_located, 4)
            ),
            "supported_of_claimed": (
                None if self.supported_of_claimed is None else round(self.supported_of_claimed, 4)
            ),
            "answers_with_citations": self.answers_with_citations,
            "answers_fully_supported": self.answers_fully_supported,
            "answer_level_rate": (
                None if self.answer_level_rate is None else round(self.answer_level_rate, 4)
            ),
        }


def load_gen_eval(payload: Mapping[str, Any]) -> GenEvalResult:
    """Rebuild a result from a written report.

    Exists so tier 2 can be re-run without re-running generation. Judging is a few
    requests; generating is one per query, and the first two-tier run showed the tier-2
    *unit* was wrong — a mistake that would otherwise have cost a full regeneration to
    correct. Everything tier 2 needs (answer text, per-citation methods and matched
    spans) is already persisted for auditability, so this is free.
    """
    result = GenEvalResult(
        generator=str(payload.get("generator") or ""),
        retriever=str(payload.get("retriever") or ""),
        k=int(payload.get("k") or 0),
    )
    for raw in payload.get("outcomes") or []:
        fields = {f: raw.get(f) for f in QueryOutcome.__slots__ if f in raw}
        fields.setdefault("checks", [])
        fields.setdefault("answer_text", "")
        result.outcomes.append(QueryOutcome(**fields))
    return result


def judge_items_for(result: GenEvalResult, unit: str = "answer") -> list[Any]:
    """Build tier-2 items from an eval run.

    Only *located* citations contribute. Asking whether a quote that does not exist
    supports a claim wastes a request on a question tier 1 already answered, and mixing
    the two would make the tier-2 rate uninterpretable.

    `unit` selects what gets judged, and **the first real run showed the choice is not a
    detail**:

    - `"answer"` (default) — one item per answer, carrying *all* of its located quotes.
      The question is "does this evidence, taken together, support this answer", which is
      the question a reader actually has.
    - `"citation"` — one item per located citation, each judged against the whole answer.

    The per-citation unit was the original default and it produced a number that measured
    the unit rather than the model: **18 of 22 verdicts came back `partial`**, giving
    `supported_of_located = 0.136`. The mechanism is structural, not judge noise — the
    judge scored 1.000 on both control sides in the same run. Asking whether *one* quote
    supports a *whole* answer is asking the wrong question whenever an answer cites
    several, and the two single-citation answers in that run were exactly the ones judged
    `supported`. The code comment here had predicted `partial` would be "the expected
    result for one citation of several" and then reported the resulting rate anyway.

    The per-citation unit is kept because it is a useful diagnostic — it localises *which*
    citation is weak — but it is no longer the headline.
    """
    from ragpipe.judge import JudgeItem

    if unit not in {"answer", "citation"}:
        raise ValueError(f"unit must be 'answer' or 'citation', got {unit!r}")

    items: list[Any] = []
    for outcome in result.scored():
        if outcome.refused:
            continue
        located = [c for c in outcome.checks if c.get("method") in cit.VERIFIED_METHODS]
        if not located:
            continue
        if unit == "citation":
            for i, check in enumerate(located):
                items.append(
                    JudgeItem(
                        item_id=f"{outcome.query_id}::{i}",
                        claim=outcome.answer_text or "",
                        quote=check.get("matched_text") or check.get("quote") or "",
                        source_id=outcome.query_id,
                    )
                )
            continue
        # One item per answer. Quotes are numbered so the judge can name which one it is
        # talking about in its reason, which is what makes a `partial` verdict actionable.
        quotes = "\n".join(
            f"({i + 1}) {c.get('matched_text') or c.get('quote') or ''}"
            for i, c in enumerate(located)
        )
        items.append(
            JudgeItem(
                item_id=outcome.query_id,
                claim=outcome.answer_text or "",
                quote=quotes,
                source_id=outcome.query_id,
            )
        )
    return items


def score_answer_faithfulness(result: GenEvalResult, judgments: Mapping[str, Any]) -> Faithfulness:
    """Score answer-level judgments: one verdict per answer, over all its evidence."""
    from ragpipe.judge import PARTIAL, SUPPORTED, UNSUPPORTED

    f = Faithfulness(unit="answer")
    for outcome in result.scored():
        located = 0
        for check in outcome.checks:
            f.n_claimed += 1
            if check.get("method") in cit.VERIFIED_METHODS:
                located += 1
        f.n_located += located
        if outcome.refused or not located:
            continue
        j = judgments.get(outcome.query_id)
        if j is None:
            # Counted, never scored. An unjudged answer's faithfulness is *undefined*, not
            # zero -- the same reasoning that keeps refusals out of citation precision, and
            # the treatment `judge.Agreement` already gave unjudged controls. Incrementing
            # `answers_with_citations` before the lookup put missing verdicts in the
            # denominator as failures: dropping 3 of 8 verdicts turned 0.875 into 0.500,
            # and one swallowed request does exactly that.
            f.n_unjudged += 1
            continue
        f.answers_with_citations += 1
        if j.verdict == SUPPORTED:
            f.n_supported += 1
            f.answers_fully_supported += 1
        elif j.verdict == PARTIAL:
            f.n_partial += 1
        elif j.verdict == UNSUPPORTED:
            f.n_unsupported += 1
    return f


def score_faithfulness(result: GenEvalResult, judgments: Mapping[str, Any]) -> Faithfulness:
    """Combine tier-1 outcomes with tier-2 verdicts into one report."""
    from ragpipe.judge import PARTIAL, SUPPORTED, UNSUPPORTED

    f = Faithfulness(unit="citation")
    for outcome in result.scored():
        # Refusals are skipped here as in `judge_items_for` and the answer-level scorer.
        # `answer.py` deliberately preserves citations on a refusal ("a model that refuses
        # *and* cites is inconsistent"), so a refused-but-citing answer is reachable -- and
        # this scorer counted its never-judged quotes as tier-2 failures, driving
        # `supported_of_located` from 1.000 to 0.333.
        if outcome.refused:
            continue
        located_ids: list[str] = []
        for i, check in enumerate(outcome.checks):
            f.n_claimed += 1
            if check.get("method") not in cit.VERIFIED_METHODS:
                continue
            f.n_located += 1
            located_ids.append(f"{outcome.query_id}::{i}")

        if not located_ids:
            continue
        f.answers_with_citations += 1
        verdicts = []
        for item_id in located_ids:
            j = judgments.get(item_id)
            if j is None:
                # Not scored as a failure -- see the answer-level scorer.
                f.n_unjudged += 1
                f.n_located -= 1
                verdicts.append(None)
                continue
            verdicts.append(j.verdict)
            if j.verdict == SUPPORTED:
                f.n_supported += 1
            elif j.verdict == PARTIAL:
                f.n_partial += 1
            elif j.verdict == UNSUPPORTED:
                f.n_unsupported += 1
        if verdicts and all(v == SUPPORTED for v in verdicts):
            f.answers_fully_supported += 1
    return f


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def render_gen_eval(payload: Mapping[str, Any]) -> str:
    """Render the Phase 5 report. Every figure is read from `payload`.

    Phase 4's audit found four stale literals hardcoded inside a report generator and
    reprinted as current on every run. Nothing here is written down that can be
    computed from the payload.
    """
    lines: list[str] = []
    a = lines.append
    a("# Generation evaluation — grounded answers with verified citations")
    a("")

    n_queries = payload["n_queries"] or 1
    scored_fraction = payload["n_scored"] / n_queries
    if scored_fraction < MIN_SCORED_FRACTION:
        a(
            f"> **This run is not usable as evidence.** Only "
            f"{payload['n_scored']} of {payload['n_queries']} queries "
            f"({scored_fraction * 100:.0f}%) produced an answer; "
            f"{payload['n_errors']} errored. Every figure below is computed over the "
            f"survivors, which is a biased sample of unknown shape — a rate-limit "
            f"failure, for instance, correlates with nothing about the queries but a "
            f"safety block correlates with everything. Re-run before citing any "
            f"number here."
        )
        a("")
    elif payload["n_errors"]:
        a(
            f"> {payload['n_errors']} of {payload['n_queries']} queries errored and are "
            f"excluded from every figure below. Denominators are the "
            f"{payload['n_scored']} scored queries."
        )
        a("")

    a(
        f"Generator `{payload['generator']}`, retriever `{payload['retriever']}`, "
        f"top-{payload['k']} context. "
        f"{payload['n_scored']} of {payload['n_queries']} queries scored"
        + (f", {payload['n_errors']} errored." if payload["n_errors"] else ".")
    )
    a("")
    a(
        "Queries are the first N of each slice from `corpus/evalset.jsonl` — a "
        "deterministic stratified prefix, not a random sample. Stratified because "
        "refusal rate is the headline and an unstratified draw would give the "
        "`unanswerable` slice a random size."
    )
    a("")

    a("## Citation verification")
    a("")
    a(
        "Tier 1 locates each claimed quote in the source it cites (plan deviation "
        "#14). Nothing the model reports about position is trusted."
    )
    a("")
    buckets = payload["buckets"]
    total_claimed = sum(buckets.values())
    a("| Outcome | Citations | Share | Meaning |")
    a("|---|---:|---:|---|")
    meanings = {
        cit.EXACT: "byte-identical to the cited chunk",
        cit.NORMALIZED: "matches modulo whitespace runs (PDF artifacts)",
        cit.LINE_NUMBER_AMBIGUOUS: (
            "**not verified** — failed the strict check, but would match if "
            "interleaved line numbers were discounted; an extraction-artifact hint"
        ),
        cit.WRONG_CHUNK: "real text, wrong provenance — a plumbing defect",
        cit.UNVERIFIED: "not in the document — fabricated or paraphrased",
        cit.TOO_SHORT: f"under {cit.MIN_QUOTE_CHARS} chars — not evidence",
    }
    # Iterate the payload, not a hardcoded list. The known methods come first for a
    # stable reading order, then anything else the payload carries -- otherwise a new
    # bucket would be omitted from the table while still inflating `total_claimed`,
    # which is the same silent-omission defect `VerificationReport.counts` exists to
    # prevent. Adding one bucket this phase required editing four separate literals.
    known = (
        cit.EXACT,
        cit.NORMALIZED,
        cit.LINE_NUMBER_AMBIGUOUS,
        cit.WRONG_CHUNK,
        cit.UNVERIFIED,
        cit.TOO_SHORT,
    )
    for key in list(known) + [k for k in sorted(buckets) if k not in known]:
        n = buckets.get(key, 0)
        share = f"{n / total_claimed * 100:.1f}%" if total_claimed else "n/a"
        a(f"| `{key}` | {n} | {share} | {meanings.get(key, '(unrecognised bucket)')} |")
    a(f"| **total claimed** | **{total_claimed}** | | |")
    a("")
    a(
        f"Mean citation precision (verified / claimed, averaged over answers that "
        f"claimed at least one citation): **{_num(payload['mean_citation_precision'])}**. "
        "Refusals claim nothing and are excluded — scoring their undefined precision "
        "as zero would make a model that refuses correctly look less faithful."
    )
    a("")
    n_verified_total = sum(buckets.get(k, 0) for k in cit.VERIFIED_METHODS)
    micro = _num(n_verified_total / total_claimed) if total_claimed else "n/a"
    a(
        f"That is a **macro** average over answers. The **micro** figure — verified "
        f"citations over all claimed citations, which is what the bucket table above "
        f"sums to — is **{micro}** ({n_verified_total}/{total_claimed}). "
        "Both are correct and they answer different questions: the macro figure "
        "weights every answer equally, the micro figure weights every citation "
        "equally. An answer with one citation moves the macro average as much as an "
        "answer with nine."
    )
    a("")

    a("## Refusal")
    a("")
    ref = payload["refusal"]
    a("| Population | Refusal rate | Correct behaviour |")
    a("|---|---:|---|")
    a(f"| `unanswerable` slice | {_pct(ref['unanswerable'])} | refusing — higher is better |")
    a(
        f"| answerable slices (all) | {_pct(ref['answerable_all'])} | "
        "answering — but see the caveat |"
    )
    a(
        f"| answerable, excl. mention-labelled | "
        f"{_pct(ref['answerable_excl_mention_labelled'])} | answering — lower is better |"
    )
    a("")
    rows_differ = ref["answerable_all"] != ref["answerable_excl_mention_labelled"]
    a(
        (
            "**The two answerable rows differ, and only the second means what the name suggests.**"
            if rows_differ
            else "**The two answerable rows happen to be equal in this run, so the "
            "caveat below changes no figure here — but it is not cosmetic, and the "
            "second row is the one that means what its name suggests.**"
        )
        + f" The {', '.join('`' + s + '`' for s in ref['mention_labelled_slices'])} "
        "slice marks a chunk relevant when an identifier *occurs* in it, which is the "
        "right label for retrieval and the wrong one for generation. Its queries ask "
        "what a regulation requires, while the corpus mostly just cites the "
        "regulation, so a refusal there is often correct rather than an over-refusal. "
        "Establishing real answerability is Phase 6's work."
    )
    a("")

    a("## Per slice")
    a("")
    a("| Slice | n | Refusal rate | Mean cite precision | Claimed | Verified | Label caveat |")
    a("|---|---:|---:|---:|---:|---:|---|")
    for name, s in payload["slices"].items():
        caveat = "mention-labelled" if s["mention_labelled"] else ""
        a(
            f"| `{name}` | {s['n']} | {_pct(s['refusal_rate'])} | "
            f"{_num(s['mean_citation_precision'])} | {s['n_claimed']} | "
            f"{s['n_verified']} | {caveat} |"
        )
    a("")

    a("## Refusal threshold calibration")
    a("")
    auc = payload["separability_auc"]
    a(
        f"Pairwise separability of top retrieval score, answerable vs unanswerable: "
        f"**{_num(auc, 4)}** AUC over {payload['separability_pairs']} pairs — the share "
        "of (answerable, unanswerable) pairs the retriever orders correctly, ties "
        "counting 0.5."
    )
    a("")
    a("| Answerable pool | AUC | Pairs |")
    a("|---|---:|---:|")
    a(f"| all answerable slices | {_num(auc, 4)} | {payload['separability_pairs']} |")
    a(
        f"| `exact_identifier` only | "
        f"{_num(payload['separability_auc_identifier_only'], 4)} | "
        f"{payload['separability_pairs_identifier_only']} |"
    )
    a("")
    a(
        "Both pools are shown because the choice is not cosmetic. Phase 3 measured "
        "separability over `exact_identifier` against `unanswerable` only, and "
        "`reports/retrieval_eval.md` records why: pooling all answerable queries let "
        "section headings and document titles — long, high-IDF strings that score "
        "highly for unrelated reasons — dominate the pool and *reverse* which chunking "
        "strategy looked better. The pooled figure is the headline here because it "
        "describes the whole eval set, but the restricted row is the one comparable in "
        "population to Phase 3 (though still not in statistic — that was accuracy)."
    )
    a("")
    sweep = payload["threshold_sweep"]
    if sweep:
        a(
            (
                "A perfect AUC says a separating threshold exists, not which to ship. "
                if auc == 1.0
                else "An AUC below 1.0 means no threshold separates the two "
                "populations cleanly, so any cut point trades one error against the "
                "other. "
            )
            + "What each candidate would do to this sample:"
        )
        a("")
        a("| Threshold | Unanswerable gated | Answerable wrongly gated |")
        a("|---:|---:|---:|")
        for row in sweep:
            a(
                f"| {row['threshold']:.4f} | "
                f"{row['unanswerable_gated']}/{row['unanswerable_total']} | "
                f"{row['answerable_gated']}/{row['answerable_total']} |"
            )
        a("")
    a(
        "The score gate is **off by default**, because the mechanism is "
        "retriever-specific. Phase 3 measured best-threshold separability accuracy at "
        "0.983 for BM25 alone on `structural`, against 0.667 for min-max fusion and "
        "0.942 for rank fusion plus reranking — so the threshold must be validated "
        "against whatever Phase 7 actually serves rather than inherited from an "
        "earlier table. Note that accuracy is a *different statistic* from the AUC "
        "above and the two are not comparable."
    )
    a("")

    faith = payload.get("faithfulness")
    if faith and faith.get("n_located"):
        a("## Faithfulness — tier 1 and tier 2 composed")
        a("")
        a(
            "Tier 1 asks whether the quote is really in the document it cites. Tier 2 asks "
            "whether that quote supports the answer. **Neither implies the other**, and "
            "Phase 6 measured the gap directly: roughly a quarter of golden pairs had "
            "quotes that were real and did not back the claim."
        )
        a("")
        unit = faith.get("unit", "answer")
        noun = "answers" if unit == "answer" else "citations"
        a("| Measure | Value | Reading |")
        a("|---|---:|---|")
        a(f"| Citations claimed | {faith['n_claimed']} | what the model asserted |")
        a(f"| Located (tier 1) | {faith['n_located']} | the quote exists where cited |")
        a("")
        a(f"Tier-2 verdicts, counted over **{noun}**:")
        a("")
        a(f"| Verdict | {noun.capitalize()} |")
        a("|---|---:|")
        a(f"| supported | {faith['n_supported']} |")
        a(f"| partial | {faith['n_partial']} |")
        a(f"| unsupported | {faith['n_unsupported']} |")
        a(f"| unjudged | {faith['n_unjudged']} |")
        a("")
        if unit == "answer":
            a(
                f"**{faith['answers_fully_supported']} of "
                f"{faith['answers_with_citations']} answers fully supported = "
                f"{_num(faith['answer_level_rate'])}.** The question put to the judge is "
                "whether *all* of an answer's located quotes together support it — one "
                "verdict per answer. This is the faithfulness figure."
            )
            a("")
            a(
                "Citation-level ratios are **deliberately absent** under this unit. "
                "`n_supported` counts answers while `n_located` counts citations, so "
                "dividing them mixes units: an earlier version of this report did exactly "
                'that and printed 7/22 = 0.318 as "the share of citations that survive '
                'reading", which it is not. Re-run with `--judge-unit citation` for a '
                "genuine per-citation breakdown."
            )
        else:
            a(
                f"**Supported of located: {_num(faith['supported_of_located'])}** — of the "
                "citations that survive the deterministic check, the share that survive "
                "reading. Dividing by all claimed citations would fold tier-1 failures "
                "into a tier-2 number."
            )
            a("")
            a(
                f"End to end, both tiers over everything claimed: "
                f"**{_num(faith['supported_of_claimed'])}**."
            )
            a("")
            a(
                "**Read this unit with care.** Each citation is judged against the whole "
                "answer, so `partial` is the *expected* verdict whenever an answer cites "
                "several — 18 of 22 in the first real run, which measured the unit rather "
                "than the model. Useful for localising a weak citation; not a "
                "faithfulness rate."
            )
        a("")
        cal = payload.get("judge_calibration")
        if cal:
            a(
                f"Judge `{payload.get('judge', '?')}`, calibrated before use: it rejected "
                f"{_pct(cal.get('negative_rate'))} of constructed negatives (a real answer "
                f"paired with another document's evidence, which cannot be supported) and "
                f"accepted {_pct(cal.get('positive_rate'))} of presumed-supported golden "
                f"pairs. The negative rate is the one that decides usability: a judge "
                f'answering "supported" to everything scores 1.000 on positives and '
                f"0.000 there."
            )
            a("")

    a("## Cost and latency")
    a("")
    tokens = payload["tokens"]
    lat = payload["latency_s"]
    a("| Measure | Value |")
    a("|---|---:|")
    a(f"| Prompt tokens | {tokens['prompt']:,} |")
    a(f"| Output tokens | {tokens['output']:,} |")
    a(f"| Thinking tokens | {tokens['thinking']:,} |")
    a(f"| Latency p50 | {_num(lat['p50'])}s |")
    a(f"| Latency p95 | {_num(lat['p95'])}s |")
    a("")
    a(
        f"The two latency rows use different conventions, which is worth stating rather "
        f"than leaving to be discovered: `p50` is the interpolated median, `p95` is "
        f"nearest-rank over {payload['n_scored']} observations — at this sample size that "
        "makes p95 the single slowest query. Neither is a computation error; mixing them "
        "silently would be."
    )
    a("")
    a(
        "Thinking tokens are reported separately because they are billed and invisible "
        "in the response. They are zero here by configuration: Gemini 3.x thinks by "
        "default, `thinkingBudget: 0` is rejected with a 400, and "
        '`thinkingLevel: "minimal"` is the only setting that reaches zero — the '
        "default spent 105 thinking tokens to emit the single token `OK`."
    )
    a("")
    a(
        "No dollar figure: the free tier bills nothing and confirmed paid per-token "
        "rates for this model were not available, so a cost column would be a guess. "
        "Phase 4's audit found four such guessed literals in a report generator."
    )
    a("")
    return "\n".join(lines)


def write_gen_eval(payload: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "generation_eval.md"
    json_path = out_dir / "generation_eval.json"
    md_path.write_text(render_gen_eval(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return md_path, json_path
