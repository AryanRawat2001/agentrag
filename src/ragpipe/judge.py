"""Tier 2 of citation verification: does the span actually support the claim?

Tier 1 (`citations.py`) is deterministic and answers a narrow question — is this quote
really in the document it cites. It cannot answer the question that matters next, and
Phase 6 produced a clean demonstration of the gap: a golden pair whose evidence quote
was real, locatable, and correctly attributed, supporting an answer that asserted
Structured Product Labeling "is an HL7 standard" when the cited chunk never mentions
HL7. Every deterministic check passed. The answer still did not follow.

That requires reading, so this tier uses a model. Which introduces the problem the rest
of this module is about.

## An uncalibrated judge is worse than no judge

A judge produces confident numbers whether or not it works, and those numbers land in a
faithfulness table that nobody re-derives. So before it is used on unknown pairs it is
run against cases whose answer is already known:

- **Known positives** — golden pairs that passed tier 1 and the numeric triage. Not
  certainly supported, but the best available approximation.
- **Injected negatives** — a real answer paired with a *different* pair's evidence, from
  a different document. These are constructed, so they are unambiguous: the evidence
  cannot support the claim. A judge that cannot reject these cannot be trusted to reject
  anything, and the failure would be invisible without the control.

`judge_agreement` reports both rates. The negatives matter more: a judge that says
"supported" to everything scores perfectly on positives alone, which is exactly the
failure mode a positives-only calibration cannot see.

## Do not let one model grade its own homework

`DEFAULT_JUDGE` deliberately differs from the drafting and answering default. A model
scoring its own output has a documented bias toward it, and here the risk is concrete:
the same model wrote both the answer and the reasoning that justified it, so agreement
would partly measure self-consistency rather than support. The judge is configurable
precisely so "does the verdict change with the judge" is answerable.

## Batching, and why the unit is the claim

Quota is counted per request against a 1M context (deviation #15), so judgments are
batched. The unit is one (claim, evidence) pair rather than one answer, because an answer
with four citations can be supported on three and fabricated on the fourth, and a
verdict on the whole answer would hide that. Aggregation happens after.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Judgments per request. Smaller than the golden-set batch of 10: each item carries a
#: full evidence quote plus an answer, and a truncated response loses the whole batch —
#: which cost 10 golden pairs before `max_output_tokens` was raised.
DEFAULT_BATCH_SIZE = 8

#: Deliberately not the drafting/answering default. See the module docstring.
DEFAULT_JUDGE = "gemini-3.5-flash-lite"

#: A judge must reject at least this share of constructed negatives to be used at all.
#: Blunt on purpose: the negatives cannot be supported under any reading, so anything
#: below this is a broken instrument rather than a borderline one.
MIN_NEGATIVE_REJECTION_RATE = 0.8

#: Below this share of presumed-supported positives, *inspect the golden set* before
#: touching the judge. Not a gate — the first calibration scored 0.750 here with a judge
#: that was correct on every disagreement.
LOW_POSITIVE_RATE_WARNING = 0.8

#: A sanity floor on positives, well below the warning threshold. Gating on negatives
#: alone has a hole: a judge that answers "unsupported" to everything rejects every
#: constructed negative and would pass. This floor closes it while still admitting the
#: real case (0.750 with correct reasoning). The band between this and the warning is
#: genuinely ambiguous — a moderate positive rate can mean a cautious judge or a golden
#: set whose answers outrun their evidence, and only reading the disagreements
#: distinguishes them, which is why `--show` exists.
MIN_POSITIVE_ACCEPTANCE_RATE = 0.4

#: Share of control items that must actually come back judged. Rates divide by judged
#: items only, so without this a batch where 15 of 16 items went missing could report a
#: perfect rate from the one survivor.
MIN_JUDGED_SHARE = 0.8

#: Minimum controls per side. Rates alone are not enough: `Agreement(n_positive=1,
#: n_negative=1, ...)` reports `negative_rate: 1.0, usable: True` off a single item, and
#: `build_control_set` silently returns fewer negatives than requested when the corpus
#: cannot supply them (20 pairs in one document and 1 in another yields 16 positives and
#: 2 negatives). `MIN_JUDGED_SHARE` cannot catch that -- items never *constructed* are
#: not counted as unjudged. Structurally the same hole as the response-side gate, moved
#: to the construction side.
MIN_CONTROLS_PER_SIDE = 4

SUPPORTED = "supported"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"
VERDICTS = (SUPPORTED, PARTIAL, UNSUPPORTED)

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "reason": {"type": "string"},
                },
                "required": ["item_id", "verdict", "reason"],
                "propertyOrdering": ["item_id", "verdict", "reason"],
            },
        }
    },
    "required": ["judgments"],
    "propertyOrdering": ["judgments"],
}

JUDGE_SYSTEM_PROMPT = """\
You decide whether a quoted passage supports a claim. You are an auditor, not an \
assistant: your job is to catch claims that outrun their evidence.

For each item you are given a CLAIM and a QUOTE taken from a source document. Judge \
only whether the quote supports the claim.

Verdicts:

- "supported" — the quote states or directly entails everything the claim asserts.
- "partial" — the quote supports part of the claim but the claim adds something the \
quote does not state.
- "unsupported" — the quote does not support the claim, or is about something else.

Rules:

1. Judge the quote in front of you, and nothing else. If a claim is true in the world, \
or true elsewhere in the same document, but is not stated in this quote, it is not \
supported. This is the most common mistake: a claim like "SPL is an HL7 standard" \
alongside a quote that mentions SPL and XML but never HL7 is **unsupported**, however \
correct it may be.
2. Do not reward fluency. A well-written claim with thin evidence is still unsupported.
3. Quotes come from PDF extraction and may contain broken words, stray digits from line \
numbers or footnote markers, and collapsed spacing. Read through those artifacts — they \
are not the claim's fault and are not grounds for a verdict.
4. "reason" must be one sentence naming the specific thing that is or is not supported. \
Do not restate the verdict.
5. Return exactly one judgment per item, using the item_id given.
"""


@dataclass(frozen=True, slots=True)
class JudgeItem:
    """One (claim, evidence) pair to judge, with the label if it is a control."""

    item_id: str
    claim: str
    quote: str
    #: "positive" | "negative" | None. None means a real pair under test.
    control: str | None = None
    source_id: str = ""

    @property
    def is_control(self) -> bool:
        return self.control is not None


@dataclass(frozen=True, slots=True)
class Judgment:
    item_id: str
    verdict: str
    reason: str = ""

    @property
    def is_supported(self) -> bool:
        return self.verdict == SUPPORTED


def build_prompt(items: Sequence[JudgeItem]) -> str:
    """Render a batch. Item ids are echoed back so judgments cannot be misaligned.

    Positional matching was the obvious alternative and is a silent-corruption risk: a
    model that returns seven judgments for eight items would shift every verdict after
    the gap onto the wrong claim, and the totals would still look plausible.
    """
    blocks = []
    for item in items:
        blocks.append(f"[item_id: {item.item_id}]\nCLAIM: {item.claim}\nQUOTE: {item.quote}")
    return (
        "Judge each item below.\n\n"
        + "\n\n".join(blocks)
        + f"\n\nReturn exactly {len(items)} judgments, one per item_id."
    )


def parse_judgments(payload: Mapping[str, Any]) -> dict[str, Judgment]:
    """Index judgments by item_id, dropping malformed entries.

    Keyed rather than ordered for the reason in `build_prompt`. A missing item is
    visible downstream as an unjudged item rather than as a shifted verdict.
    """
    out: dict[str, Judgment] = {}
    for raw in payload.get("judgments") or []:
        if not isinstance(raw, Mapping):
            continue
        item_id = str(raw.get("item_id") or "")
        verdict = str(raw.get("verdict") or "")
        if not item_id or verdict not in VERDICTS:
            continue
        out[item_id] = Judgment(item_id, verdict, str(raw.get("reason") or ""))
    return out


def build_control_set(pairs: Sequence[Any], n_positive: int, n_negative: int) -> list[JudgeItem]:
    """Build a calibration set from golden pairs: real ones, plus mismatched negatives.

    A negative takes one pair's answer and another pair's evidence, requiring the two to
    come from **different documents** so the mismatch cannot be accidentally supported by
    related text. These are the only items whose correct verdict is known with certainty,
    which is what makes them the useful half of the calibration.

    Deterministic: documents are visited in sorted order and negatives are formed by
    pairing each claim with the evidence of the nearest following pair from a *different*
    document, so a re-run calibrates on the same set and a change in agreement is
    attributable to the judge. (There is no fixed offset any more — see the zero-negatives
    note below.)

    **Controls are spread across documents, not taken as a prefix.** A prefix of
    pair_id-sorted golden pairs is one document's worth, which makes the calibration a
    measurement of that document rather than of the set.
    """
    accepted = sorted(
        (p for p in pairs if getattr(p, "accepted", False) and p.evidence),
        key=lambda p: p.pair_id,
    )
    # Spread across documents, round-robin. Sorting by `pair_id` and taking a prefix --
    # which is what this did -- draws every control from ONE document, because golden
    # pair_ids are prefixed by doc_id. The Phase 6 audit gate found all 8 positives coming
    # from `ctgov-NCT00567567-Prot_SAP_000`, which is also the document Phase 6a singled
    # out for spurious intra-word spaces and a 10/10 -> 5/10 drafting regression. So the
    # calibration was measuring the corpus's worst document and the resulting 0.750
    # positive rate was being extrapolated to "golden pairs" generally.
    #
    # `sample_for_review` in `golden.py` already did this for exactly the same reason. The
    # lesson did not travel between modules.
    by_doc: dict[str, list[Any]] = {}
    for pair in accepted:
        by_doc.setdefault(pair.doc_id, []).append(pair)
    usable: list[Any] = []
    depth = 0
    while len(usable) < len(accepted):
        added = False
        for doc in sorted(by_doc):
            bucket = by_doc[doc]
            if depth < len(bucket):
                usable.append(bucket[depth])
                added = True
        if not added:
            break
        depth += 1

    items: list[JudgeItem] = []
    for p in usable[:n_positive]:
        items.append(
            JudgeItem(
                item_id=f"pos::{p.pair_id}",
                claim=p.answer,
                quote=p.evidence[0]["quote"],
                control="positive",
                source_id=p.pair_id,
            )
        )
    # For each claim, walk forward cyclically to the first pair from a *different*
    # document. Deterministic, and correct for any ordering.
    #
    # The previous version offset by half the list and skipped same-document collisions.
    # That was fine under pair_id ordering and silently catastrophic under the
    # round-robin ordering above: with D documents interleaved, an offset of len//2 is a
    # multiple of D whenever len is, so *every* candidate collided and the function
    # returned **zero negatives**. Zero negatives means `negative_rate` is None, `usable`
    # is False, and faithfulness is omitted -- it fails safe, but it fails silently, and
    # only a test with uniform document sizes exposed it.
    made = 0
    for i, p in enumerate(usable):
        if made >= n_negative:
            break
        other = next(
            (
                usable[(i + step) % len(usable)]
                for step in range(1, len(usable))
                if usable[(i + step) % len(usable)].doc_id != p.doc_id
            ),
            None,
        )
        if other is None:
            continue
        # A different `doc_id` is not sufficient. This corpus repeats passages across
        # documents -- 239 normalised 180-character windows occur in exactly two -- so a
        # mismatched pairing can genuinely *contain* the claim, which a correct judge must
        # call `supported` and which is then scored as a negative-side failure. Reject any
        # candidate whose claim is largely covered by the mismatched quote. Uses the same
        # lexical coverage as the golden-set filter; the current corpus maxes out at 0.400
        # cross-document, so this rejects nothing today and prevents a silent regression on
        # the next golden set.
        from ragpipe.golden import answer_evidence_coverage

        if answer_evidence_coverage(p.answer, [other.evidence[0]["quote"]]) > 0.5:
            continue
        items.append(
            JudgeItem(
                item_id=f"neg::{p.pair_id}",
                claim=p.answer,
                quote=other.evidence[0]["quote"],
                control="negative",
                source_id=f"{p.pair_id}|{other.pair_id}",
            )
        )
        made += 1
    return items


@dataclass
class Agreement:
    """How well the judge did on items whose answer was known."""

    n_positive: int = 0
    n_negative: int = 0
    positive_supported: int = 0
    negative_rejected: int = 0
    unjudged: int = 0
    verdicts: dict[str, int] = field(default_factory=dict)

    @property
    def positive_rate(self) -> float | None:
        """Share of known-good items called supported. High is good, but see below."""
        return self.positive_supported / self.n_positive if self.n_positive else None

    @property
    def negative_rate(self) -> float | None:
        """Share of mismatched items correctly *not* called supported.

        This is the number that decides whether the judge is usable. A judge that
        answers "supported" to everything scores 1.0 on positives and 0.0 here, and a
        positives-only calibration would call it perfect.
        """
        return self.negative_rejected / self.n_negative if self.n_negative else None

    @property
    def usable(self) -> bool:
        """Whether the judge can be trusted to produce faithfulness numbers.

        **Gated on the negative rate alone**, and the asymmetry is the point. The two
        rates are not equally informative:

        - A low **negative** rate is unambiguous. The negatives are constructed — a real
          answer paired with a different document's evidence — so there is no reading
          under which they are supported. Failing them means the judge is broken.
        - A low **positive** rate is ambiguous. The positives are only *presumed*
          supported: they are golden pairs that passed tier 1 and numeric triage, which
          proves the quote is real and says nothing about whether the answer follows.

        The first calibration made the distinction concrete. Negatives scored 1.000 and
        positives 0.750, and reading the judge's four disagreements showed it was right
        every time: the golden answers synthesised across a whole chunk while citing a
        single span ("the quote does not mention Dr. Robert C. Seeger", "does not mention
        the condition of microscopic or gross hematuria"). An earlier version of this
        property required both rates to clear 0.8 and therefore declared a working judge
        unusable on the strength of defective reference data.

        `positive_rate` is still reported, and a low value is a signal worth acting on —
        it just points at the golden set first, not at the judge. It is also held to a
        low floor, because gating on negatives alone lets a judge that answers
        "unsupported" to everything through: it rejects every constructed negative
        perfectly. Its own test caught that.
        """
        if self.n_positive < MIN_CONTROLS_PER_SIDE or self.n_negative < MIN_CONTROLS_PER_SIDE:
            return False
        judged = self.n_positive + self.n_negative
        total = judged + self.unjudged
        if not total or judged / total < MIN_JUDGED_SHARE:
            # Rates are computed over judged items only, so a batch that mostly failed
            # to return judgments can report a perfect rate off two survivors. Requiring
            # most items to come back keeps the rates from being read as a calibration
            # when they are a fragment of one.
            return False
        p, n = self.positive_rate, self.negative_rate
        if n is None or n < MIN_NEGATIVE_REJECTION_RATE:
            return False
        return p is not None and p >= MIN_POSITIVE_ACCEPTANCE_RATE

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "positive_rate": None if self.positive_rate is None else round(self.positive_rate, 4),
            "negative_rate": None if self.negative_rate is None else round(self.negative_rate, 4),
            "unjudged": self.unjudged,
            "verdicts": dict(self.verdicts),
            "usable": self.usable,
            "inspect_golden_set": (
                self.positive_rate is not None and self.positive_rate < LOW_POSITIVE_RATE_WARNING
            ),
        }


def score_agreement(items: Sequence[JudgeItem], judgments: Mapping[str, Judgment]) -> Agreement:
    """Compare judgments against the known labels of the control items."""
    from collections import Counter

    agreement = Agreement()
    counts: Counter[str] = Counter()
    for item in items:
        if not item.is_control:
            continue
        judgment = judgments.get(item.item_id)
        if judgment is None:
            agreement.unjudged += 1
            continue
        counts[judgment.verdict] += 1
        if item.control == "positive":
            agreement.n_positive += 1
            if judgment.is_supported:
                agreement.positive_supported += 1
        else:
            agreement.n_negative += 1
            # `partial` counts as rejected: the point of a negative is that the evidence
            # does not support the claim, and anything short of "supported" catches it.
            if not judgment.is_supported:
                agreement.negative_rejected += 1
    agreement.verdicts = dict(counts)
    return agreement


def batches(items: Sequence[JudgeItem], size: int = DEFAULT_BATCH_SIZE) -> list[list[JudgeItem]]:
    """Split into request-sized batches, round-robin across positives, negatives, and
    items under test.

    Controls are spread rather than grouped, so one bad response cannot wipe out the
    whole calibration while leaving the pairs under test intact.

    Round-robin explicitly, not via a sort key. The first implementation sorted on
    `(pair_id suffix, item_id)`, which degenerates *within each suffix group*: `neg::`
    sorts before `pos::`, so items sharing a suffix clump by label. Replaying that sort on
    the real 32-item control set gives `NNPPNNPPNPNN...` — a 4-positive/4-negative first
    batch, not the global grouping an earlier version of this docstring claimed (the
    shipped set has 102 distinct id suffixes, not one). The fix is right; the stated cause
    was overstated, and the Phase 6 code-review gate said so.
    """
    groups: dict[str | None, list[JudgeItem]] = {"positive": [], "negative": [], None: []}
    for item in sorted(items, key=lambda i: i.item_id):
        groups.setdefault(item.control, []).append(item)

    ordered: list[JudgeItem] = []
    depth = 0
    while len(ordered) < len(items):
        added = False
        for key in ("positive", "negative", None):
            bucket = groups.get(key) or []
            if depth < len(bucket):
                ordered.append(bucket[depth])
                added = True
        if not added:
            break
        depth += 1
    return [list(ordered[i : i + size]) for i in range(0, len(ordered), size)]


def write_agreement(
    agreement: Agreement,
    path: Any,
    items: Sequence[JudgeItem] = (),
    judgments: Mapping[str, Judgment] | None = None,
    judge_model: str = "",
    written_by: str = "",
) -> Any:
    """Persist the calibration, including every per-item verdict and reason.

    Aggregates alone are not enough. The first calibration had to be re-scored after the
    `usable` rule was corrected, and without the per-item verdicts that would have meant
    re-spending quota to recover data already paid for. It is the same lesson as storing
    claimed quotes in the generation eval: an aggregate that cannot be recomputed is an
    aggregate that has to be trusted.

    The reasons matter most. They are what distinguished "the judge is broken" from "the
    golden answers outrun their evidence" — a distinction no number in the aggregate can
    carry.
    """
    import pathlib

    judgments = judgments or {}
    payload = dict(agreement.as_dict())
    # Which model produced these verdicts, and which command wrote the file. Neither was
    # recorded, so nothing on disk distinguished a `DEFAULT_JUDGE` calibration from the
    # `make gen-eval-full` one — and because both commands write this single path, the
    # in-run 8-control calibration silently overwrote the standalone 16-control one,
    # making an earlier claim about the positive rate unverifiable from any artifact.
    payload["judge"] = judge_model or "unrecorded"
    payload["written_by"] = written_by or "unrecorded"
    payload["items"] = [
        {
            "item_id": item.item_id,
            "control": item.control,
            "source_id": item.source_id,
            "verdict": (judgments.get(item.item_id).verdict if item.item_id in judgments else None),
            "reason": (judgments.get(item.item_id).reason if item.item_id in judgments else ""),
            "claim": item.claim[:400],
            "quote": item.quote[:400],
        }
        for item in items
    ]
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out
