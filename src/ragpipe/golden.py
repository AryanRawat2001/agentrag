"""Phase 6: the golden answer set — drafting, validation, and persistence.

Phase 5 shipped the deterministic half of verification: given a quote, is it really in
the source. This phase needs the other half, and both need reference answers to score
against. That is what this module builds.

## The golden set cannot be derived from the retrieval golden set

This is the constraint that shapes everything here, and it came out of a Phase 5
finding rather than from planning. `corpus/evalset.jsonl` marks a chunk relevant when
the *identifier occurs in it* — correct for retrieval, because "find where this
regulation is mentioned" is exactly the retrieval task. It is not answerability.
`ident-0000` is the worked example: the gold span is `'21 CFR 1.980(k)'` sitting in a
footnote list of recordkeeping citations, so the document mentions the regulation and
nowhere states its requirements. A model that refuses is right, and scoring it as an
over-refusal would be the harness's error.

So questions here are drafted **from chunk content outward** — read the text, ask what
it actually supports — rather than from an identifier inward. Answerability is a
property the drafter must assert and the validator must check, not a label inherited
from a different task.

## Every golden pair is verified with the same verifier that scores the model

The drafter is required to supply verbatim `evidence` quotes for its own reference
answer, and `validate_pair` runs `citations.verify_answer` over them. A pair whose own
evidence does not locate in the cited chunk is **rejected before a human ever reads
it**.

This is the highest-leverage thing in the module. A golden set is upstream of every
generation metric, so a hallucinated reference answer does not fail loudly — it makes
a correct model look unfaithful, forever, in a table nobody re-derives. Using the
Phase 5 verifier on the gold data costs nothing (no model call, no quota) and means
the gold is held to exactly the standard the model under test is held to.

It also means the same tier-1 limitations apply, which is the honest caveat: a pair
rejected for a whitespace or line-number artifact is a *drafting* failure only in the
sense that the drafter could have quoted more carefully.

**The first calibration batch rejected 4 of 40, and all four were near-misses rather
than fabrications** — best-window word similarity 0.85-0.90 on three of them. Diagnosed
precisely, the causes were: inline superscript footnote markers, which extract as
separate digit tokens (`partners 2`, `information, 8`) and which the drafter either
joined to the neighbouring word or dropped; and one spliced quote stitched across a gap.

A second batch, re-drafted from the same four documents with the footnote rule added,
took both FDA guidances to 10/10 and flipped evidence methods from 2 exact / 34
normalized to 21 exact / 14 normalized — the quotes got markedly more faithful. But the
clinical protocol regressed to 5/10, and all five rejects had one cause: **the source
text contains spurious intra-word spaces** (`communi cable`, `requir ed`, `coll ected`,
`vi able`, `9-dimethylaminomethyl-10-hydroxycamptot hecin`) and the drafter was silently
*repairing* them, which makes the quote unlocatable.

That is a fifth text-layer defect class — the mirror image of `space_collapsed`, and a
mild relative of `character_spaced`. **It did not earn a detector.** A short-fragment
share proxy over all 152 indexable documents put the offending protocol 22nd, at 0.0843
against a median of 0.0589 and a maximum of 0.1503, so the phenomenon is a mild and
widespread property of PDF extraction rather than a separable class of bad document.
Adding a gate on a signal that does not separate would quarantine good documents. The
measurement is recorded here precisely because it failed to justify the change it was
looking for.

Both the footnote and the repair failures are fixed in the prompt, and **deliberately
not in the verifier.** A tolerance for
dropped digits is precisely what Phase 5 added and then had to withdraw: it scored a
model's "40 CFR" as a verified citation of a document's "21 CFR", because "dropped a
footnote marker" and "changed a regulation number" are the same edit. Telling the
drafter to end its quote before an ambiguous digit costs nothing and removes the
ambiguity at the source instead of tolerating it downstream.

## Batched drafting, and the quota arithmetic behind it

Plan deviation #15. The free tier allows **20 requests per day per model**, counted per
*request* against a 1M-token context — so the binding constraint is calls, not tokens.
One pair per call makes a 200-pair set a multi-day job; `DEFAULT_PAIRS_PER_CALL` pairs
grounded in one document's chunks makes it roughly 20 calls. An earlier estimate in
this project assumed one-per-call and concluded Phase 6 would take a week, which is the
same error the Phase 4 contextual-retrieval cost estimate made before grouping.

The cost of batching is real: asking for ten questions about one document tends toward
repetitive phrasing and clustered difficulty. Two mitigations, both cheap. The prompt
demands a spread of question shapes, and `sample_for_review` draws *across* documents
rather than taking the first N, so a curation pass sees the variety rather than one
document's worth of near-duplicates.

## Unanswerable pairs are constructed, never drafted

Asking a model to invent a question its own corpus cannot answer is unreliable — it
tends to produce questions the corpus answers obliquely, which are the worst possible
refusal tests because neither answering nor refusing is clearly right. The
`unanswerable` slice is built the way `evalset.py` builds its own: synthesise a
well-formed citation to a regulation that does not appear in the corpus, and verify
its absence directly. No model involved, and the ground truth is exact.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragpipe import citations as cit

#: Pairs requested per API call. See the module docstring for why this is not 1.
DEFAULT_PAIRS_PER_CALL = 10

#: Target size of the curated set. Plan deviation #5: the source guide's 50 is 10 per
#: slice across five slices, where one flipped example swings a slice by 10 points.
TARGET_PAIRS = 200

#: A reference answer shorter than this is almost always a restatement of the question
#: or a bare identifier, neither of which can be scored for completeness.
MIN_ANSWER_CHARS = 40

#: A pair whose evidence appears in more than this many documents is boilerplate, not
#: evidence about a document. Measured over the first 93 accepted pairs: 82 had evidence
#: in exactly 1 document and 2 in exactly 2, then a clean gap to 3, 6, 9, 10, 14, 30,
#: 30, 39, 39. The worst offenders are the standard nonbinding-recommendations
#: disclaimer ("the word *should* means that something is suggested", in 39 documents),
#: the "alternative approach" paragraph, and comment-submission addresses.
#:
#: Such pairs verify perfectly and measure nothing: the answer is identical across
#: dozens of documents, so retrieving the *right* document is not required to answer
#: them, and a model scores well by recognising a template. Two of the first batch were
#: literally the same question phrased twice, which is the near-duplicate risk batching
#: was always going to carry.
#:
#: The limit is 2 rather than 1 on purpose. The corpus deliberately contains 8 complete
#: draft/final pairs of the same guidance, which legitimately share long passages, so a
#: quote in exactly two documents is plausibly a real version-currency question.
MAX_EVIDENCE_DOCUMENT_SPREAD = 2

#: Two accepted pairs whose answers overlap by more than this fraction of tokens are
#: redundant: they ask the same thing of different documents, so keeping both weights
#: one template twice in every mean. Measured on the first curated batch, exactly one
#: pair exceeded it — two "where do I mail written comments" pairs whose answers were
#: the same FDA docket address, one at `Dockets Management Staff (HFA-305)` and one at
#: `Division of Dockets Management (HFA-305)`.
#:
#: This is the near-duplicate case `MAX_EVIDENCE_DOCUMENT_SPREAD` structurally cannot
#: catch. Spread asks "does this exact string appear in many documents", and slight
#: wording drift ("Room 1061" versus "rm. 1061") makes each variant look unique. The
#: redundancy is between *pairs*, so it has to be measured between pairs.
#:
#: Jaccard over lowercased word tokens rather than embeddings: the duplicates here are
#: near-identical boilerplate, not paraphrases, so the cheap measure suffices and needs
#: no model. On the shipped 158-pair pool (12,403 pairings) the duplicate scores 0.750 and
#: the next highest overlap is **0.381**, so 0.7 sits in a wide empty gap. Two earlier
#: versions of this comment got the runner-up wrong: 0.42 written without measuring it,
#: then 0.304 measured on an 84-pair population that no longer exists (0.304 is the
#: *third* highest on the current set). The gap conclusion survived all three readings;
#: the figure did not.
MAX_ANSWER_OVERLAP = 0.7

#: Documents whose lines are numbered this densely are skipped when auto-selecting
#: documents to draft from. They stay fully indexed and retrievable — this is only a
#: statement that they are poor *sources of verbatim reference quotes*.
#:
#: FDA draft guidances number their lines, and extraction interleaves those numbers
#: inside sentences. Two such documents yielded 0/10 and 1/10 accepted pairs, against
#: 7/10–10/10 for everything else, because almost every quote the drafter produced
#: silently dropped a line number and so could not be located.
#:
#: Two prompt attempts failed to fix it, and the instruction is **not** impossible: the
#: median longest digit-free run inside a chunk of the worst offender is 101 characters,
#: four times the 24-character quote floor. The model can comply and does not. Spending
#: a request to get zero pairs is waste, so those documents are skipped.
#:
#: **The threshold is bounded by observation, not measured from a gap.** The density
#: distribution is continuous (median 0.05, p90 0.51, no natural break). What is known
#: is that 0.014–0.18 drafts well and 0.57–0.65 does not; 0.18–0.57 is untested. 0.35
#: sits inside that untested range, chosen to be safely above every observed success and
#: below every observed failure. It excludes **21 of 152 documents** — an earlier version
#: of this comment said 11, which is the count at density >= 0.6415, not at 0.35. If yield
#: needs those documents, the real fix is upstream — strip line numbers during
#: extraction — which is deferred because it re-invalidates the ablation table.
MAX_LINE_NUMBER_DENSITY = 0.35

#: Words too common to signal whether an answer is grounded. Not a general stopword list
#: — it also drops regulatory filler ("provided", "including", "shall") that appears in
#: every document and would inflate coverage for free.
_COVERAGE_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "for",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "no",
        "of",
        "on",
        "or",
        "so",
        "the",
        "to",
        "up",
        "us",
        "we",
        "he",
        "she",
        "him",
        "her",
        "his",
        "my",
        "me",
        "you",
        "your",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "all",
        "any",
        "can",
        "had",
        "has",
        "have",
        "was",
        "were",
        "will",
        "with",
        "from",
        "but",
        "out",
        "per",
        "via",
        "non",
        "pre",
        "re",
        "co",
        "ex",
        "vs",
        "am",
        "pm",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "why",
        "how",
        "than",
        "then",
        "there",
        "here",
        "also",
        "such",
        "would",
        "could",
        "been",
        "being",
        "other",
        "more",
        "most",
        "some",
        "each",
        "every",
        "under",
        "over",
        "into",
        "upon",
        "within",
        "without",
        "both",
        "either",
        "neither",
        "only",
        "same",
        "including",
        "include",
        "includes",
        "included",
        "provided",
        "provide",
        "provides",
        "using",
        "used",
        "use",
        "uses",
    ]
)

#: Minimum share of an answer's content words that must appear in its evidence quotes.
#:
#: Tier 2 found that roughly a quarter of golden answers synthesise across a whole chunk
#: while citing a single span. This is a **free proxy** for that defect, validated against
#: the 16 judged positives from the first calibration:
#:
#:     judge said supported (n=12):  coverage median 0.888, min 0.471
#:     judge said partial/unsup (n=4): coverage median 0.225, max 0.611
#:
#: Best single-threshold accuracy was 0.938 at 0.471, against a 0.750 majority baseline.
#: The shipped value is **0.45, deliberately below the optimum**: it keeps every pair the
#: judge called supported and rejects 3 of the 4 it did not. Erring toward keeping pairs
#: is right because the judge is the real arbiter and this is a cheap pre-filter.
#:
#: **Limits.** n=16 is small and there is one known miss (a judge-rejected pair scored
#: 0.611). More importantly it is a **bag of words**, which bounds what it can ever catch.
#: The Phase 6 code-review gate measured the failure modes on a real-shaped quote
#: ("...shall submit the report to the Agency within 30 days..."):
#:
#:     30 -> 60 days     now 0.86  (was 1.000 -- every 1-2 digit number was invisible)
#:     shall -> may      0.86      -- one word in eight; still clears this floor
#:     shall NOT submit  0.88      -- likewise
#:     roles swapped     1.00      -- word order is not modelled at all
#:
#: So **coverage does not catch negation, modal weakening, or role reversal**, and no
#: threshold on it will. Numbers are now catchable here and are additionally checked by
#: `unsupported_numeric_claims`; the rest is what tier 2 exists for. This filter removes
#: the obvious cases cheaply — it is not a substitute for reading.
MIN_ANSWER_EVIDENCE_COVERAGE = 0.45


def _content_words(text: str) -> set[str]:
    """Tokens that carry meaning, for coverage purposes.

    **Any token containing a digit is kept regardless of length.** The length floor was
    meant to drop noise; in this corpus it dropped every one- and two-digit number, so
    "within 60 days" scored 1.000 coverage against "within 30 days" and was accepted as a
    golden pair. Quantities are the highest-value content in regulatory text and cannot be
    the thing the filter is blind to.
    """
    out: set[str] = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if any(ch.isdigit() for ch in w) or len(w) > 2 and w not in _COVERAGE_STOPWORDS:
            out.add(w)
    return out


def answer_evidence_coverage(answer: str, quotes: Sequence[str]) -> float:
    """Share of the answer's content words that appear in its evidence quotes.

    1.0 for an answer with no content words at all, so a degenerate answer is caught by
    the length floor rather than scoring 0 here and being reported as ungrounded.
    """
    wanted = _content_words(answer)
    if not wanted:
        return 1.0
    have = _content_words(" ".join(quotes))
    return len(wanted & have) / len(wanted)


_LINE_NUMBER_LINE = re.compile(r"^\s*\d+\s*$")
_TRAILING_NUMBER = re.compile(r"\s\d+\s*$")
#: Numbers at the *start* of a line -- the commoner FDA draft-guidance layout, and
#: invisible to the two patterns above. `fda-71536` is unambiguously line-numbered
#: ("1 This guidance has been prepared by...", ~45% of its lines) and scored 0.255, so it
#: was admitted as a drafting source. The test fixture put its numbers at line *ends*,
#: which is why this went unexercised.
_LEADING_NUMBER = re.compile(r"^\s*\d+\s")


def line_number_density(text: str) -> float:
    """Share of lines that are a bare number or end in one.

    A proxy for "is this document typeset with line numbers", not a defect metric:
    ordinary documents score 0.05 because page numbers and table cells also end in
    digits. Returns 0.0 for text with too few lines to judge.
    """
    lines = text.split("\n")
    if len(lines) < 10:
        return 0.0
    hits = sum(
        1
        for ln in lines
        if _LINE_NUMBER_LINE.match(ln) or _TRAILING_NUMBER.search(ln) or _LEADING_NUMBER.match(ln)
    )
    return hits / len(lines)


STATUS_DRAFT = "draft"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"

#: Question shapes the drafter is asked to spread across. Deliberately not the
#: retrieval slice names: those describe how a *query* is phrased for a retriever,
#: whereas these describe what an answer has to do, which is what generation is scored
#: on. Keeping the two vocabularies separate is what stopped Phase 5 from reading
#: `exact_identifier` as an answerability label.
QUESTION_SHAPES = (
    "requirement",  # what does this document require of whom
    "definition",  # what does this document say a term means
    "procedure",  # what steps does it prescribe, in order
    "condition",  # under what circumstances does something apply
    "quantity",  # a threshold, deadline, or numeric limit stated in the text
    "scope",  # what the document says it does and does not cover
)

#: Character budget for the excerpts handed to one drafting call. Not a context limit
#: — the largest document in this corpus renders to ~870,000 characters, about 217k
#: tokens against a 1,048,576-token window, so everything fits. It is an *attention*
#: budget. Chunks per document run median 46, p90 234, max 765; handing a model 765
#: excerpts and asking for ten questions reliably anchors it on the opening few, which
#: is the same clustering problem batching already risks. 40,000 characters is roughly
#: 39 chunks, so the median document is passed whole and only the long tail is sampled.
DRAFT_CHAR_BUDGET = 40_000


def select_chunks_for_drafting(
    chunks: Sequence[Mapping[str, Any]], budget: int = DRAFT_CHAR_BUDGET
) -> list[Mapping[str, Any]]:
    """Take an evenly-spread subset of a document's chunks within a character budget.

    Stride sampling rather than a prefix: a prefix of a 765-chunk protocol is its title
    page and table of contents, which supports almost no answerable question. Striding
    keeps document order (so the excerpts still read coherently) while reaching the
    middle and end.

    Deterministic by construction — no RNG — so a re-draft of the same document sees
    the same excerpts and any change in output is attributable to the prompt or the
    model rather than to sampling.
    """
    if not chunks:
        return []
    total = sum(len(c.get("text") or "") for c in chunks)
    if total <= budget:
        return list(chunks)
    # Ceiling division: stride 2 keeps every other chunk, 3 keeps every third, and so
    # on. Recomputed against the realised total afterwards, because chunk lengths vary
    # and a stride chosen from the mean can still overshoot.
    stride = max(2, -(-total // budget))
    picked = list(chunks[::stride])
    # Keep at least one chunk below. The trim loop pops from the tail until the budget is
    # met, so a single chunk larger than the budget emptied the list entirely — and
    # `build_draft_prompt` would then render zero excerpts and still ask for ten pairs,
    # spending a request for certain zero yield. Latent on this corpus (largest chunk is
    # 1,461 characters against a 40,000 budget) and cheap to close.
    while len(picked) > 1 and sum(len(c.get("text") or "") for c in picked) > budget:
        picked.pop()
    return picked


GOLDEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "shape": {"type": "string", "enum": list(QUESTION_SHAPES)},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "chunk_id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["chunk_id", "quote"],
                            "propertyOrdering": ["chunk_id", "quote"],
                        },
                    },
                },
                "required": ["shape", "question", "answer", "evidence"],
                "propertyOrdering": ["shape", "question", "answer", "evidence"],
            },
        }
    },
    "required": ["pairs"],
    "propertyOrdering": ["pairs"],
}

DRAFT_SYSTEM_PROMPT = """\
You draft evaluation questions for a retrieval system over regulatory and \
clinical-trial documents. Your output becomes reference data, so it is held to a \
higher standard than an answer would be: a question whose answer is not fully stated \
in the excerpts is worse than no question at all.

Rules:

1. Ask only what the excerpts actually answer. Do not ask what a regulation requires \
when the excerpt merely *cites* that regulation — a passing mention is not a \
statement of its content, and questions of that kind are the single most common defect \
in sets like this.
2. The "answer" must be fully supported by the excerpts, self-contained, and at least \
{min_answer} characters. Do not restate the question. Do not add outside knowledge.
3. "evidence" must **cover the whole answer**. Supply as many quotes as it takes: if \
your answer states three things, quote the passage supporting each. An answer that \
synthesises across the excerpt while citing one span of it is the most common defect in \
sets like this — roughly a quarter of a previous batch failed on exactly that, with \
reviewers noting "the quote does not mention" whatever the answer had added. If you \
cannot cover a claim with a quote, remove the claim from the answer.
4. Each quote must be copied CHARACTER FOR CHARACTER from the \
excerpt it cites, of at least {min_quote} characters. An automated check locates every \
quote in the source text and discards the pair if it is not found, so paraphrasing here \
wastes the pair. You may collapse runs of whitespace and line breaks, nothing else.
   In particular: **do not repair the text.** PDF extraction sometimes breaks a word \
with a stray space — "communi cable", "requir ed", "coll ected", "vi able". Copy it \
exactly as shown, spaces and all, or choose a different span. Silently fixing it to \
"communicable" makes the quote unlocatable, and this is the single largest source of \
discarded pairs on clinical protocols.
5. "chunk_id" must be copied exactly from the [chunk_id: ...] label the quote came from.
6. Each quote must be ONE CONTIGUOUS span of a single excerpt. Do not join text from \
either side of a gap, and do not stitch two sentences together with words of your own.
7. These excerpts are extracted from PDFs, so **stray digits appear inside sentences**, \
from two sources. Superscript footnote markers land after a word — "trading partners 2 \
must", "information, 8 which". And FDA draft guidances number their lines, so a number \
lands wherever a line broke — "the reference product and also to 107 108 demonstrate", \
"reviewed by FDA staff from more than 77 one center".
   Do not attach such a digit to a neighbouring word, and **do not silently drop it**. \
Either reproduce it exactly where it appears, or — easier and always safe — **end the \
quote before it.** A shorter span that stops short of the digit locates every time; \
guessing at it is by far the largest source of discarded pairs, and on line-numbered \
draft guidances it can discard nearly every pair from a document.
8. Spread the questions across different "shape" values and different excerpts. Do not \
ask {n} variations of one question, and do not draw every question from the same excerpt.
9. Write questions someone would plausibly ask without already knowing the answer. \
Avoid questions that quote the answer back, and avoid "according to the excerpt" \
phrasing — the question should stand on its own.
"""


@dataclass(frozen=True, slots=True)
class GoldenPair:
    """One reference Q&A pair, with its own evidence held to the tier-1 standard."""

    pair_id: str
    question: str
    answer: str
    evidence: tuple[dict[str, str], ...]
    doc_id: str
    shape: str
    answerable: bool = True
    status: str = STATUS_DRAFT
    reject_reason: str = ""
    #: Per-evidence verification methods from `citations`, for auditability.
    evidence_methods: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "question": self.question,
            "answer": self.answer,
            "evidence": [dict(e) for e in self.evidence],
            "doc_id": self.doc_id,
            "shape": self.shape,
            "answerable": self.answerable,
            "status": self.status,
            "reject_reason": self.reject_reason,
            "evidence_methods": list(self.evidence_methods),
        }

    @property
    def accepted(self) -> bool:
        return self.status == STATUS_ACCEPTED


def draft_system_prompt(
    n: int = DEFAULT_PAIRS_PER_CALL,
    min_answer_chars: int = MIN_ANSWER_CHARS,
    min_quote_chars: int = cit.MIN_QUOTE_CHARS,
) -> str:
    """The drafting instructions, with both floors interpolated from their constants.

    Interpolated rather than written twice: a prompt asking for a 20-character quote
    while the validator requires 24 would manufacture rejections that look like model
    failures. Phase 5 made the same mistake in the opposite direction and fixed it the
    same way.
    """
    return DRAFT_SYSTEM_PROMPT.format(n=n, min_answer=min_answer_chars, min_quote=min_quote_chars)


def build_draft_prompt(doc_id: str, chunks: Sequence[Mapping[str, Any]], n: int) -> str:
    """Render one document's chunks as labelled excerpts and ask for `n` pairs.

    Uses `chunk["text"]`, never `embed_text`, for the same reason `answer.py` does:
    the validator searches `text`, so the drafter must be shown `text` or its quotes
    will be located against a string it never saw.
    """
    from ragpipe.answer import render_context

    selected = select_chunks_for_drafting(chunks)
    note = (
        ""
        if len(selected) == len(chunks)
        else f" ({len(selected)} of {len(chunks)} excerpts, sampled evenly across the document)"
    )
    return (
        f"Document: {doc_id}{note}\n\nExcerpts:\n\n{render_context(selected)}\n\n"
        f"Draft exactly {n} question-answer pairs grounded in these excerpts, "
        "following every rule."
    )


def parse_pairs(
    payload: Mapping[str, Any], doc_id: str, *, start_index: int = 0
) -> list[GoldenPair]:
    """Turn a drafting response into `GoldenPair`s. No validation here.

    Malformed entries are skipped rather than raised on: a batch of ten where one
    entry is unusable should yield nine pairs, not zero. The count reaching
    `validate_pairs` is what the report divides by, so silent loss is visible as a
    yield figure rather than hidden.
    """
    out: list[GoldenPair] = []
    raw_pairs = payload.get("pairs") or []
    for i, raw in enumerate(raw_pairs):
        if not isinstance(raw, Mapping):
            continue
        evidence = tuple(
            {"chunk_id": str(e.get("chunk_id", "")), "quote": str(e.get("quote", ""))}
            for e in (raw.get("evidence") or [])
            if isinstance(e, Mapping)
        )
        out.append(
            GoldenPair(
                pair_id=f"{doc_id}::gold::{start_index + i:03d}",
                question=str(raw.get("question") or "").strip(),
                answer=str(raw.get("answer") or "").strip(),
                evidence=evidence,
                doc_id=doc_id,
                shape=str(raw.get("shape") or "unknown"),
            )
        )
    return out


def build_document_index(chunks: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """doc_id -> whitespace-normalised full text, for boilerplate detection.

    Normalised once here rather than per-check: the corpus is 13,423 chunks and the
    spread test runs for every drafted pair, so normalising inside the loop turned a
    validation pass into a minute of string work.
    """
    per_doc: dict[str, list[str]] = {}
    for c in chunks:
        per_doc.setdefault(c.get("doc_id", ""), []).append(
            normalize_whitespace(c.get("text") or "")[0]
        )
    return {d: "\n".join(v) for d, v in per_doc.items()}


def evidence_document_spread(quote: str, doc_index: Mapping[str, str]) -> int:
    """How many documents contain this quote, ignoring whitespace differences.

    `doc_index` values must **already be whitespace-normalised** — build it with
    `build_document_index`. Only the needle is normalised here, because the haystack is
    the whole corpus and normalising it per call would dominate the validation pass.
    """
    needle = normalize_whitespace(quote)[0]
    if len(needle) < cit.MIN_QUOTE_CHARS:
        return 0
    return sum(1 for text in doc_index.values() if needle in text)


def validate_pair(
    pair: GoldenPair,
    chunks: Sequence[Mapping[str, Any]],
    *,
    min_answer_chars: int = MIN_ANSWER_CHARS,
    doc_index: Mapping[str, str] | None = None,
    max_spread: int = MAX_EVIDENCE_DOCUMENT_SPREAD,
    min_coverage: float = MIN_ANSWER_EVIDENCE_COVERAGE,
) -> GoldenPair:
    """Accept or reject one drafted pair. Deterministic, no model call.

    Checks run cheapest-first, and each names its own reason so the reject log is a
    diagnosis rather than a count. The evidence check is the one that matters: it runs
    the Phase 5 verifier, so a reference answer whose own quote cannot be located is
    discarded before a human spends attention on it.
    """
    if not pair.question:
        return _reject(pair, "empty question")
    if not pair.answer:
        return _reject(pair, "empty answer")
    if len(pair.answer) < min_answer_chars:
        return _reject(pair, f"answer under {min_answer_chars} chars ({len(pair.answer)})")
    if pair.question.strip().lower() == pair.answer.strip().lower():
        return _reject(pair, "answer restates the question")
    if not pair.evidence:
        return _reject(pair, "no evidence supplied")

    report = cit.verify_answer(pair.evidence, chunks)
    methods = tuple(c.method for c in report.checks)
    if report.n_verified == 0:
        return _reject(pair, f"no evidence verified (methods: {', '.join(methods)})", methods)
    coverage = answer_evidence_coverage(pair.answer, [e["quote"] for e in pair.evidence])
    if coverage < min_coverage:
        return _reject(
            pair,
            f"answer outruns its evidence: {coverage:.0%} of content words covered",
            methods,
        )
    if report.n_verified < report.n_claimed:
        # Checked *before* the spread rule. A pair with one boilerplate quote and one
        # fabricated quote was logged as "boilerplate", the less informative diagnosis,
        # which also miscategorises the yield accounting. Fabrication is what the drafting
        # prompt targets, so it is what the reject reason should name.
        return _reject(
            pair,
            f"only {report.n_verified}/{report.n_claimed} evidence quotes verified "
            f"(methods: {', '.join(methods)})",
            methods,
        )
    if doc_index is not None:
        spread = max(
            (evidence_document_spread(e["quote"], doc_index) for e in pair.evidence),
            default=0,
        )
        if spread > max_spread:
            return _reject(pair, f"boilerplate: evidence appears in {spread} documents", methods)
    if report.n_verified < report.n_claimed:
        # A partially-verified pair is not salvaged by dropping the bad quote: the
        # reference answer may depend on the span that does not exist.
        return _reject(
            pair,
            f"only {report.n_verified}/{report.n_claimed} evidence quotes verified "
            f"(methods: {', '.join(methods)})",
            methods,
        )
    return GoldenPair(
        **{
            **{
                k: v
                for k, v in pair.as_dict().items()
                if k not in {"evidence", "status", "reject_reason", "evidence_methods"}
            },
            "evidence": pair.evidence,
            "status": STATUS_ACCEPTED,
            "reject_reason": "",
            "evidence_methods": methods,
        }
    )


def _reject(pair: GoldenPair, reason: str, methods: tuple[str, ...] = ()) -> GoldenPair:
    return GoldenPair(
        **{
            **{
                k: v
                for k, v in pair.as_dict().items()
                if k not in {"evidence", "status", "reject_reason", "evidence_methods"}
            },
            "evidence": pair.evidence,
            "status": STATUS_REJECTED,
            "reject_reason": reason,
            "evidence_methods": methods,
        }
    )


def validate_pairs(
    pairs: Iterable[GoldenPair],
    chunks_by_doc: Mapping[str, Sequence[Mapping[str, Any]]],
    **kwargs: Any,
) -> list[GoldenPair]:
    """Validate a batch, scoping each pair's evidence to its own document's chunks."""
    return [validate_pair(p, chunks_by_doc.get(p.doc_id, []), **kwargs) for p in pairs]


def normalize_whitespace(text: str) -> tuple[str, list[int]]:
    """Re-exported from `citations` so this module has one obvious import for it."""
    return cit.normalize_whitespace(text)


#: Numbers and identifiers asserted in an answer but absent from the cited source are
#: the highest-value factual error to catch: regulatory answers are dense with them
#: ("within 30 days", "21 CFR 507", "4 parts per million"), a wrong one is materially
#: wrong rather than stylistically off, and unlike prose entailment it is checkable
#: without a model.
#:
#: Support is checked against the whole cited **chunk**, not just the evidence quote. A
#: drafter legitimately summarises more of a chunk than it quotes, so requiring every
#: number to appear inside the quote would flag correct answers in bulk. A number absent
#: from the entire chunk is a much stronger signal.
#:
#: Written-out numerals are resolved for the small cases that actually occur, because
#: "three years" supporting "3 years" is a real and common paraphrase rather than an
#: error.
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "fifteen": "15",
    "twenty": "20",
    "thirty": "30",
    "sixty": "60",
    "ninety": "90",
}
_CLAIM_RE = re.compile(r"\d[\d,\.]*")


def _numeric_claims(text: str) -> set[str]:
    """Digit strings in `text`, plus digits implied by written-out numerals.

    Commas and trailing periods are stripped so "5,630" and "5630." compare equal to
    "5630" -- PDF text and generated prose disagree about thousands separators
    constantly, and treating that as a factual difference would be noise.
    """
    found = {m.group(0).replace(",", "").rstrip(".") for m in _CLAIM_RE.finditer(text)}
    for word, digit in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text, re.I):
            found.add(digit)
    # "a year" / "an hour" mean one of the unit. The only flag this check raised on the
    # first 83 pairs was exactly this: a source reading "kept for a year" against an
    # answer reading "retained for 1 year" -- the same fact, scored as a discrepancy.
    # Restricted to time units so it cannot fire on "a supplier" or "an applicant".
    # "a year", "each year", "every year", "per year" and "annually" all mean one of the
    # unit. Two of the three flags this check has raised on real data were of exactly this
    # shape -- "once a year" against "once each year", and "for 1 year" against "for a
    # year" -- identical facts scored as discrepancies. Scoped to time units so it cannot
    # fire on "a supplier".
    #
    # Known limits, since the surrounding text used to overstate this: only the 19 words in
    # `_NUMBER_WORDS` resolve, so "forty-five"/45 and "third"/3rd still flag; and decimal
    # or percent formatting is not normalised, so 4/4.0 and 20%/20.0 percent flag too. The
    # injection below is also coarse: a source mentioning any time unit contributes a bare
    # "1" to the supported set, so a stray "1" in an answer (Phase 1, section 1) is
    # auto-supported. Measured on the shipped set: 9 of 157 sources trigger it and it is
    # load-bearing for exactly one pair.
    if re.search(
        r"\b(a|an|each|every|per)\s+(year|month|week|day|hour|decade)s?\b|\bannually\b",
        text,
        re.I,
    ):
        found.add("1")
    return {f for f in found if f}


def unsupported_numeric_claims(answer: str, source: str) -> list[str]:
    """Numbers asserted in `answer` that appear nowhere in `source`.

    `source` should be the full text of every chunk the pair cites. Returns the
    offending strings so a reviewer sees what to check rather than a boolean.
    """
    supported = _numeric_claims(source)
    # A written-out numeral in the source supports a digit in the answer and vice versa,
    # so both sides go through the same resolver.
    return sorted(c for c in _numeric_claims(answer) if c not in supported)


def flag_factual_risk(
    pairs: Sequence[GoldenPair], chunk_index: Mapping[str, Mapping[str, Any]]
) -> list[tuple[GoldenPair, list[str]]]:
    """Accepted pairs whose answers assert numbers absent from their cited chunks.

    This is a *triage* signal, not a verdict: a flagged pair may be paraphrasing a
    quantity the source states in words the resolver does not know. It exists so a human
    curation pass can look at ten pairs instead of two hundred.
    """
    out: list[tuple[GoldenPair, list[str]]] = []
    for pair in pairs:
        if not pair.accepted:
            continue
        source = " ".join(
            (chunk_index.get(e["chunk_id"], {}) or {}).get("text") or "" for e in pair.evidence
        )
        missing = unsupported_numeric_claims(pair.answer, source)
        if missing:
            out.append((pair, missing))
    return out


def answer_overlap(a: str, b: str) -> float:
    """Jaccard similarity of two answers over lowercased word tokens."""
    ta = {w for w in a.lower().split() if len(w) > 2}
    tb = {w for w in b.lower().split() if len(w) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def drop_redundant(
    pairs: Sequence[GoldenPair], max_overlap: float = MAX_ANSWER_OVERLAP
) -> list[GoldenPair]:
    """Reject accepted pairs whose answer duplicates an earlier accepted one.

    Order is by `pair_id`, so which of a duplicate group survives is deterministic given
    the same ids. Rejected rather than deleted, so the reject log still shows what was
    removed and why.

    **The tie-break is deterministic; the surviving *count* is not, when similarity is
    non-transitive.** With A~B, B~C and A/C below the threshold, two pairs survive if A or
    C sorts first and one survives if B does. Ids come from a counter that accumulates
    across documents, so re-drafting in a different order — or with one document erroring —
    can change them and therefore the outcome. An earlier version of this docstring said
    the result was independent of drafting order, which is true of the tie-break and false
    of the set size.
    """
    out: list[GoldenPair] = []
    kept: list[GoldenPair] = []
    for pair in sorted(pairs, key=lambda p: p.pair_id):
        if not pair.accepted:
            out.append(pair)
            continue
        dupe = next((k for k in kept if answer_overlap(pair.answer, k.answer) > max_overlap), None)
        if dupe is not None:
            out.append(
                _reject(pair, f"redundant: answer duplicates {dupe.pair_id}", pair.evidence_methods)
            )
        else:
            kept.append(pair)
            out.append(pair)
    return out


def sample_for_review(pairs: Sequence[GoldenPair], n: int) -> list[GoldenPair]:
    """Draw `n` accepted pairs spread across documents, round-robin.

    Not the first `n`: batched drafting groups by document, so a prefix is one or two
    documents' worth of near-duplicates. A curation pass needs to see the variety it is
    judging, and round-robin is deterministic, which a random sample would not be.
    """
    by_doc: dict[str, list[GoldenPair]] = {}
    for p in pairs:
        if p.accepted:
            by_doc.setdefault(p.doc_id, []).append(p)
    out: list[GoldenPair] = []
    depth = 0
    while len(out) < n:
        added = False
        for doc in sorted(by_doc):
            bucket = by_doc[doc]
            if depth < len(bucket):
                out.append(bucket[depth])
                added = True
                if len(out) >= n:
                    break
        if not added:
            break
        depth += 1
    return out


def yield_report(pairs: Sequence[GoldenPair]) -> dict[str, Any]:
    """Acceptance yield and the reason breakdown for everything discarded."""
    from collections import Counter

    accepted = [p for p in pairs if p.accepted]
    rejected = [p for p in pairs if p.status == STATUS_REJECTED]
    # Reason strings carry counts, so bucket on the leading phrase to keep the
    # breakdown readable without discarding the detail on the pair itself.
    reasons = Counter(p.reject_reason.split(" (")[0] for p in rejected)
    methods = Counter(m for p in accepted for m in p.evidence_methods)
    return {
        "n_drafted": len(pairs),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "acceptance_rate": round(len(accepted) / len(pairs), 4) if pairs else None,
        "reject_reasons": dict(reasons.most_common()),
        "accepted_evidence_methods": dict(methods.most_common()),
        "shapes": dict(Counter(p.shape for p in accepted).most_common()),
        "documents": len({p.doc_id for p in accepted}),
    }


def write_golden(pairs: Sequence[GoldenPair], path: Path) -> Path:
    """Persist every pair, accepted and rejected alike.

    Rejects are kept on disk deliberately. The reject log is the record of what the
    drafter got wrong, which is the input to improving the prompt; discarding it would
    make the acceptance rate an unexplainable number.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in sorted(pairs, key=lambda x: x.pair_id):
            fh.write(json.dumps(p.as_dict(), sort_keys=True) + "\n")
    return path


def load_golden(path: Path) -> list[GoldenPair]:
    if not path.exists():
        return []
    out = []
    # Iterate the file, not `read_text().splitlines()`. `str.splitlines()` also splits on
    # U+2028, U+2029, form feed and friends, none of which terminate a JSONL record — and
    # this corpus contains six literal U+2028 characters, so the same pattern applied to
    # `data/chunks/structural.jsonl` yields 13,429 "lines" for 13,423 records and tears six
    # of them mid-JSON. Safe here today only because `write_golden` uses `json.dumps` with
    # `ensure_ascii=True`, which escapes them; that is a property of the writer, not of the
    # reader, and the reader should not depend on it.
    with path.open(encoding="utf-8") as fh:
        lines = list(fh)
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(
            GoldenPair(
                pair_id=d["pair_id"],
                question=d["question"],
                answer=d["answer"],
                evidence=tuple(d["evidence"]),
                doc_id=d["doc_id"],
                shape=d["shape"],
                answerable=d.get("answerable", True),
                status=d.get("status", STATUS_DRAFT),
                reject_reason=d.get("reject_reason", ""),
                evidence_methods=tuple(d.get("evidence_methods") or ()),
            )
        )
    return out
