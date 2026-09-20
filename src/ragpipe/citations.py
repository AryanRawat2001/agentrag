"""Tier-1 citation verification: locating a quoted span in its source text.

A grounded answer is only as good as the check that its citations are real. This
module is that check, and it is deliberately the *deterministic* half: no model is
consulted here. Given a claimed quote and a chunk id, either the text exists in the
source at a locatable offset or it does not.

## Why we locate quotes instead of trusting offsets

The original plan (deviation #3) used Claude's native citations, which return
`cited_text` plus character offsets into the supplied document. Tier 1 was then a
span check: does `document[start:end]` equal the quote? With no Claude access
available, tier 1 instead *locates* the quote by searching the source (deviation
#14).

This is strictly stronger. Nothing the model reports about position is trusted —
position is recovered independently. It also works with any generator, which is
what makes "citation-verification failure rate by model" a measurable row rather
than a property of one vendor's API.

The cost is real and shows up as a higher failure rate: a model not trained to emit
verbatim spans paraphrases, and a paraphrase is correctly unverifiable. That is the
number to report, not a number to engineer away.

## Why matching is tiered rather than exact

Exact `str.find` is the honest first attempt, but on this corpus it under-reports
badly. PDF text layers carry whitespace artifacts that no model reproduces when
quoting — a real chunk in `structural.jsonl` opens:

    'ANBL0532 \\nPage 1 \\n \\nActivated: 11/05/07      Version Date: 08/16/11 \\n'

A model quoting that passage will collapse the runs. Rejecting it as fabricated
would be wrong, and would make the headline failure rate a measurement of pypdf's
spacing rather than of the model's grounding. So a whitespace-insensitive tier
follows, and it maps back to **real offsets in the original text** so the span
stays auditable. A match that cannot be expressed as an offset range into the
untouched source is not accepted at all.

**Normalization that counts as verification stops at whitespace.** Case, punctuation,
unicode folding, and digits are deliberately *not* normalized: those change what the
text says, and a citation verifier that tolerates changes in meaning is theatre. A
third tier does normalize further, but its hits are reported as a *failure* sub-type
rather than as verification — see `LINE_NUMBER_AMBIGUOUS`. This distinction was
briefly lost and the code-review gate caught it.

## The six outcomes, and why near-misses get their own names

A boolean verified/unverified conflates two very different model failures. Both are
reported distinctly:

    exact         found in the cited chunk, byte-identical
    normalized    found in the cited chunk, modulo whitespace runs
    line_number_ambiguous
                  failed strict verification, but *would* match if interleaved
                  legislative line numbers were discounted. **Not verified** -- that
                  tolerance was measured to bless meaning changes (a quote saying
                  "40 CFR" where the document says "21 CFR"). Diagnostic hint only
    wrong_chunk   found in the cited *document*, but not in the chunk cited
    unverified    not found in the document at all
    too_short     quote below MIN_QUOTE_CHARS; not evidence of anything

`wrong_chunk` is a misattribution — the model read real text and pointed at the
wrong provenance. `unverified` is a fabricated or paraphrased quote. Lumping them
together hides the fact that the first is a retrieval/plumbing bug and the second
is a grounding failure; they have different fixes. Phase 8's failure taxonomy needs
them apart.

`too_short` exists because a short quote verifies trivially. The string `"the"`
appears in nearly every chunk, so accepting it would inflate citation precision
without any evidence having been cited. It is reported as its own bucket so the cause
is visible, but note it **does** count against `citation_precision` — it sits in the
denominator and not the numerator, exactly like `unverified`. An earlier version of
this paragraph claimed it counted as neither, which the code never did.

## Known limitations

- A quote spanning two chunks cannot match either, and lands in `wrong_chunk` (if
  the parent document is available) or `unverified`. Real, and counted.
- Ligatures, soft hyphens, and hyphenation across a line break are not normalized,
  so a quote that crosses a hyphenated line break will fail. Checked on the chunks
  behind the first observed all-citations-failed query and found absent there (zero
  hyphenated line breaks, zero ligatures), so it remains an unmeasured hazard rather
  than a known cost. The line-number tier below is what that investigation actually
  turned up, which is the general lesson: each of these tiers should be added because
  a measurement demanded it, not because it seemed plausible.
- The `line_number_ambiguous` bucket is a *hypothesis*, not a verdict. Because 53.2%
  of chunks contain line-adjacent digits, a fabricated quote can land in it by
  coincidence. Read it as "check extraction first", never as "the model was right".
- `MIN_QUOTE_CHARS` is a judgement, not a measurement. It is set below.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# A quote shorter than this is not treated as evidence. 24 characters is roughly
# four or five words: long enough that an exact hit in a 1,024-character chunk is
# unlikely to be coincidental, short enough to admit a genuine terse citation such
# as "Predetermined Change Control Plan" (33). This is a judgement call, and the
# `too_short` bucket exists so its cost is visible rather than absorbed.
MIN_QUOTE_CHARS = 24

EXACT = "exact"
NORMALIZED = "normalized"
LINE_NUMBER_AMBIGUOUS = "line_number_ambiguous"
WRONG_CHUNK = "wrong_chunk"
UNVERIFIED = "unverified"
TOO_SHORT = "too_short"

#: Outcomes that count as a citation actually supporting provenance. Strictly two:
#: byte-identical, or identical modulo whitespace. Nothing that can alter a token is in
#: here. See the `LINE_NUMBER_AMBIGUOUS` note below for why a third tier was demoted.
VERIFIED_METHODS = frozenset({EXACT, NORMALIZED})

# FDA draft guidances are typeset with legislative line numbers, and pypdf extracts
# them *inside* the sentence they sit beside:
#
#   'ropic virus, Treponema 28 \npallidum (syphilis), vaccinia virus, and ...'
#   '...to assist you, establishments making donor 17 \neligibility determinations,1'
#
# A model asked to quote that passage writes "Treponema pallidum (syphilis)", which is
# the correct reading of the sentence and is not findable in the source under any
# amount of whitespace normalization -- `28` is a token, not whitespace. Phase 4 had
# already noticed this artifact while piloting prompts, as a hypothesis about BM25 false
# matches; its figures came from a hand count with no rule in code and do not reproduce,
# so they are not restated here. Its real cost turned out to be here: in an early Phase
# 5 run one `title_lookup` query had all of its citations scored `unverified` for this
# reason alone. (That run's artifact is not committed, so the per-slice figures an
# earlier version of this comment quoted are unverifiable and have been removed --
# the shipped run's slice precisions are not the same numbers.)
#
# **This tier was briefly counted as verified. That was wrong, and it is now a
# diagnostic-only failure bucket.** The Phase 5 code-review gate demonstrated on real
# corpus text that discounting line-adjacent integers blesses quotes which change what
# the document says:
#
#   source: 'subject to 21 CFR part 123, Fish and Fishery Products or 21 \nCFR part 120'
#   quote:  'subject to 21 CFR part 123, Fish and Fishery Products or 40 \nCFR part 120'
#   -> matched, because the line-adjacent "21" is deleted from the source and the
#      line-adjacent "40" from the quote, so both reduce to "or CFR part 120"
#
# A model asserting **40 CFR** where the regulation is **21 CFR** scored as a verified
# citation. So did a quote that *omits* a stated number: source "must respond within
# 30 \ndays", quote "must respond within days".
#
# The premise that failed was "a line number by construction sits alone at a line
# boundary". pypdf splits real identifiers across line breaks -- Phase 1 recorded this
# at 63 occurrences (`21 CFR 812.2(c)` extracting as `21 CFR 71 \n812.2(c)`) -- so
# content numbers sit at line boundaries too. Measured exposure on the current corpus:
# **7,141 of 13,423 chunks (53.2%) lose digits** under this normalization, 86,231 digit
# characters
# corpus-wide, and 16 of the 22 quotes in the shipped run contain a literal newline, so
# the precondition is ordinary behaviour and not an edge case.
#
# It cannot be repaired by local pattern matching, because "the model dropped a line
# number" and "the model dropped or altered a content number" are structurally
# identical. Separating them needs document-level evidence that a line-numbering scheme
# exists (a monotonic sequence with unit increments). Deferred, with the measurement
# above as the reason to do it properly rather than cheaply.
#
# What survives is the diagnostic: a citation here failed strict verification but would
# match if line numbers were discounted, which makes an extraction artifact the leading
# hypothesis over fabrication. Useful for triage. **Not evidence of provenance, and not
# counted as verified.**
_LINE_NUM_PATTERNS = (
    r"(?m)^[ \t]*\d+[ \t]*$",  # a line that is only a number
    r"(?<=\s)\d+[ \t]*(?=\n)",  # a number ending a line
    r"(?<=\n)[ \t]*\d+(?=\s)",  # a number starting a line
)


def normalize_whitespace(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to single spaces, keeping a map back to `text`.

    Returns `(normalized, index_map)` where, **for every non-space character**,
    `index_map[i]` is the index in the original `text` of `normalized[i]`. Synthesized
    separators are the exception: their entry points at the collapsed run rather than
    at an identical character, so the contract holds for real characters only. That is
    sufficient, because a normalized quote never begins or ends on a space, so no
    reported offset is ever taken from a synthesized entry.

    The map is what makes the normalized tier usable: a match found in `normalized` can
    be reported as an offset range into the untouched source, so every accepted citation
    still names a real span.

    Leading and trailing whitespace is dropped rather than collapsed, so the map
    never begins or ends on a synthesized space.
    """
    chars: list[str] = []
    index_map: list[int] = []
    in_space = False

    for i, ch in enumerate(text):
        if ch.isspace():
            # Collapse the run to one space, anchored at the run's first character.
            # Nothing is emitted for leading whitespace, so `chars` stays empty
            # until the first real character.
            if not in_space and chars:
                chars.append(" ")
                index_map.append(i)
            in_space = True
            continue
        chars.append(ch)
        index_map.append(i)
        in_space = False

    # A trailing collapsed space would let a match end on a synthesized character,
    # whose `index_map` entry points at the *start* of the original run — which
    # would under-cover the span when converted back to offsets.
    while chars and chars[-1] == " ":
        chars.pop()
        index_map.pop()

    return "".join(chars), index_map


def normalize_line_numbers(text: str) -> tuple[str, list[int]]:
    """Drop line-adjacent bare integers, then collapse whitespace.

    Returns `(normalized, index_map)` with the same contract as
    `normalize_whitespace`: `index_map[i]` is the index in the original `text` of
    `normalized[i]`, so a match can still be reported as a span of untouched source.

    Composes with the whitespace pass rather than replacing it -- a dropped line
    number leaves whitespace on both sides, and that whitespace has to collapse for
    the surrounding words to become adjacent.
    """
    keep = [True] * len(text)
    for pattern in _LINE_NUM_PATTERNS:
        for match in re.finditer(pattern, text):
            for i in range(match.start(), match.end()):
                keep[i] = False

    chars: list[str] = []
    index_map: list[int] = []
    in_space = False
    for i, ch in enumerate(text):
        # A removed line number is treated as whitespace, so the words either side of
        # it can join -- otherwise "Treponema 28 \npallidum" would close up into
        # "Treponemapallidum" and match nothing.
        if not keep[i] or ch.isspace():
            in_space = True
            continue
        # The separator is emitted here, on the transition back to real text, and
        # nowhere else. Emitting it in the whitespace branch *as well* produced two
        # spaces per gap and made every match fail.
        #
        # It anchors on the following character rather than the run's start, unlike
        # `normalize_whitespace`. Harmless: a normalized quote never begins or ends on
        # a space, so no reported offset is ever taken from a synthesized character.
        if in_space and chars:
            chars.append(" ")
            index_map.append(i)
        chars.append(ch)
        index_map.append(i)
        in_space = False

    while chars and chars[-1] == " ":
        chars.pop()
        index_map.pop()
    return "".join(chars), index_map


@dataclass(frozen=True, slots=True)
class SpanMatch:
    """A located quote. `start`/`end` are offsets into the searched source text."""

    start: int
    end: int
    method: str
    matched_text: str

    @property
    def is_exact(self) -> bool:
        return self.method == EXACT


def locate_quote(quote: str, source: str) -> SpanMatch | None:
    """Find `quote` in `source`, exactly if possible and whitespace-insensitively
    otherwise. Returns None if it is not present under either reading.

    `matched_text` is the actual slice of `source`, which under the normalized tier
    may differ from `quote` in whitespace. Callers that need to show the user what
    the document really says should display `matched_text`, not the quote.
    """
    if not quote or not source:
        return None

    exact_at = source.find(quote)
    if exact_at >= 0:
        return SpanMatch(exact_at, exact_at + len(quote), EXACT, quote)

    norm_source, index_map = normalize_whitespace(source)
    norm_quote, _ = normalize_whitespace(quote)
    if not norm_quote:
        return None

    at = norm_source.find(norm_quote)
    if at < 0:
        # Diagnostic tier only. A hit here is NOT verified -- it records that the
        # failure is consistent with an interleaved-line-number artifact rather than
        # with fabrication. See the long note above for why it cannot be trusted.
        ln_source, ln_map = normalize_line_numbers(source)
        ln_quote, _ = normalize_line_numbers(quote)
        if ln_quote:
            ln_at = ln_source.find(ln_quote)
            if ln_at >= 0:
                start = ln_map[ln_at]
                end = ln_map[ln_at + len(ln_quote) - 1] + 1
                return SpanMatch(start, end, LINE_NUMBER_AMBIGUOUS, source[start:end])
        return None

    # Map the normalized span back to original offsets. `index_map[at]` is exact;
    # the end is the original index of the final matched character, plus its width
    # of one -- normalization never expands a character, so this cannot overshoot.
    start = index_map[at]
    end = index_map[at + len(norm_quote) - 1] + 1
    return SpanMatch(start, end, NORMALIZED, source[start:end])


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """The verdict on one claimed citation."""

    chunk_id: str
    quote: str
    method: str
    start: int | None = None
    end: int | None = None
    matched_text: str | None = None
    found_in_chunk_id: str | None = None  # set when method == WRONG_CHUNK

    @property
    def verified(self) -> bool:
        return self.method in VERIFIED_METHODS


def _chunks_by_doc(chunks: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    by_doc: dict[str, list[Mapping[str, Any]]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.get("doc_id", ""), []).append(chunk)
    return by_doc


def verify_citation(
    chunk_id: str,
    quote: str,
    chunk_index: Mapping[str, Mapping[str, Any]],
    *,
    by_doc: Mapping[str, list[Mapping[str, Any]]] | None = None,
    min_quote_chars: int = MIN_QUOTE_CHARS,
) -> CitationCheck:
    """Verify one `(chunk_id, quote)` pair against the indexed corpus.

    `chunk_index` maps chunk_id -> chunk record. `by_doc` is the optional
    doc_id -> chunks mapping that enables `wrong_chunk` detection; without it a
    misattributed quote is indistinguishable from a fabricated one, so the
    taxonomy collapses to verified/unverified.

    A `chunk_id` that is not in the index is `unverified`: the model invented a
    provenance handle, which is a fabrication even if the prose happens to be true.
    """
    stripped = quote.strip()
    if len(stripped) < min_quote_chars:
        return CitationCheck(chunk_id=chunk_id, quote=quote, method=TOO_SHORT)

    chunk = chunk_index.get(chunk_id)
    if chunk is not None:
        match = locate_quote(stripped, chunk.get("text") or "")
        if match is not None:
            return CitationCheck(
                chunk_id=chunk_id,
                quote=quote,
                method=match.method,
                start=match.start,
                end=match.end,
                matched_text=match.matched_text,
            )

    # Not in the cited chunk. Before calling it fabricated, check the rest of the
    # document -- a real quote pointed at the wrong chunk is a different defect.
    if by_doc is not None:
        doc_id = chunk.get("doc_id") if chunk is not None else None
        siblings = by_doc.get(doc_id or "", []) if doc_id else []
        for sibling in siblings:
            # Stringified to match how `chunk_index` is keyed. Comparing the raw value
            # meant a non-str chunk id never matched, so the cited chunk was searched
            # again as its own sibling -- wasted work, and a `wrong_chunk` verdict that
            # could point at the chunk that was cited.
            if str(sibling.get("chunk_id")) == chunk_id:
                continue
            match = locate_quote(stripped, sibling.get("text") or "")
            if match is not None:
                return CitationCheck(
                    chunk_id=chunk_id,
                    quote=quote,
                    method=WRONG_CHUNK,
                    start=match.start,
                    end=match.end,
                    matched_text=match.matched_text,
                    found_in_chunk_id=str(sibling.get("chunk_id")),
                )

    return CitationCheck(chunk_id=chunk_id, quote=quote, method=UNVERIFIED)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Aggregate verdict over one answer's citations."""

    checks: tuple[CitationCheck, ...]

    @property
    def n_claimed(self) -> int:
        return len(self.checks)

    @property
    def n_verified(self) -> int:
        return sum(1 for c in self.checks if c.verified)

    @property
    def counts(self) -> dict[str, int]:
        """Per-method tally. Every method key is present, including zeros, so a
        report never silently omits a bucket that happened not to occur."""
        out = dict.fromkeys(
            (EXACT, NORMALIZED, LINE_NUMBER_AMBIGUOUS, WRONG_CHUNK, UNVERIFIED, TOO_SHORT), 0
        )
        for check in self.checks:
            out[check.method] += 1
        return out

    @property
    def citation_precision(self) -> float | None:
        """Verified citations over claimed citations.

        None -- not 0.0 -- when nothing was claimed. An answer with no citations
        has an *undefined* precision, and scoring it zero would let a refusal drag
        down a mean that is supposed to measure citation quality. Refusal is
        scored separately, by the refusal path.
        """
        if not self.checks:
            return None
        return self.n_verified / len(self.checks)

    @property
    def fully_grounded(self) -> bool:
        """True when at least one citation was claimed and all of them verified."""
        return bool(self.checks) and self.n_verified == len(self.checks)


def verify_answer(
    citations: Iterable[Mapping[str, Any]],
    chunks: Iterable[Mapping[str, Any]],
    *,
    min_quote_chars: int = MIN_QUOTE_CHARS,
) -> VerificationReport:
    """Verify every citation in one generated answer.

    `citations` is the model's claimed list of `{"chunk_id": ..., "quote": ...}`.
    `chunks` is the retrieved context that was actually shown to the model --
    verification is scoped to what the model could see, so a quote lifted from
    elsewhere in the corpus is still a fabrication with respect to this answer.
    """
    chunk_list = list(chunks)
    chunk_index = {str(c["chunk_id"]): c for c in chunk_list if c.get("chunk_id")}
    by_doc = _chunks_by_doc(chunk_list)

    checks = tuple(
        verify_citation(
            str(cit.get("chunk_id", "")),
            str(cit.get("quote", "")),
            chunk_index,
            by_doc=by_doc,
            min_quote_chars=min_quote_chars,
        )
        for cit in citations
    )
    return VerificationReport(checks=checks)
