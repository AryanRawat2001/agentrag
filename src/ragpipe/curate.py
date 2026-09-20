"""Curation triage for the golden answer set: the defect classes coverage cannot see.

`corpus/golden.jsonl` has always been **machine-validated and spot-checked, not curated**,
and `plan.md` said so. Validation checks five things: tier-1 evidence located, answer
length, boilerplate document spread, cross-pair overlap, and answer/evidence lexical
coverage. Phase 6 then measured what that last one misses, and the answer was the part
that matters most in regulatory text:

    "60 days" against "30 days", "may" against "shall", and "shall not submit" against
    "shall submit" all scored coverage 1.000 and were accepted.

Coverage is a bag of words. Every one of those substitutions preserves the bag.

The numeric case is already handled — `golden.unsupported_numeric_claims` resolves written
numerals and flags digits the source never states. This module covers the two that remain,
and they are the ones that silently invert an obligation.

## Modal strength

FDA guidance uses a documented convention: **must** is a requirement, **should** is a
recommendation, **may** is an option. So an answer that says "must" about a source that
says "should" has manufactured a legal obligation, and one that says "may" about a source
that says "shall" has dissolved one. Either makes the pair wrong as a reference answer
while leaving it lexically indistinguishable from a right one.

Modals are ranked and the answer is flagged when it asserts *strictly more* obligation
than its cited source supports. **Weakening is not flagged**: a branch for it fired on 19
of 159 pairs, contributed 45% of all flags, and reading every one found zero real
defects — a cited chunk is a thousand characters of regulatory prose and nearly always
contains a "must" somewhere, so an answer correctly paraphrasing one permissive sentence
looks weaker than "the chunk". It was detecting that regulations contain obligations.

## Negation

The highest-severity class, and the rarest. An answer that drops a "not" reverses the
source exactly.

Detected as **polarity attached to a modal**, not as a count. The first version compared
negation-marker counts between answer and source and flagged any asymmetry: it fired on
**96 of 159 pairs**, because a fifty-word answer is being compared against a
thousand-character chunk and the source is nearly always more negated. A check that fires
on 60% of the data is not triage — it teaches a reviewer to skip it, which is strictly
worse than no check. Scoped to "the answer states this obligation unnegated where the
source only ever negates it", it fires on **2**.

## What this module deliberately does not do

**It does not decide.** Every function here returns *flags with the evidence attached*, for
a human to read. This project has been burned by a check that silently matched nothing and
looked like clean data, and the opposite failure — a heuristic confident enough to delete
real pairs — would be worse. The flags exist so a curation pass reads twenty pairs instead
of a hundred and fifty-nine.

**It does not attempt role reversal** ("the sponsor shall" vs "the Agency shall").
Detecting it needs to know which actor governs which clause, and a bag-of-actors proxy
would fire on every pair that mentions two parties — which in this corpus is most of them.
Pairs with multiple actors *and* an obligation are surfaced for reading instead, which is
honest about the method being human judgement with machine assistance.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Modal verbs ranked by how much obligation they assert. The ordering follows FDA's own
#: stated convention for its guidance documents -- "must" for requirements, "should" for
#: recommendations, "may" for options -- extended with the forms that appear in clinical
#: protocols.
#:
#: `will` sits between recommendation and requirement deliberately, and it is the least
#: certain rank here. In a protocol "subjects will be randomised" is a description of the
#: planned procedure, carrying more commitment than "should" but not the legal force of
#: "shall". Ranking it *below* mandatory is what makes "will" -> "must" a flagged
#: strengthening, which is the substitution this corpus actually produces.
MODAL_STRENGTH: dict[str, int] = {
    "may": 1,
    "might": 1,
    "can": 1,
    "could": 1,
    "optional": 1,
    "permitted": 1,
    "should": 2,
    # All of `recommend`'s forms, for the reason the `require` comment below gives. Only
    # the past participle was present, so "The following recommendations apply to
    # amendments, updates, and supplements" and "We recommend you provide the following"
    # both scored **0** -- and a source scoring 0 routes to the *lenient* branch, on the
    # rationale that imperatives and bare specifications oblige without modals. That
    # rationale is sound for a protocol imperative and exactly inverted for an imperative
    # nested inside a recommendation frame, which is the commonest shape in FDA Q&A
    # guidance. It cost a wrong keep on `fda-78268::gold::094`.
    "recommend": 2,
    "recommends": 2,
    "recommendation": 2,
    "recommendations": 2,
    "recommended": 2,
    "encouraged": 2,
    "expected": 2,
    "will": 3,
    "shall": 4,
    "must": 4,
    # All inflections, because the bare infinitive was missing and it is the form this
    # corpus uses most: "MR scanning sessions require participants to lie flat" went
    # unmatched, so the source's strongest modal read as "will" and a correct answer
    # saying "must" was flagged as a strengthening. 8 of 159 sources contain bare
    # "require". An incomplete lexicon does not weaken a checker gently -- it produces
    # confident false positives, which is how a reviewer learns to distrust it.
    "require": 4,
    "required": 4,
    "requires": 4,
    # "requirement" is deliberately absent. It is a *noun*, not a modal assertion: an
    # answer reading "may receive a transfusion to reach this requirement" refers back to
    # a threshold rather than imposing one, and ranking it mandatory flagged a pair whose
    # own modals ("should", "may") matched its source exactly.
    "mandatory": 4,
    "prohibited": 4,
}

STRENGTH_LABEL = {1: "permissive", 2: "recommended", 3: "descriptive", 4: "mandatory"}

#: Negation markers. `except` and `unless` are included because they invert the scope of
#: an obligation as surely as "not" does.
NEGATION_MARKERS = (
    "not",
    "no",
    "never",
    "cannot",
    "neither",
    "nor",
    "except",
    "unless",
    "without",
    "prohibited",
    "may not",
)

#: Parties an obligation can attach to. Used only to decide whether a pair needs reading,
#: never to decide whether it is wrong.
ACTORS = (
    "sponsor",
    "applicant",
    "agency",
    "fda",
    "investigator",
    "irb",
    "manufacturer",
    "secretary",
    "institution",
    "submitter",
    "subject",
    "patient",
    "physician",
)


def _spaced_pattern(word: str) -> str:
    """A pattern matching `word` even with single spaces inserted between its letters.

    PDF text extraction in this corpus splits words: one source reads "the requir ed
    reporting", and `\brequired\b` does not match it. That one document was enough to
    make a *correct* answer look like it had manufactured an obligation, because the
    source's strongest detected modal fell back to "will" from an unrelated clause.

    Tolerating one space per gap rather than any whitespace run: `character_spaced`
    documents are quarantined out of the index by `pdfcheck`, so what survives here is
    occasional single-character splitting, not fully spaced-out text. A looser pattern
    would start matching across word boundaries.
    """
    return r"\b" + r"\s?".join(re.escape(c) for c in word) + r"\b"


def _modals_present(text: str) -> dict[str, int]:
    """Modal -> strength, for every ranked modal appearing in `text`.

    Space-tolerant, so PDF word-splitting cannot hide a modal and thereby invent a
    strength mismatch. Only the strongest spelling of each strength matters downstream,
    so the extra cost of the looser pattern buys correctness rather than noise.
    """
    found: dict[str, int] = {}
    for modal, strength in MODAL_STRENGTH.items():
        if re.search(_spaced_pattern(modal), text, re.I):
            found[modal] = strength
    return found


#: How far either side of a modal to look for a negation. English puts it on either
#: side depending on the modal: "shall not submit" and "may not be used" negate
#: forward, while "are not required" and "is not needed" negate backward. Two tokens
#: covers both; widening it starts catching the negation of a different clause in the
#: same sentence.
_POLARITY_TOKENS = 2
#: Word-bounded on both sides, and that is load-bearing. Without `\b` the alternative
#: `not` matches *inside* "notified", "notice" and "annotation", so "the FDA must be
#: notified" was read as a negated obligation and flagged as a polarity inversion. Same
#: class of defect as the `\bcomment\b` regex this project found in Phase 6 — that one had
#: a boundary it should not have, this one lacked one it needed. Both make a check report
#: confidently about text it never actually matched.
_NEGATION_NEAR = r"\b(?:not|never|cannot|no\s+longer)\b"


def modal_polarities(text: str) -> set[tuple[str, bool]]:
    """`(modal, is_negated)` for every ranked modal occurrence in `text`.

    "shall submit" yields `("shall", False)`; "shall not submit" yields `("shall", True)`.
    A text containing both yields both, which is the point: it is what lets the check
    distinguish "the source never says this unnegated" from "the source says both".

    The window looks **both directions**. English puts the negation on either side
    depending on the modal -- "shall *not* submit" negates forward, "are *not* required"
    negates backward -- and checking only forward, which the first version did, read "are
    not required" as an assertion of obligation. That is the exact inversion this function
    exists to detect, so the bug made the check blind to its own subject.
    """
    out: set[tuple[str, bool]] = set()
    for modal in MODAL_STRENGTH:
        before = rf"((?:\w+\s+){{0,{_POLARITY_TOKENS}}})"
        after = rf"((?:\s+\w+){{0,{_POLARITY_TOKENS}}})"
        # Space-tolerant, like `_modals_present`. Routing `max_modal_strength` through
        # this function for polarity awareness silently re-broke the PDF word-splitting
        # fix, because only the other matcher had been made tolerant -- caught by the
        # test written for that earlier bug, which is the whole argument for pinning a
        # fix with a test rather than trusting the fix.
        for match in re.finditer(rf"{before}{_spaced_pattern(modal)}{after}", text, re.I):
            window = f"{match.group(1)} {match.group(2)}"
            out.add((modal, bool(re.search(_NEGATION_NEAR, window, re.I))))
    return out


def max_modal_strength(text: str) -> int:
    """The strongest obligation `text` asserts, or 0 if it asserts none.

    **Negated occurrences do not count.** "Bookmarks are not required" asserts the
    *absence* of an obligation, and counting it as mandatory made a correct answer
    ("not required", paraphrasing a source's "not needed") read as a strengthening. A
    modal is only evidence of obligation in the polarity it was used in.
    """
    polarities = modal_polarities(text)
    asserted = [MODAL_STRENGTH[m] for m, negated in polarities if not negated]
    return max(asserted, default=0)


def count_negations(text: str) -> int:
    """Negation markers in `text`, counted with occurrences.

    **Not used by any check**, and kept deliberately rather than deleted. The first
    negation check counted these and flagged any asymmetry between answer and source; it
    fired on 96 of 159 pairs, because a fifty-word answer is compared against a
    thousand-character chunk. `check_negation` now compares polarity attached to a modal
    instead. This function remains as the measurement that justified that rewrite -- and
    as a caution, since the counting approach is the obvious first idea and this is what
    it cost.

    Multi-word markers are removed before the single words are counted, so "may not"
    contributes one rather than being counted again by bare "not".
    """
    lowered = text.lower()
    total = 0
    for phrase in (m for m in NEGATION_MARKERS if " " in m):
        hits = len(re.findall(rf"\b{re.escape(phrase)}\b", lowered))
        total += hits
        lowered = re.sub(rf"\b{re.escape(phrase)}\b", " ", lowered)
    for word in (m for m in NEGATION_MARKERS if " " not in m):
        total += len(re.findall(rf"\b{re.escape(word)}\b", lowered))
    return total


def actors_present(text: str) -> set[str]:
    return {a for a in ACTORS if re.search(rf"\b{re.escape(a)}\b", text, re.I)}


@dataclass(frozen=True, slots=True)
class Flag:
    """One reason a pair needs a human to read it."""

    kind: str
    severity: str  # "high" | "medium" | "low"
    detail: str
    #: The text on each side that produced the flag, so a reviewer never has to go and
    #: find it. A flag without its evidence is a to-do item, not a finding.
    answer_excerpt: str = ""
    source_excerpt: str = ""


@dataclass
class PairReview:
    """A pair, every flag raised against it, and where the flags came from."""

    pair_id: str
    doc_id: str
    question: str
    answer: str
    source: str = field(repr=False, default="")
    flags: list[Flag] = field(default_factory=list)
    #: The source sentence a reviewer actually needs to read, chosen by content overlap
    #: with the answer rather than by proximity to whichever modal tripped the flag.
    relevant_source: str = ""

    @property
    def severity(self) -> str:
        for level in ("high", "medium", "low"):
            if any(f.severity == level for f in self.flags):
                return level
        return "none"

    @property
    def needs_review(self) -> bool:
        return bool(self.flags)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "doc_id": self.doc_id,
            "question": self.question,
            "answer": self.answer,
            "relevant_source": self.relevant_source,
            "severity": self.severity,
            "flags": [
                {
                    "kind": f.kind,
                    "severity": f.severity,
                    "detail": f.detail,
                    "answer_excerpt": f.answer_excerpt,
                    "source_excerpt": f.source_excerpt,
                }
                for f in self.flags
            ],
        }


def _excerpt(text: str, needle: str, width: int = 110) -> str:
    """The window of `text` around `needle`, for a reviewer to read in context."""
    # Space-tolerant, like `_modals_present`. Without this, a modal matched only because
    # PDF extraction split it ("requir ed") produces an *empty* excerpt -- and `Flag`'s
    # own docstring says a flag without its evidence is a to-do item, not a finding.
    match = re.search(rf"\b{re.escape(needle)}\b", text, re.I) or re.search(
        _spaced_pattern(needle), text, re.I
    )
    if not match:
        return ""
    start = max(0, match.start() - width // 2)
    end = min(len(text), match.end() + width // 2)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{' '.join(text[start:end].split())}{suffix}"


_SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+")
#: Words too common to carry meaning in the overlap score. Modals are included
#: deliberately: scoring on them would surface whichever sentence shares the answer's
#: modal, which is the thing under dispute rather than evidence about it.
_REVIEW_STOPWORDS = frozenset(
    (
        *[
            "the",
            "a",
            "an",
            "of",
            "and",
            "or",
            "to",
            "in",
            "for",
            "be",
            "is",
            "are",
            "was",
            "were",
            "with",
            "that",
            "this",
            "on",
            "by",
            "as",
            "at",
            "from",
        ],
        *["will", "shall", "must", "may", "should", "not", "no"],
    )
)


def relevant_source_sentence(answer: str, source: str) -> str:
    """The source sentence with the most content-word overlap with `answer`.

    Exists because the first version of this worksheet showed the window around the
    *source's strongest modal*, which is frequently a different clause entirely: one pair
    about central venous lines was presented beside a sentence about glomerular
    filtration rate, because that unrelated sentence happened to contain the chunk's only
    "will". A reviewer then has to go and find the relevant text by hand, which is exactly
    the work the worksheet exists to remove -- and a worksheet that makes judging harder
    than reading the raw pair is worse than no worksheet.

    Modals are excluded from the overlap vocabulary on purpose: matching on them would
    preferentially surface whichever sentence shares the answer's *modal*, which is the
    thing under dispute rather than evidence about it.
    """
    answer_words = {
        w
        for w in re.findall(r"[a-z0-9]+", answer.lower())
        if len(w) > 3 and w not in _REVIEW_STOPWORDS
    }
    if not answer_words:
        return ""
    best, best_score = "", 0.0
    for sentence in _SENTENCE_SPLIT.split(source):
        cleaned = " ".join(sentence.split())
        if len(cleaned) < 30:
            continue
        words = {
            w
            for w in re.findall(r"[a-z0-9]+", cleaned.lower())
            if len(w) > 3 and w not in _REVIEW_STOPWORDS
        }
        if not words:
            continue
        score = len(answer_words & words) / len(answer_words | words)
        if score > best_score:
            best, best_score = cleaned, score
    return best[:400]


def _pick(modals: set[str], strength: int) -> str:
    """The canonical spelling among modals sharing `strength`.

    One tie-break rule, used by every check, so two reports of the same data cannot
    disagree. Longest-then-alphabetical: it prefers the more specific spelling
    ("recommendations" over "should") and is stable across processes, which a `set`
    iteration order is not.
    """
    candidates = sorted(
        (m for m in modals if MODAL_STRENGTH[m] == strength), key=lambda m: (-len(m), m)
    )
    return candidates[0] if candidates else ""


def check_modal_strength(answer: str, source: str) -> list[Flag]:
    """Flag an answer that asserts more (or less) obligation than its source.

    Strengthening is the dangerous direction: it invents a requirement the document does
    not impose, and a reference answer asserting one scores a *correct* model answer as
    wrong. Weakening is not reported -- see the comment below for the measurement behind
    that.
    """
    # Polarity-aware on both sides, via `max_modal_strength` rather than by re-deriving
    # the maximum here. The first version called `_modals_present` directly and took its
    # own max, so making `max_modal_strength` skip negated modals fixed nothing --
    # "bookmarks are not required" still read as an assertion of obligation. A check that
    # recomputes a rule the module already implements will not notice when the rule is
    # corrected, which is this project's recurring defect in its purest form.
    a_max, s_max = max_modal_strength(answer), max_modal_strength(source)
    if a_max == 0:
        return []

    asserted = {m for m, negated in modal_polarities(answer) if not negated}
    strongest = _pick(asserted, a_max)
    s_modals = {m: MODAL_STRENGTH[m] for m, negated in modal_polarities(source) if not negated}
    # Deterministic, and that is load-bearing rather than tidy. `modal_polarities` returns
    # a **set**, so `max(s_modals, key=...)` broke ties by whichever modal the set happened
    # to yield first -- which depends on `PYTHONHASHSEED`. Four seeds produced four
    # different `reports/golden_curation.md` files, so the artifact was not reproducible by
    # the command that writes it, and the flag's `source_excerpt` sometimes pointed at FDA
    # page-header boilerplate instead of the clause under dispute.
    #
    # Adding four `recommend*` spellings at strength 2 made this far worse: eight modals
    # now share that rank, so ties went from rare to routine. `check_modal_in_context`
    # already used this exact sort; this call site is the one that did not get it.
    s_strongest = _pick(set(s_modals), s_max) if s_modals else ""
    if a_max > s_max:
        # A source stating *no* obligation level is a different situation from one stating
        # a weaker level, and conflating them produced most of this check's false
        # positives. Regulatory text carries obligation without modals all the time --
        # imperatives ("Ship via Federal Express"), bare specifications ("slice thickness
        # 5 mm"), and enumerated statutory lists ("(5) Have systems and processes in
        # place") all oblige, and all score 0. An answer that reads a modal into one of
        # those is interpreting, which is worth a look but is not the `should` -> `must`
        # error this check exists to find.
        severity = "high" if s_max else "medium"
        kind = "modal_strengthened" if s_max else "modal_added_to_unmodalised_source"
        return [
            Flag(
                kind=kind,
                severity=severity,
                detail=(
                    f'answer asserts "{strongest}" ({STRENGTH_LABEL[a_max]}); strongest in '
                    f"source is "
                    + (
                        f'"{s_strongest}" ({STRENGTH_LABEL[s_max]})'
                        if s_modals
                        else "no modal at all"
                    )
                ),
                answer_excerpt=_excerpt(answer, strongest),
                source_excerpt=(_excerpt(source, s_strongest) if s_strongest else ""),
            )
        ]
    # Weakening is deliberately **not** reported, and that is a measured decision rather
    # than an oversight. A `modal_weakened` branch here fired on 19 of 159 pairs and
    # contributed 45% of the worksheet's flags. Reading all 19 found **zero** real
    # defects, and a sentence-level re-test confirmed it: 18 of the 19 vanish when the
    # answer is compared against the sentence that actually addresses it rather than
    # against the whole chunk, and the survivor was a false positive too.
    #
    # The cause is structural. A cited chunk is a thousand characters of regulatory prose
    # and nearly always contains a "must" or a "required" *somewhere*; an answer
    # correctly paraphrasing one permissive sentence out of it therefore looks weaker
    # than "the chunk". So the check was not detecting under-claiming, it was detecting
    # that regulatory documents contain obligations.
    #
    # Under-claiming is also the safe direction: a reference answer that hedges scores a
    # confident model answer as unsupported, which is a missed point rather than a
    # manufactured requirement. Not worth 19 false positives to catch zero instances of.
    return []


def check_modal_in_context(answer: str, source: str) -> list[Flag]:
    """Compare the answer's obligation against the *most relevant source sentence*.

    `check_modal_strength` compares against the whole cited chunk, which is deliberately
    lenient -- an answer may draw on several sentences, so requiring its modal to match
    the chunk's maximum would flag correct multi-sentence paraphrase. But leniency has a
    cost, and reading real pairs found it: the chunk-level comparison can get the
    *direction* backwards.

    One pair asserts "the study **will** be discontinued" where its source says "we
    **may** discontinue the study". That is a discretionary safety action restated as a
    commitment -- a strengthening -- yet it was reported as `modal_weakened` at **low**
    severity, because the chunk happened to contain "required" in an unrelated clause and
    so out-ranked the answer overall.

    So this compares like with like: the answer against the single sentence that actually
    addresses it. Narrower, and wrong in a different direction (a pair whose answer spans
    two sentences can flag spuriously), which is why both checks run and neither decides.
    """
    sentence = relevant_source_sentence(answer, source)
    if not sentence:
        return []
    a_max, s_max = max_modal_strength(answer), max_modal_strength(sentence)
    if s_max == 0 or a_max <= s_max:
        return []
    strongest = _pick({m for m, negated in modal_polarities(answer) if not negated}, a_max)
    s_strongest = _pick({m for m, negated in modal_polarities(sentence) if not negated}, s_max)
    return [
        Flag(
            kind="modal_strengthened_in_context",
            severity="high",
            detail=(
                f'against the sentence that addresses it, the answer asserts "{strongest}" '
                f'({STRENGTH_LABEL[a_max]}) where the source says "{s_strongest}" '
                f"({STRENGTH_LABEL[s_max]})"
            ),
            answer_excerpt=_excerpt(answer, strongest),
            source_excerpt=sentence[:220],
        )
    ]


def check_negation(answer: str, source: str) -> list[Flag]:
    """Flag an answer whose obligation carries the opposite polarity to its source.

    Compares **polarity attached to each modal**, not negation counts. The first version
    of this check counted negation markers on each side and flagged any asymmetry, which
    fired on **96 of 159 pairs** — because a fifty-word answer is being compared against a
    thousand-character chunk, so the source almost always contains more negations. A check
    that fires on 60% of the data is not triage; it trains a reviewer to skip it, which is
    worse than having no check at all.

    Scoped to the actual defect instead: the answer asserts a modal *unnegated* where the
    source only ever states that same modal *negated* (or the reverse). That is the
    "shall not submit" -> "shall submit" inversion, and it fires on 2 pairs in 159.
    """
    a_pol, s_pol = modal_polarities(answer), modal_polarities(source)
    flags: list[Flag] = []
    for modal, negated in sorted(a_pol):
        opposite = (modal, not negated)
        # Only when the source is unambiguous: it states the opposite polarity and never
        # the one the answer used. If the source says both, the answer picked one of two
        # readings the document genuinely supports and only a reader can judge which.
        if opposite in s_pol and (modal, negated) not in s_pol:
            flags.append(
                Flag(
                    kind="polarity_inverted",
                    severity="high",
                    detail=(
                        f'answer uses "{modal}" '
                        f"{'negated' if negated else 'unnegated'}; the cited source only "
                        f"ever states it {'unnegated' if negated else 'negated'}"
                    ),
                    answer_excerpt=_excerpt(answer, modal),
                    source_excerpt=_excerpt(source, modal),
                )
            )
    return flags


def check_attribution(answer: str, source: str) -> list[Flag]:
    """Surface pairs where an obligation is attached to a party, for reading.

    Not a detector of role reversal, and the docstring says so because an earlier version
    of this project shipped three checks whose descriptions promised more than they did.
    Deciding whether "the sponsor shall" was correctly transcribed from "the Agency shall"
    needs to know which actor governs which clause; a bag-of-actors comparison would fire
    on most pairs in this corpus, since regulatory text names both parties constantly.

    This fires only when the answer asserts an obligation *and* names an actor the source
    does not -- a much narrower condition, and one that cannot be resolved without
    reading.
    """
    if max_modal_strength(answer) < 3:
        return []
    a_actors, s_actors = actors_present(answer), actors_present(source)
    novel = a_actors - s_actors
    if not novel:
        return []
    who = sorted(novel)[0]
    return [
        Flag(
            kind="actor_not_in_source",
            severity="medium",
            detail=(
                f'answer places an obligation on "{who}", which the cited source never '
                f"names (source names: {', '.join(sorted(s_actors)) or 'no party'})"
            ),
            answer_excerpt=_excerpt(answer, who),
        )
    ]


def check_numeric(answer: str, source: str) -> list[Flag]:
    """Reuse the existing numeric resolver, expressed as a flag."""
    from ragpipe.golden import unsupported_numeric_claims

    missing = unsupported_numeric_claims(answer, source)
    if not missing:
        return []
    return [
        Flag(
            kind="unsupported_number",
            severity="high",
            detail=f"answer asserts {', '.join(missing)}, absent from the cited source",
            answer_excerpt=_excerpt(answer, missing[0]),
        )
    ]


CHECKS = (
    check_modal_strength,
    check_modal_in_context,
    check_negation,
    check_attribution,
    check_numeric,
)


def review_pair(
    pair: Mapping[str, Any], chunk_index: Mapping[str, Mapping[str, Any]]
) -> PairReview:
    """Run every check against one pair.

    The comparison source is the **full text of each cited chunk**, not the quoted span.
    That is the same choice `flag_factual_risk` makes, and it is the lenient one on
    purpose: an answer may legitimately draw on the sentence after the quote it cites, so
    comparing against the quote alone would flag correct paraphrase as invention. The
    strict reading -- answer must be supported by the quoted span alone -- is a different
    and harsher standard than this set was drafted under.
    """
    source = " ".join(
        (chunk_index.get(e["chunk_id"], {}) or {}).get("text") or ""
        for e in pair.get("evidence", [])
    )
    review = PairReview(
        pair_id=str(pair.get("pair_id", "")),
        doc_id=str(pair.get("doc_id", "")),
        question=str(pair.get("question", "")),
        answer=str(pair.get("answer", "")),
        source=source,
    )
    # `.strip()`, not truthiness. Two unresolvable chunk ids join to " " -- truthy -- so
    # the guard was skipped and `modal_added_to_unmodalised_source`, `actor_not_in_source`
    # and `unsupported_number` all fired against whitespace, producing three confident
    # false positives on a pair whose evidence had vanished.
    if not source.strip():
        review.flags.append(
            Flag(
                kind="no_source",
                severity="high",
                detail="no cited chunk resolved, so nothing can be checked against",
            )
        )
        return review
    for check in CHECKS:
        review.flags.extend(check(review.answer, source))
    review.relevant_source = relevant_source_sentence(review.answer, source)
    return review


def review_all(
    pairs: Sequence[Mapping[str, Any]], chunk_index: Mapping[str, Mapping[str, Any]]
) -> list[PairReview]:
    return [review_pair(p, chunk_index) for p in pairs]


def summarise(reviews: Sequence[PairReview]) -> dict[str, Any]:
    """Counts by flag kind and severity, plus the review burden this implies."""
    by_kind: dict[str, int] = {}
    by_severity: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for review in reviews:
        for flag in review.flags:
            by_kind[flag.kind] = by_kind.get(flag.kind, 0) + 1
        if review.needs_review:
            by_severity[review.severity] += 1
    flagged = sum(1 for r in reviews if r.needs_review)
    return {
        "n_pairs": len(reviews),
        "n_flagged": flagged,
        "flagged_share": round(flagged / len(reviews), 4) if reviews else None,
        "by_kind": dict(sorted(by_kind.items())),
        "by_severity": by_severity,
    }


def load_verdicts(path: Any) -> dict[str, dict[str, Any]]:
    """Human verdicts on flagged pairs, keyed by `pair_id`.

    Separate from the golden set and committed alongside it, because these are the one
    artifact in this project that **cannot be regenerated** -- only redone. Every derived
    report here can be rebuilt by re-running a command; a judgement about whether an
    answer misstates a regulation is a person's reading, and Phase 6 shipped a
    faithfulness figure that was not re-derivable because the verdicts behind it were
    never written down.

    Each entry records the verdict, the reason, and which flag prompted it, so a later
    reader can disagree with a specific call rather than with the set as a whole.
    """
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    entries = raw.get("verdicts", raw) if isinstance(raw, dict) else {}
    verdicts = {str(k): v for k, v in entries.items() if isinstance(v, dict)}

    # Validated, loudly, because this file is hand-edited and the failure is silent in the
    # dangerous direction. Only `verdict == "reject"` is acted on, so a typo -- `"rejct"`,
    # or a stray capital -- leaves a pair a human rejected sitting in the accepted set,
    # with the worksheet still reporting it as judged. Nothing downstream would notice.
    bad = {
        pid: entry.get("verdict")
        for pid, entry in verdicts.items()
        if entry.get("verdict") not in VERDICT_VALUES
    }
    if bad:
        listed = ", ".join(f"{pid}={value!r}" for pid, value in sorted(bad.items())[:5])
        raise ValueError(
            f"{p}: {len(bad)} verdict(s) with an unrecognised value ({listed}). "
            f"Allowed: {', '.join(VERDICT_VALUES)}. A rejection with a misspelled verdict "
            f"is silently ignored, so this is refused rather than skipped."
        )

    # `amendments` is an audit trail no code reads, which is exactly why it is checked: an
    # entry recording a keep->reject that the verdicts block never received would leave the
    # file *documenting* a rejection the pipeline does not apply.
    if isinstance(raw, dict):
        for i, entry in enumerate(raw.get("amendments") or []):
            if not isinstance(entry, dict):
                raise ValueError(f"{p}: amendments[{i}] is not an object")
            missing = {"pair_id", "from", "to", "why", "date"} - set(entry)
            if missing:
                raise ValueError(f"{p}: amendments[{i}] is missing {sorted(missing)}")
            if entry["to"] not in VERDICT_VALUES:
                raise ValueError(
                    f"{p}: amendments[{i}] records to={entry['to']!r}, "
                    f"which is not one of {', '.join(VERDICT_VALUES)}"
                )
            current = verdicts.get(str(entry["pair_id"]), {}).get("verdict")
            if current != entry["to"]:
                raise ValueError(
                    f"{p}: amendments[{i}] says {entry['pair_id']} was changed to "
                    f"{entry['to']!r}, but the verdicts block records {current!r}. The "
                    f"audit trail and the verdict disagree; one of them was not updated."
                )
    return verdicts


#: `edit` is accepted by the loader and refused by `cmd_curate`: it is a legitimate thing
#: for a human to want and no consumer implements it, so recording one would silently do
#: nothing. Kept in the vocabulary so the refusal can name it.
VERDICT_VALUES = ("keep", "reject", "edit")


def render_worksheet(
    reviews: Sequence[PairReview],
    summary: dict[str, Any],
    verdicts: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """A worksheet a human reads, ordered so the dangerous flags come first.

    Recorded verdicts are rendered beside their pair, so the artifact shows what has been
    judged and what has not. A worksheet that looked identical before and after a curation
    pass would make the pass unverifiable from the outside.
    """
    verdicts = verdicts or {}
    lines: list[str] = []
    a = lines.append
    a("# Golden set — curation worksheet")
    a("")
    # `flagged_share` is None when there are no pairs at all, and formatting None as a
    # percentage raises. A clean set is the *goal* state of this whole exercise, so the
    # renderer crashing on it would mean the artifact could never record success.
    share = f" ({summary['flagged_share']:.1%})" if summary.get("flagged_share") is not None else ""
    a(
        f"{summary['n_flagged']} of {summary['n_pairs']} pairs carry at least one flag"
        f"{share}. Flags are **triage, not verdicts**: each one "
        "names something a bag-of-words check cannot see and a human can, and a flagged "
        "pair is as likely to be a legitimate paraphrase as an error."
    )
    a("")
    a("| severity | pairs |")
    a("|---|---:|")
    for level in ("high", "medium", "low"):
        a(f"| {level} | {summary['by_severity'][level]} |")
    a("")
    if summary.get("verdicts_recorded") is not None:
        # Two populations, not one ratio. `verdicts_recorded` counts every verdict on file;
        # `n_flagged` counts what the *current* detectors flag. Dividing one by the other
        # printed "Judged: 25 of 24" once the detector fixes retired a flag whose pair had
        # already been judged -- a ratio above 1, which is not a progress figure at all.
        judged_flagged = summary["n_flagged"] - summary.get("flagged_unjudged", 0)
        a(
            f"**Judged: {judged_flagged} of {summary['n_flagged']} flagged pairs.** "
            f"{summary.get('flagged_unjudged', 0)} still unread."
        )
        retained = summary.get("verdicts_retired_by_detector_fixes", 0)
        if retained:
            a("")
            noun = "verdict is" if retained == 1 else "verdicts are"
            a(
                f"{retained} further {noun} kept for pairs the current detectors no longer "
                f"flag. That is not backlog: a human read the pair when an earlier detector "
                f"flagged it, and the fixed detector now agrees with the verdict. Such "
                f"verdicts are retained so a rejection cannot be silently undone by a "
                f"detector change — see `revalidate`, which honours them."
            )
        a("")
    a("| flag | count | what it means |")
    a("|---|---:|---|")
    meanings = {
        "modal_strengthened": "answer asserts more obligation than the source ('should' -> 'must')",
        "modal_added_to_unmodalised_source": (
            "answer reads an obligation into a source that states none — an imperative, a "
            "bare specification, or an enumerated statutory duty"
        ),
        "modal_strengthened_in_context": (
            "against the single sentence that addresses it, the answer asserts more "
            "obligation than the source"
        ),
        "polarity_inverted": (
            "the answer states an obligation in the polarity the source never uses"
        ),
        "actor_not_in_source": "an obligation is attached to a party the source never names",
        "unsupported_number": "answer states a number the cited source does not",
        "no_source": "cited chunk did not resolve",
    }
    for kind, count in summary["by_kind"].items():
        a(f"| `{kind}` | {count} | {meanings.get(kind, '')} |")
    a("")

    order = {"high": 0, "medium": 1, "low": 2}
    for review in sorted(
        (r for r in reviews if r.needs_review), key=lambda r: (order[r.severity], r.pair_id)
    ):
        a(f"## `{review.pair_id}` — {review.severity}")
        a("")
        a(f"**Q.** {review.question}")
        a("")
        a(f"**A.** {review.answer}")
        a("")
        if review.relevant_source:
            a(f"**Closest source sentence.** {review.relevant_source}")
            a("")
        for flag in review.flags:
            a(f"- **{flag.kind}** ({flag.severity}) — {flag.detail}")
            if flag.answer_excerpt:
                a(f"  - answer: `{flag.answer_excerpt}`")
            if flag.source_excerpt:
                a(f"  - source: `{flag.source_excerpt}`")
        verdict = verdicts.get(review.pair_id)
        if verdict:
            a("")
            a(f"**Verdict: {verdict.get('verdict', '?')}** — {verdict.get('reason', '')}")
        else:
            a("")
            a("**Verdict: not yet judged.**")
        a("")
    return "\n".join(lines)
