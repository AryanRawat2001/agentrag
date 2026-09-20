"""Semantic chunking: split where consecutive sentences stop being about the same thing.

Deferred from Phase 1b because it needs an embedding model, which Phase 3 supplies.

## The method

Split the document into sentences, embed each one, and take the cosine distance
between consecutive sentence embeddings. A large distance is a topic shift, so the
largest distances are the natural cut points. Cut at distances above a **percentile
of that document's own distribution**, not above a fixed distance.

The percentile matters more than it looks. Absolute cosine distances are not
comparable across documents: a tightly-worded regulation has uniformly low
consecutive distances and a heterogeneous protocol has uniformly high ones, so a
fixed threshold like 0.25 splits one document into sentences and the other not at
all. A percentile adapts per document, which is the only way one parameter can mean
the same thing corpus-wide.

## Where it usually goes wrong

Three failure modes, all guarded here, because "semantic chunking" implemented
naively produces chunks that *look* principled and retrieve badly:

1. **Unbounded chunk size.** A document with no strong topic shifts yields one
   enormous chunk that blows the context window — the truncation problem from
   `embedding.py` in its worst form. `target` bounds chunk size regardless of what
   the distances say. Not exactly a hard cap, for two compounding reasons: cuts snap
   *forward* to the nearest boundary, adding up to `BOUNDARY_SEARCH_CHARS` (250), and
   absorbing a runt extends its predecessor by up to `MIN_CHUNK_CHARS - 1` (199). So
   the real bound is `target + 449`, and the largest semantic chunk in this corpus is
   2,464 characters against a 2,048 target — against 2,298 for `fixed` and
   `structural`, which snap but never absorb into an already-full window. Bounded and
   accounted for, not unbounded.
2. **Sentence-sized chunks.** A trough-dense region cuts on nearly every sentence,
   producing fragments with no retrievable content. `MIN_CHUNK_CHARS` suppresses a
   cut that would leave a runt — enforced in **both** `_spans_from_cuts` and
   `_oversize_split`, because two of the three return paths skip the former.
3. **Losing the tail.** Off-by-one in the breakpoint loop silently drops the text
   after the last cut. Spans are asserted to tile `[0, len(text))` exactly.

## Tiling versus assembled coverage

The spans this module returns tile exactly. The *chunks* built from them do not
quite, and the difference is worth stating because it looks like a bug in the
numbers: `build_chunks` drops any span whose text is entirely whitespace, since a
whitespace-only span is not a chunk. Sentence splitting on `\\s+` can produce such a
span where a blank-line block sits between two sentences, which `fixed` and
`structural` never generate because their windows always contain prose.

Measured on this corpus: 272 characters in 1 document fall outside the assembled
chunks, and **0 of them are non-whitespace**. So `semantic` reports 2,996,532 corpus
tokens against 2,996,600 for the other two strategies — a 68-token difference that is
entirely blank space. (Before runt absorption was fixed in `_oversize_split` it was
1,522 characters across 116 documents, still all whitespace; merging runts swallowed
most of those spans.) The invariant that actually matters is therefore not raw tiling
of the assembled output but that no non-whitespace character is ever lost, which is
what the tests assert.


## What it costs

Every sentence is embedded, so this is the only chunking strategy whose cost scales
with corpus size at *chunking* time rather than at indexing time — and the sentence
embeddings are thrown away afterwards, since chunks get re-embedded as units. That
is a real argument against it that a comparison table should surface rather than
bury: if it does not beat `structural`, it is strictly worse, because `structural`
is free.

Sentence embeddings deliberately do **not** use the query prefix. These are
document-side representations being compared to each other, not queries.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ragpipe.chunking import (
    BOUNDARY_SEARCH_CHARS,
    MIN_CHUNK_CHARS,
    _absorb_runts,
    _snap_forward,
)

# Sentence boundary: terminal punctuation, optional closing quote/bracket, then
# whitespace. Deliberately the same pattern `chunking._SENTENCE_END` uses for
# boundary snapping, so the two notions of "sentence" cannot drift apart.
#
# Matched forwards and cut at `m.end()` rather than written as a lookbehind: the
# optional closing bracket makes the prefix variable-width, and Python's `re`
# rejects a variable-width lookbehind outright. Requiring trailing whitespace is
# also what keeps `21 CFR 314.50` and `0.5 mg` from becoming sentence boundaries.
_SENTENCE_SPLIT = re.compile(r"[.!?]['\")\]]?\s+")

# Percentile of a document's own consecutive-distance distribution above which a
# gap is treated as a topic shift. 95 cuts at roughly one sentence in twenty.
DEFAULT_BREAKPOINT_PERCENTILE = 95.0

# Sentences shorter than this are merged into their neighbour before embedding.
# A fragment like "See Table 3." carries almost no signal, and its embedding is
# noise that invents topic shifts on both sides of it.
MIN_SENTENCE_CHARS = 40

# Cap on sentences embedded per document. A 900-page protocol would otherwise
# dominate the chunking run; beyond this the document is split on size alone.
MAX_SENTENCES_PER_DOC = 4000


def sentence_spans(text: str, min_chars: int = MIN_SENTENCE_CHARS) -> list[tuple[int, int]]:
    """Contiguous `[start, end)` spans, one per sentence, tiling the whole text.

    Tiling is the invariant: spans are adjacent and cover `[0, len(text))`, so a
    chunk assembled from them is still a byte-exact source slice. Short fragments
    are merged forward rather than dropped, because dropping any span would break
    that coverage and silently lose text.
    """
    if not text:
        return []

    cuts = [0]
    for m in _SENTENCE_SPLIT.finditer(text):
        cuts.append(m.end())
    cuts.append(len(text))

    spans: list[tuple[int, int]] = []
    for i in range(len(cuts) - 1):
        start, end = cuts[i], cuts[i + 1]
        if end <= start:
            continue
        # Merge a too-short sentence into the previous span, preserving adjacency.
        if spans and end - start < min_chars:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))

    # A leading fragment has no predecessor; merge it forward instead.
    if len(spans) > 1 and spans[0][1] - spans[0][0] < min_chars:
        spans[1] = (spans[0][0], spans[1][1])
        spans.pop(0)
    return spans


def breakpoints(
    distances: np.ndarray, percentile: float = DEFAULT_BREAKPOINT_PERCENTILE
) -> np.ndarray:
    """Indices where the distance to the next sentence exceeds the percentile.

    Relative to the document's own distribution, so one parameter means the same
    thing in a tightly-worded regulation and a heterogeneous protocol.
    """
    if distances.size == 0:
        return np.zeros(0, dtype=int)
    threshold = float(np.percentile(distances, percentile))
    # Strictly greater: with a flat distribution (every distance identical) the
    # threshold equals every value, and `>=` would cut at every sentence.
    return np.flatnonzero(distances > threshold)


def _spans_from_cuts(
    text: str,
    sentences: list[tuple[int, int]],
    cut_after: set[int],
    target: int,
) -> list[tuple[int, int]]:
    """Accumulate sentences into chunks, cutting at breakpoints and at `target`."""
    spans: list[tuple[int, int]] = []
    start = sentences[0][0]

    for i, (_s, end) in enumerate(sentences):
        size = end - start
        last = i == len(sentences) - 1
        # A semantic cut is only honoured once the chunk is big enough to be worth
        # retrieving; `target` forces a cut regardless of what the distances say.
        if last or (size >= target) or (i in cut_after and size >= MIN_CHUNK_CHARS):
            spans.append((start, end))
            if not last:
                start = sentences[i + 1][0]

    # Guard the tail: the loop above ends on the final sentence, but a `target`
    # cut landing exactly there could leave a gap. Extend rather than lose text.
    if spans and spans[-1][1] < len(text):
        spans[-1] = (spans[-1][0], len(text))
    return _absorb_runts(spans)


def _oversize_split(text: str, spans: list[tuple[int, int]], target: int) -> list[tuple[int, int]]:
    """Hard-split any span still over `target` after semantic cutting.

    Reachable whenever a single sentence exceeds `target` — extraction produces
    them from tables and un-punctuated lists, so this is not a theoretical case.

    Ends in `_absorb_runts`, and that call is the whole point of this note. This
    function began as a copy of `chunking._windows`, which ends the same way, and the
    copy dropped that line. The result: forward boundary-snapping leaves a remainder
    of `target - snap` characters, so a span of `target + 2` split into `target + snap`
    plus a **1-character** chunk. 597 of 8,001 semantic chunks (7.5%) came out below
    `MIN_CHUNK_CHARS` and the smallest was a single character — while this module's
    docstring claimed `MIN_CHUNK_CHARS` suppressed exactly that. Two of the three
    `chunk_semantic` return paths reach this function without passing through
    `_spans_from_cuts`, so absorbing there was not enough.
    """
    out: list[tuple[int, int]] = []
    for start, end in spans:
        if end - start <= target:
            out.append((start, end))
            continue
        cursor = start
        while cursor < end:
            raw = min(cursor + target, end)
            cut = _snap_forward(text, raw, BOUNDARY_SEARCH_CHARS) if raw < end else end
            cut = min(max(cut, cursor + 1), end)
            out.append((cursor, cut))
            cursor = cut
    return _absorb_runts(out)


def chunk_semantic(
    extracted: dict[str, Any],
    target: int,
    overlap: int,
    *,
    embedder: Any,
    percentile: float = DEFAULT_BREAKPOINT_PERCENTILE,
) -> list[tuple[int, int, int | None]]:
    """Semantic spans for one document. Returns `(start, end, section_idx)`.

    `overlap` is accepted for signature compatibility and deliberately unused: the
    premise of semantic chunking is that a boundary falls where the topic changes,
    and overlapping across it re-introduces the bleed the method exists to avoid.
    Reported rather than silently ignored — see the chunking report.
    """
    text = extracted["text"]
    if not text:
        return []

    sentences = sentence_spans(text)
    if len(sentences) < 2:
        # Nothing to compare, so there are no distances and no breakpoints — but the
        # size cap still applies. Returning `[(0, len(text))]` here ignored `target`
        # outright, so a document of un-punctuated table rows became one span of
        # arbitrary length. Extraction produces exactly that.
        return [(s, e, None) for s, e in _oversize_split(text, [(0, len(text))], target)]

    if len(sentences) > MAX_SENTENCES_PER_DOC:
        # Too large to embed sentence-wise; fall back to size-only splitting. Doing
        # this loudly beats an unbounded embedding bill on one outlier document.
        return [(s, e, None) for s, e in _oversize_split(text, [(0, len(text))], target)]

    vectors = embedder.encode_documents([text[s:e] for s, e in sentences])
    # Vectors are unit-norm, so the dot product is cosine similarity and cosine
    # distance is 1 - that. Clipped because floating-point error can put an
    # identical pair marginally above 1.0 and yield a negative distance.
    sims = np.clip(np.sum(vectors[:-1] * vectors[1:], axis=1), -1.0, 1.0)
    distances = 1.0 - sims

    cut_after = set(breakpoints(distances, percentile).tolist())
    spans = _spans_from_cuts(text, sentences, cut_after, target)
    spans = _oversize_split(text, spans, target)
    return [(s, e, None) for s, e in spans]
