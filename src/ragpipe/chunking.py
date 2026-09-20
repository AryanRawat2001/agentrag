"""Chunking strategies.

The point of this module is not to chunk well — it is to make chunking a
*measured* decision. Every strategy produces chunks in the same shape with the
same metadata, so Phase 2's retrieval harness can score them against each other
and Phase 3's ablation can report which wins on which slice.

## Strategies

  `fixed`      Fixed-size character windows with overlap, snapped to the nearest
               natural boundary. The baseline. Ignores document structure entirely.
  `structural` Section boundaries from Phase 1a, with over-long sections
               sub-split and under-length sections merged.
  `semantic`   Splits at troughs in consecutive-sentence embedding similarity,
               thresholded at a percentile of each document's own distance
               distribution. Implemented in Phase 4 once an embedder existed; see
               `semantic.py`. It is the only strategy whose cost scales at
               *chunking* time rather than at indexing time.

## Two text fields, deliberately

`text` is the **exact source slice** — `extracted.text[chunk.start:chunk.end]`
byte for byte — because Phase 5's citation verifier asserts that a cited span
exists in the source document, and a chunk whose text has been decorated cannot
support that check.

`embed_text` is what actually gets embedded and indexed: the section heading
prepended to the body. A chunk reading "...must be submitted within 15 days" is
far more retrievable when it carries "Subpart C — Adverse Event Reporting" with
it, but that heading is not part of the source span. Keeping them separate is the
same split contextual retrieval will need in Phase 4, so the seam exists now.

## Sizes are in characters

Chunk sizes are character counts, not token counts, with `CHARS_PER_TOKEN` as the
documented conversion. The approximation is fine for *setting* a target — all
strategies use the same one — but it is not what the embedding model sees. Phase 3
measured the real ratio at ~4.5 chars/token median and far lower at the floor
(table-of-contents dot leaders run near 1), which is why truncation is reported
against a real tokenizer in `embedding.py` rather than inferred from these counts.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

CHARS_PER_TOKEN = 4

# Default target ~256 tokens with ~15% overlap. **Changed from 512 tokens (2,048
# chars) in Phase 4 after the chunk-size sweep. The reason is narrower than it first
# looked — see the trade-off note below the table.**
#
# The sweep (`reports/chunk_size_sweep.md`) ran 512/1024/2048/4096 chars at a fixed
# 15% overlap. `recall@10` on section lookup rises cleanly with chunk size, which
# reads as "bigger is better" and is an artifact: larger chunks mean fewer chunks per
# section, so recall's denominator shrinks. The denominator-free metrics say the
# opposite:
#
#     target   ident hit@1   ident hit@10   section hit@10   separability   truncated
#       512        0.900        0.967           0.948          0.992          0.0%
#     1,024        0.850        0.983           0.966          0.983          0.6%
#     2,048        0.833        0.983           0.966          0.942         10.4%
#     4,096        0.700        0.967           0.966          0.908         31.1%
#
# (Figures above are the `bm25` rows; separability is `best_accuracy`.)
#
# **This is a trade, not a domination — an earlier version of this comment claimed
# "nothing is traded away" and the sweep's own JSON refutes it.** Across all
# denominator-free metrics (`hit@1/5/10/20`, `mrr@10`) over 3 answerable slices x 4
# retrievers, 2,048 wins 22 comparisons, 1,024 wins 11, and 27 tie. `mrr@10` — which
# `metrics.py` calls "what a user actually sees" — prefers 2,048 on 8 of 12
# slice x retriever pairs, and section `hit@1` is 0.672 at 1,024 against 0.724 at 2,048.
# The five figures tabulated above genuinely favour 1,024; generalising from them to
# "every undistorted measure" was cherry-picking.
#
# 1,024 is still chosen, on two narrow grounds that hold:
#   * **Truncation**, 10.4% -> 0.6%. At 2,048 a tenth of chunks lose text before a
#     vector exists, which is a correctness defect rather than a ranking preference.
#   * **Refusal separability**, +0.042, which Phase 5's refusal path depends on.
# The cost is top-of-ranking precision: smaller chunks fragment a section across more
# pieces, so the single best chunk is less complete and `hit@1`/`mrr` suffer.
#
# Revisit if Phase 5 turns out to need rank-1 precision more than it needs the refusal
# threshold — that would be a real argument for 2,048, and the data is already here.
DEFAULT_TARGET_CHARS = 256 * CHARS_PER_TOKEN
DEFAULT_OVERLAP_CHARS = int(DEFAULT_TARGET_CHARS * 0.15)

# A chunk below this is too small to retrieve usefully; merge it forward.
MIN_CHUNK_CHARS = 200

# How far to look for a natural boundary when snapping a cut point. Beyond this
# the cut is taken as-is rather than distorting chunk sizes to chase a break.
BOUNDARY_SEARCH_CHARS = 250

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"[.!?]['\")\]]?\s")


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    strategy: str
    idx: int

    text: str  # exact source slice — do not decorate
    embed_text: str  # what gets embedded/indexed: heading + body
    start: int  # char offset into ExtractedDoc.text
    end: int
    n_chars: int

    page_start: int
    page_end: int
    section_idx: int | None = None
    section_heading: str | None = None
    # True when this document had no usable structure and `structural` had to fall
    # back to fixed windows. Reported, because it bounds what the comparison means.
    structural_fallback: bool = False

    content_hash: str = ""
    identifiers: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- boundary snapping -------------------------------------------------------


def _snap_forward(text: str, cut: int, limit: int) -> int:
    """Move `cut` forward to the nearest natural boundary within `limit` chars.

    Preference order: paragraph break, sentence end, whitespace. Falls back to the
    original cut so that a run of unbroken text cannot inflate a chunk without
    bound. Mid-word cuts are worth avoiding — they corrupt the lexical index and
    make retrieved snippets read as broken.
    """
    if cut >= len(text):
        return len(text)
    window = text[cut : min(len(text), cut + limit)]

    if m := _PARAGRAPH.search(window):
        return cut + m.end()
    if m := _SENTENCE_END.search(window):
        return cut + m.end()
    space = window.find(" ")
    if space >= 0:
        return cut + space + 1
    return cut


# --- strategies --------------------------------------------------------------


def _windows(text: str, start: int, end: int, target: int, overlap: int) -> list[tuple[int, int]]:
    """Overlapping [start, end) windows over text[start:end], boundary-snapped."""
    spans: list[tuple[int, int]] = []
    if end - start <= 0:
        return spans
    if end - start <= target:
        return [(start, end)]

    cursor = start
    while cursor < end:
        raw_cut = min(cursor + target, end)
        cut = _snap_forward(text, raw_cut, BOUNDARY_SEARCH_CHARS) if raw_cut < end else end
        cut = min(cut, end)
        if cut <= cursor:  # pathological: no progress possible
            cut = min(cursor + target, end)
        spans.append((cursor, cut))
        if cut >= end:
            break
        # Step back by the overlap, but never behind the previous start, or the
        # loop fails to advance on short windows.
        cursor = max(cursor + 1, cut - overlap)

    # The overlap step-back can leave a runt final window — a section ending a few
    # characters past the last cut yields an 11-character chunk. Fold any span
    # below the minimum into its predecessor rather than indexing a fragment.
    return _absorb_runts(spans)


def _absorb_runts(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge spans shorter than MIN_CHUNK_CHARS into the preceding span."""
    if len(spans) < 2:
        return spans
    out: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        if end - start < MIN_CHUNK_CHARS:
            prev_start, _ = out[-1]
            out[-1] = (prev_start, end)
        else:
            out.append((start, end))
    # A leading runt has no predecessor to merge into; merge it forward instead.
    if len(out) > 1 and out[0][1] - out[0][0] < MIN_CHUNK_CHARS:
        out[1] = (out[0][0], out[1][1])
        out.pop(0)
    return out


def chunk_fixed(
    extracted: dict[str, Any], target: int, overlap: int
) -> list[tuple[int, int, int | None]]:
    """Fixed-size windows over the whole document. Returns (start, end, section_idx)."""
    text = extracted["text"]
    return [(s, e, None) for s, e in _windows(text, 0, len(text), target, overlap)]


def chunk_structural(
    extracted: dict[str, Any], target: int, overlap: int
) -> list[tuple[int, int, int | None]]:
    """Section-aligned chunks, sub-splitting long sections and merging short ones.

    Falls back to `chunk_fixed` when the document has no sections — 8 documents in
    this corpus. That fallback is recorded per chunk rather than hidden, because it
    bounds what a structural-versus-fixed comparison actually measures.
    """
    sections = extracted.get("sections") or []
    if not sections:
        return chunk_fixed(extracted, target, overlap)

    text = extracted["text"]
    out: list[tuple[int, int, int | None]] = []

    # Merge forward until a section is worth emitting on its own. Regulatory
    # documents are full of one-line headings whose body is the next heading down.
    merged: list[tuple[int, int, int | None]] = []  # (start, end, section_idx)
    pending_start: int | None = None
    pending_idx: int | None = None
    for sec in sections:
        s, e = sec["start"], sec["end"]
        if pending_start is None:
            pending_start, pending_idx = s, sec["idx"]
        if e - pending_start >= MIN_CHUNK_CHARS:
            merged.append((pending_start, e, pending_idx))
            pending_start, pending_idx = None, None
    if pending_start is not None and pending_start < len(text):
        # Trailing remainder. If it is too short to stand alone, extend the previous
        # entry to cover it rather than emitting a fragment.
        if len(text) - pending_start < MIN_CHUNK_CHARS and merged:
            prev_start, _, prev_idx = merged[-1]
            merged[-1] = (prev_start, len(text), prev_idx)
        else:
            merged.append((pending_start, len(text), pending_idx))

    # Text before the first section (title pages, front matter) is often where the
    # document's identity lives, so it must be covered — but if it is too short to
    # stand alone, fold it into the first section rather than emitting a fragment.
    #
    # Only the fold happens here. Emitting the leading window is left to the loop
    # below, which covers *every* uncovered region uniformly. Handling the leading
    # case separately as well emitted it twice: 104 documents ended up with
    # duplicate spans, because the loop starts at cursor 0 and does not know the
    # region was already claimed.
    if merged and 0 < merged[0][0] < MIN_CHUNK_CHARS:
        _, first_end, first_idx = merged[0]
        merged[0] = (0, first_end, first_idx)

    # Sections are contiguous in practice (extract._close_sections sets each end to
    # the next start, and 0 of 155 documents have a mid or tail gap), but nothing
    # here enforced it. One loop covers the leading region, any gap between
    # sections, and any text past the final section, so a future structure source
    # that violates contiguity cannot silently lose text.
    cursor = 0
    for s, e, sec_idx in merged:
        if s > cursor:
            for ws, we in _windows(text, cursor, s, target, overlap):
                out.append((ws, we, None))
        if e - s <= target:
            out.append((s, e, sec_idx))
        else:
            for ws, we in _windows(text, s, e, target, overlap):
                out.append((ws, we, sec_idx))
        cursor = max(cursor, e)

    if cursor < len(text) and text[cursor:].strip():
        for ws, we in _windows(text, cursor, len(text), target, overlap):
            out.append((ws, we, None))

    out.sort(key=lambda span: span[0])
    return out


Strategy = Callable[..., list[tuple[int, int, int | None]]]


def _semantic_strategy(
    extracted: dict[str, Any], target: int, overlap: int, *, embedder: Any = None
) -> list[tuple[int, int, int | None]]:
    """Thin adapter so `semantic` lives in its own module without a circular import.

    `semantic.py` imports the boundary helpers from here, so importing it at module
    scope would close the cycle. Deferring the import to call time keeps the
    strategy registry the single place strategies are declared.
    """
    from ragpipe.semantic import chunk_semantic

    return chunk_semantic(extracted, target, overlap, embedder=embedder)


STRATEGIES: dict[str, Strategy] = {
    "fixed": chunk_fixed,
    "structural": chunk_structural,
    "semantic": _semantic_strategy,
}

# Strategies that need an embedding model. Named as a set rather than checked by
# string, so `build_chunks` cannot drift from the registry.
EMBEDDING_STRATEGIES = frozenset({"semantic"})

# Nothing is deferred any more: `semantic` was the only entry, and Phase 3's
# embedder made it implementable. Kept as an empty registry rather than deleted —
# `build_chunks` and the CLI both report against it, and the mechanism for
# declaring an unimplemented strategy honestly is worth keeping.
DEFERRED_STRATEGIES: dict[str, str] = {}


# --- assembly ----------------------------------------------------------------


def _page_range(pages: list[dict[str, Any]], start: int, end: int) -> tuple[int, int]:
    """First and last page a span touches. Pages are ordered and non-overlapping."""
    first = last = pages[0]["page"] if pages else 1
    for p in pages:
        if p["start"] <= start < p["end"] or (p["start"] < end and p["end"] > start):
            first = p["page"]
            break
    for p in reversed(pages):
        if p["start"] < end and p["end"] > start:
            last = p["page"]
            break
    return first, max(first, last)


def build_chunks(
    extracted: dict[str, Any],
    strategy: str,
    target: int = DEFAULT_TARGET_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
    embedder: Any = None,
) -> list[Chunk]:
    """Chunk one extracted document with the named strategy.

    `embedder` is required only by `semantic`, and demanded rather than defaulted:
    silently falling back to a size-based split would put a row labelled "semantic"
    in the comparison table that was nothing of the kind.
    """
    if strategy in DEFERRED_STRATEGIES:
        raise NotImplementedError(
            f"strategy {strategy!r} is not implemented: {DEFERRED_STRATEGIES[strategy]}"
        )
    if strategy not in STRATEGIES:
        raise KeyError(f"unknown strategy {strategy!r}; have {sorted(STRATEGIES)}")
    if strategy in EMBEDDING_STRATEGIES and embedder is None:
        raise ValueError(f"strategy {strategy!r} requires an embedder")

    # Both of these degrade silently rather than failing, which is worse than an
    # error: a target below the runt threshold makes every window a runt, so
    # `_absorb_runts` folds an entire document into one chunk, and an overlap at or
    # above the target advances the cursor one character at a time, producing a span
    # per character. Neither is a configuration anyone means to choose.
    if target <= MIN_CHUNK_CHARS:
        raise ValueError(
            f"target ({target}) must exceed MIN_CHUNK_CHARS ({MIN_CHUNK_CHARS}); "
            "below it every window is a runt and the document collapses to one chunk"
        )
    if overlap >= target:
        raise ValueError(f"overlap ({overlap}) must be less than target ({target})")
    if overlap < 0:
        raise ValueError(f"overlap ({overlap}) must not be negative")

    text: str = extracted["text"]
    pages = extracted.get("pages") or []
    sections = {s["idx"]: s for s in (extracted.get("sections") or [])}
    all_ids = extracted.get("identifiers") or []
    fallback = strategy == "structural" and not sections

    if strategy in EMBEDDING_STRATEGIES:
        spans = STRATEGIES[strategy](extracted, target, overlap, embedder=embedder)
    else:
        spans = STRATEGIES[strategy](extracted, target, overlap)
    chunks: list[Chunk] = []

    # `idx` is assigned after filtering, not before, so it stays a dense
    # per-document index. Enumerating the raw spans left holes wherever a
    # whitespace-only span was dropped, which would break any later neighbour
    # expansion by `idx +/- 1`.
    for start, end, sec_idx in spans:
        body = text[start:end]
        if not body.strip():
            continue  # a span of pure whitespace is not a chunk
        i = len(chunks)

        heading = sections.get(sec_idx, {}).get("heading") if sec_idx is not None else None
        embed_text = f"{heading}\n\n{body}" if heading else body

        # Identifiers wholly inside this span, with offsets rebased onto the chunk
        # so a citation can be located without carrying the whole document.
        inside = [
            {**ident, "start": ident["start"] - start, "end": ident["end"] - start}
            for ident in all_ids
            if ident["start"] >= start and ident["end"] <= end
        ]

        page_start, page_end = _page_range(pages, start, end)
        chunks.append(
            Chunk(
                chunk_id=f"{extracted['doc_id']}::{strategy}::{i:05d}",
                doc_id=extracted["doc_id"],
                source=extracted["source"],
                strategy=strategy,
                idx=i,
                text=body,
                embed_text=embed_text,
                start=start,
                end=end,
                n_chars=len(body),
                page_start=page_start,
                page_end=page_end,
                section_idx=sec_idx,
                section_heading=heading,
                structural_fallback=fallback,
                content_hash=content_hash(body),
                identifiers=inside,
                metadata=extracted.get("metadata") or {},
            )
        )
    return chunks


def content_hash(text: str) -> str:
    """Hash of normalised text, for exact-duplicate detection.

    Whitespace is collapsed and case folded first: the same regulatory boilerplate
    reflowed differently by two PDFs is the same content, and treating it as
    distinct would defeat the purpose.
    """
    normalised = re.sub(r"\s+", " ", text).strip().casefold()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
