"""Golden retrieval set, derived programmatically with exact ground truth.

## Why this set is built from the corpus rather than written by hand

Phase 2's whole purpose is a ruler that can gate *every* change. That requires the
retrieval suite to be free and fast — an eval that costs money or an afternoon of
labelling gets run once and then quietly skipped, which is how projects end up with
unjustified parameters. So every query here is derived from structure already
present in the corpus, and its ground truth is exact by construction:

  `exact_identifier`  Query cites a CFR regulation. Relevant = the text around each
                      occurrence of that citation. The slice the hybrid-versus-dense
                      claim rests on.
  `section_lookup`    Query is a section heading unique across the corpus.
                      Relevant = that section's span.
  `title_lookup`      Query is a document title. Relevant = the document. A
                      document-level task, so `hit@k` is its metric, not `recall@k`.
  `unanswerable`      Query cites a regulation absent from the corpus. Relevant is
                      empty. Scored on score separability, not retrieval accuracy.

**These are programmatic probes, not natural-language questions, and the report
says so.** They test retrieval mechanics precisely and cost nothing to run, which is
what a gating suite needs. Natural-language questions require an LLM to draft and a
human to curate; they arrive with the generation eval in Phase 6 and answer a
different question.

## Ground truth is character spans, not chunk ids

This matters more than it sounds. The obvious design stores relevant *chunk ids* —
and it silently breaks the moment two chunking strategies are compared, because a
chunk id from one does not exist in the other. Scored that way, the first version of
this harness reported `recall@10` of 0.993 for `structural` and 0.123 for `fixed` on
the same queries, with `fixed` simultaneously scoring *higher* on `precision@10`.
That is not a retrieval difference; it is an artifact of falling back to
document-level truth for the strategy the set was not built on. A wrong ruler
invalidates every number measured with it.

Ground truth is therefore `(doc_id, start, end)` character spans into the extracted
document text, and a chunk is relevant iff its own span overlaps a target span. That
definition is identical for every chunking strategy, chunk size, and future
retriever, so cross-strategy comparison is meaningful by construction rather than by
remapping. It also means the eval set is built from *extracted documents*, never
from chunks.

Every selection step is seeded, so the same corpus and seed reproduce the same set.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from ragpipe import identifiers as ident_mod

# A citation occurring more often than this is too diffuse to be a precise
# retrieval target — matching any one of forty passages says little about precision.
MAX_OCCURRENCES = 6
MIN_OCCURRENCES = 1

# Section headings shorter than this are too generic to identify a section
# ("Scope", "Purpose") even when the exact string happens to be unique.
MIN_HEADING_CHARS = 18

# A section shorter than this has too little text to retrieve on its own.
MIN_SECTION_CHARS = 400

DEFAULT_PER_SLICE = 60


@dataclass(slots=True)
class EvalQuery:
    query_id: str
    slice_name: str
    query: str
    # Chunking-agnostic ground truth: character spans into the extracted document
    # text. A chunk is relevant iff its span overlaps one of these.
    relevant_spans: list[dict[str, Any]]
    relevant_doc_ids: list[str]
    # True when the target is the document itself rather than a passage within it.
    # Recall is not meaningful for these — see eval_report.
    doc_level: bool = False
    # How this query was derived, so a surprising result can be traced to its
    # construction rather than assumed to be a retrieval failure.
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _span(doc_id: str, start: int, end: int) -> dict[str, Any]:
    return {"doc_id": doc_id, "start": start, "end": end}


def build_exact_identifier(
    docs: list[dict[str, Any]], rng: random.Random, limit: int
) -> list[EvalQuery]:
    """Queries citing a real regulation. Ground truth = its occurrence spans."""
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        for rec in doc.get("identifiers") or []:
            if rec.get("kind") == ident_mod.PRIMARY_KIND and rec.get("canonical"):
                by_id[rec["canonical"]].append(_span(doc["doc_id"], rec["start"], rec["end"]))

    eligible = sorted(
        cid
        for cid, spans in by_id.items()
        if MIN_OCCURRENCES <= len(spans) <= MAX_OCCURRENCES
        # A citation with no section number ("21 CFR 312") names a whole part and is
        # a far looser target than a specific section; keep the specific ones.
        and "." in cid
    )
    rng.shuffle(eligible)

    out: list[EvalQuery] = []
    for canonical in eligible[:limit]:
        spans = by_id[canonical]
        out.append(
            EvalQuery(
                query_id=f"ident-{len(out):04d}",
                slice_name="exact_identifier",
                query=f"What are the requirements of {canonical}?",
                relevant_spans=spans,
                relevant_doc_ids=sorted({s["doc_id"] for s in spans}),
                provenance={"canonical": canonical, "n_occurrences": len(spans)},
            )
        )
    return out


def build_section_lookup(
    docs: list[dict[str, Any]], rng: random.Random, limit: int
) -> list[EvalQuery]:
    """Queries naming a corpus-unique section heading. Ground truth = its span."""
    candidates: list[tuple[str, str, int, int, int]] = []  # doc, heading, idx, s, e
    heading_keys: dict[str, set[tuple[str, int]]] = defaultdict(set)

    for doc in docs:
        for sec in doc.get("sections") or []:
            heading = (sec.get("heading") or "").strip()
            start, end = sec["start"], sec["end"]
            heading_keys[_normalise_heading(heading)].add((doc["doc_id"], sec["idx"]))
            if len(heading) >= MIN_HEADING_CHARS and end - start >= MIN_SECTION_CHARS:
                candidates.append((doc["doc_id"], heading, sec["idx"], start, end))

    eligible = [
        c
        for c in candidates
        # Unique corpus-wide, else ground truth is ambiguous: the same heading in
        # three documents makes "the" correct section undefined.
        if len(heading_keys[_normalise_heading(c[1])]) == 1
    ]
    eligible.sort()
    rng.shuffle(eligible)

    out: list[EvalQuery] = []
    for doc_id, heading, section_idx, start, end in eligible[:limit]:
        out.append(
            EvalQuery(
                query_id=f"section-{len(out):04d}",
                slice_name="section_lookup",
                query=heading,
                relevant_spans=[_span(doc_id, start, end)],
                relevant_doc_ids=[doc_id],
                provenance={
                    "doc_id": doc_id,
                    "section_idx": section_idx,
                    "section_chars": end - start,
                },
            )
        )
    return out


def build_title_lookup(
    docs: list[dict[str, Any]], rng: random.Random, limit: int
) -> list[EvalQuery]:
    """Queries naming a document title. The whole document is the target.

    Marked `doc_level`, because recall over "every chunk of the document" is bounded
    far below 1 by chunk count rather than by retrieval quality, and reporting it as
    if it were a retrieval score would misrepresent the slice.
    """
    eligible = sorted(
        (d for d in docs if len((d.get("title") or "").strip()) >= MIN_HEADING_CHARS),
        key=lambda d: d["doc_id"],
    )
    rng.shuffle(eligible)

    out: list[EvalQuery] = []
    for doc in eligible[:limit]:
        out.append(
            EvalQuery(
                query_id=f"title-{len(out):04d}",
                slice_name="title_lookup",
                query=doc["title"].strip(),
                relevant_spans=[_span(doc["doc_id"], 0, doc["n_chars"])],
                relevant_doc_ids=[doc["doc_id"]],
                doc_level=True,
                provenance={"doc_id": doc["doc_id"], "doc_chars": doc["n_chars"]},
            )
        )
    return out


def build_unanswerable(
    docs: list[dict[str, Any]], rng: random.Random, limit: int
) -> list[EvalQuery]:
    """Queries citing regulations that appear nowhere in the corpus.

    Constructed rather than sampled: a well-formed citation is synthesised and then
    *verified absent* from every document. Phrased identically to the
    `exact_identifier` slice, so the only difference between the two is whether the
    answer exists — which is what makes score separability meaningful.
    """
    present = {
        rec["canonical"]
        for doc in docs
        for rec in doc.get("identifiers") or []
        if rec.get("kind") == ident_mod.PRIMARY_KIND and rec.get("canonical")
    }
    # Titles that genuinely exist in the CFR, so the citation is well formed.
    titles = [7, 9, 16, 21, 40, 42, 45]

    out: list[EvalQuery] = []
    attempts = 0
    while len(out) < limit and attempts < limit * 200:
        attempts += 1
        canonical = f"{rng.choice(titles)} CFR {rng.randint(100, 1999)}.{rng.randint(1, 99)}"
        if canonical in present:
            continue
        present.add(canonical)  # never emit the same synthetic citation twice
        out.append(
            EvalQuery(
                query_id=f"unans-{len(out):04d}",
                slice_name="unanswerable",
                query=f"What are the requirements of {canonical}?",
                relevant_spans=[],
                relevant_doc_ids=[],
                provenance={"canonical": canonical, "synthetic": True},
            )
        )
    return out


def _normalise_heading(heading: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", heading.lower()).strip()


def relevant_chunks(query: EvalQuery, chunks_by_doc: dict[str, list[dict[str, Any]]]) -> set[str]:
    """Chunk ids relevant to `query` under span overlap.

    This is the whole point of span-based ground truth: it resolves to chunk ids at
    scoring time, against whichever chunk set is being evaluated, so the same query
    is scored correctly for every chunking strategy.
    """
    relevant: set[str] = set()
    for span in query.relevant_spans:
        for chunk in chunks_by_doc.get(span["doc_id"], ()):
            # Half-open interval overlap.
            if chunk["start"] < span["end"] and chunk["end"] > span["start"]:
                relevant.add(chunk["chunk_id"])
    return relevant


def build_all(
    docs: list[dict[str, Any]],
    seed: int = 20260813,
    per_slice: int = DEFAULT_PER_SLICE,
) -> list[EvalQuery]:
    """Build every slice from extracted documents.

    Each slice uses its own seeded RNG so sizes are independent — changing
    `per_slice` for one must not perturb the others.
    """
    return [
        *build_exact_identifier(docs, random.Random(f"{seed}:ident"), per_slice),
        *build_section_lookup(docs, random.Random(f"{seed}:section"), per_slice),
        *build_title_lookup(docs, random.Random(f"{seed}:title"), per_slice),
        *build_unanswerable(docs, random.Random(f"{seed}:unans"), per_slice),
    ]
