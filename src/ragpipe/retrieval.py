"""Retriever interface and the sparse (BM25) baseline.

Every retriever answers the same question — given a query, return ranked
`(chunk_id, score)` — so Phase 2's harness can score any of them and Phase 3 can
add dense, fused, and reranked variants without touching the evaluation code. The
interface is the point; BM25 is just the first implementation and the ablation's
floor.

`bm25s` is used rather than a hand-rolled implementation. A subtly wrong BM25
would corrupt every number in the ablation table while looking entirely plausible,
which is precisely the failure mode this project keeps finding. For the baseline
that all other rows are measured against, a tested implementation is worth the
dependency.

## The heading confound, and why `use_headings` exists

Phase 1b gives `structural` chunks an `embed_text` with the section heading
prepended, and `fixed` chunks none — 94.8% versus 0%. Indexing `embed_text`
unconditionally therefore made the strategy comparison measure two variables at
once: section-aligned boundaries *and* heading prefixes. Worse, the
`section_lookup` slice queries *with a heading*, so for `structural` the query text
was partly written into the index — near-leakage rather than retrieval.

`heading_mode` separates them, and needs **three** settings rather than two. The
obvious two-way switch (`embed_text` vs `text`) does not isolate anything, because a
section's `text` *begins with its own heading* — the span starts at the heading line —
so **27.1%** of `structural` chunks still carry the heading in the "off" condition
(3,720 of 13,706; equivalently 28.6% of the 13,000 chunks that carry a heading label at
all — the two figures have different denominators, and an earlier version of this note
printed the labelled-only figure beside the 94.8% above as though they were comparable).

Those counts are from the **pre-quarantine** corpus, superseded: the corpus now holds
13,423 chunks over 152 indexable documents, after three were excluded for broken text
layers. The overlap is of the same order on the current corpus and the argument for three
`heading_mode` settings is unaffected, but the exact figure is not re-derivable here —
the original measurement's definition of "carries the heading" is not recorded, and
substituting a freshly-invented definition would produce a number that looks like a
correction while measuring something slightly different. Recomputing it properly means
re-running the ablation, which is a Phase 9 task.

Measured on `section_lookup` recall@10 for `structural` at the shipped 1,024-char
target, the two-way switch understates the heading effect by 1.9x:

    prepend   0.828   heading prepended to every chunk (duplicated in openers)
    source    0.587   `text` as-is: heading still present in ~27% of chunks
    strip     0.495   heading removed everywhere

So the real effect is +0.333, not the +0.241 a two-way switch reports. The `source`
row is kept because it is the honest "do nothing" baseline, but `strip` is what the
comparison needs.

(These figures move with the chunk target *and* with the golden set. At the previous
2,048 default the same measurement read 0.892 / 0.774 / 0.511 for a +0.381 effect; on
the pre-quarantine golden set at 1,024 it read 0.876 / 0.631 / 0.408 for +0.467. The
numbers above are 1,024 on the post-quarantine set. **Cross-target comparisons are
therefore confounded** — the 2,048 figures were never re-measured against this golden
set, so "the effect is larger at 1,024" is not a claim this file can support any more.
The direction and the reason are what generalise, not the magnitudes.)

For `fixed`, all three modes score identically by construction: those chunks carry no
`section_heading`, so there is nothing to prepend or strip. The mode only moves
`structural`, which is why the confound was invisible until the strategies were
compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import bm25s

from ragpipe import tokenize as tok_mod


def _index_text(chunk: dict[str, Any], mode: str) -> str:
    """The text to index for one chunk, under the given heading mode."""
    text = chunk.get("text") or ""
    heading = chunk.get("section_heading")

    if mode == "prepend":
        return chunk.get("embed_text") or text
    if mode == "source":
        return text
    # strip: remove the heading wherever it appears, so the "off" condition really
    # is heading-free. A section's text opens with its heading, so simply choosing
    # `text` over `embed_text` leaves it indexed in most chunks.
    if heading:
        return text.replace(heading, " ")
    return text


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: str
    score: float
    rank: int  # 1-based


class Retriever(Protocol):
    """Anything that can rank chunk ids for a query."""

    name: str

    def search(self, query: str, k: int) -> list[Hit]: ...


class BM25Retriever:
    """Sparse lexical retrieval over chunk text, with identifier atoms.

    `stemmer=None` by default and deliberately. Stemming helps loose natural
    language but is a liability for precise identifiers: it can conflate distinct
    regulatory terms, and the atoms carry the exact-match work regardless. The
    knob is exposed so the ablation can measure the tradeoff rather than assume it.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        name: str = "bm25",
        use_identifier_atoms: bool = True,
        heading_mode: str = "prepend",  # prepend | source | strip
        stopwords: str | None = "en",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.name = name
        if heading_mode not in {"prepend", "source", "strip"}:
            raise ValueError(f"heading_mode must be prepend|source|strip, got {heading_mode!r}")
        self.use_identifier_atoms = use_identifier_atoms
        self.heading_mode = heading_mode
        self.stopwords = stopwords
        self._chunk_ids: list[str] = [c["chunk_id"] for c in chunks]

        corpus: list[str] = []
        for chunk in chunks:
            body = _index_text(chunk, heading_mode)
            if use_identifier_atoms:
                atoms = tok_mod.atoms_from_records(chunk.get("identifiers") or [])
                body = tok_mod.augment(body, atoms)
            corpus.append(body)

        tokens = bm25s.tokenize(corpus, stopwords=stopwords, show_progress=False, return_ids=True)
        self._index = bm25s.BM25(k1=k1, b=b)
        self._index.index(tokens, show_progress=False)
        self._vocab = getattr(tokens, "vocab", None)

    def __len__(self) -> int:
        return len(self._chunk_ids)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        prepared = tok_mod.augment_query(query) if self.use_identifier_atoms else query
        query_tokens = bm25s.tokenize(
            [prepared], stopwords=self.stopwords, show_progress=False, return_ids=False
        )

        # An empty token list is one way a query can carry no lexical signal, but it
        # is not the only one, and the earlier comment here was wrong about the
        # mechanism: it claimed `bm25s` raises rather than returning an empty
        # ranking. For *string* queries it does neither. `get_tokens_ids` silently
        # drops out-of-vocabulary terms, so a query of entirely unseen terms yields a
        # non-empty token list, an empty id list, and an **all-zero score vector** —
        # whereupon the top-k of a constant array is an arbitrary k documents.
        #
        # That is not a hypothetical. `NEPA_Final_Guidance` is one token under
        # bm25s' `\b\w\w+\b` pattern (underscore is a word character) and is out of
        # vocabulary, so this returned 100 chunks all scoring 0.0 in an order that
        # depended on internal argpartition behaviour — and rank fusion then consumed
        # that noise as ranks 1..100, making the fused rows irreproducible.
        if not query_tokens or not query_tokens[0]:
            return []

        k = max(1, min(k, len(self._chunk_ids)))
        try:
            indices, scores = self._index.retrieve(query_tokens, k=k, show_progress=False)
        except ValueError:
            return []

        hits: list[Hit] = []
        for idx, score in zip(indices[0].tolist(), scores[0].tolist(), strict=True):
            if idx < 0:
                continue
            # A zero BM25 score means no query term contributed anything to this
            # document. It is the absence of evidence, not weak evidence, so it must
            # not be handed to fusion as a rank — a document that matched nothing
            # would otherwise outrank one that matched, purely by appearing in both
            # candidate lists. Dropping these also makes the all-OOV case above
            # collapse to an empty ranking, which is the honest answer.
            if score <= 0.0:
                continue
            hits.append(Hit(chunk_id=self._chunk_ids[idx], score=float(score), rank=0))

        # Ties broken on chunk_id, matching `fusion._ranked` and
        # `CrossEncoderReranker.search`, which both do this and both cite determinism.
        # This retriever did not, so an exact score tie between a relevant and an
        # irrelevant chunk resolved on `bm25s`' internal ordering.
        #
        # **This makes ordering deterministic, not membership.** `bm25s` selects the
        # top-k by `argpartition` *before* this sort runs, so a score tie straddling
        # the k boundary is still resolved arbitrarily — which of the tied chunks is
        # inside the window is not reproducible, only their order once inside. Phase
        # 4's audit found this accounts for exactly 5 non-reproducible figures out of
        # 2,055 (two `recall@20`/`precision@20` pairs and one rerank ceiling, where a
        # tie straddles rank 20 or rank 50). Fixing it properly means retrieving k+ties
        # and truncating after the deterministic sort; not done, because the affected
        # metrics are window-edge ones and no headline figure depends on them.
        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return [Hit(chunk_id=h.chunk_id, score=h.score, rank=i) for i, h in enumerate(hits, 1)]


def load_chunks(path: Any) -> list[dict[str, Any]]:
    """Stream a chunk JSONL into memory. Text is kept — the index needs it."""
    import json

    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
