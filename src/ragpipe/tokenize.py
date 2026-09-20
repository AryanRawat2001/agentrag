"""Identifier-aware tokenization for lexical retrieval.

## The problem this exists to solve

Standard BM25 tokenization shatters regulatory citations into common numbers:

    '21 CFR 314.50'  ->  ['21', 'cfr', '314', '50']
    '21 CFR 314.5'   ->  ['21', 'cfr', '314']        <- the '.5' vanishes entirely
    '21 CFR 50'      ->  ['21', 'cfr', '50']

Three consequences, all fatal to the exact-identifier eval slice:

  1. `314.50` and `314.5` become the same token bag — BM25 cannot tell two
     different regulations apart.
  2. `21 CFR 314.50` and `21 CFR 50` share two of three tokens.
  3. `21`, `314`, `50` are among the most common tokens in regulatory text, so
     the citation carries almost no discriminative weight. High document
     frequency is exactly what BM25 down-weights.

The exact-identifier slice exists to demonstrate that lexical retrieval beats
dense retrieval on precise identifiers. With default tokenization it would fail —
and would fail for a tokenizer reason while looking like evidence that lexical
retrieval does not help. That is the kind of wrong architectural conclusion this
project's verification process exists to prevent, and it would have been drawn
from a table that looked entirely plausible.

## The fix

Each identifier occurrence contributes an extra **atomic token** derived from its
canonical form: `21 CFR 314.50` yields `idz21zqcqfqrz314z50` — the `q` before each
uppercase letter is the case marking added so `(a)` and `(A)` cannot collide. The
encoding is
lowercase alphanumeric with `z` separators, chosen because it survives every
tokenizer's splitting rules intact — no punctuation, no whitespace, nothing to
split on.

Atoms are **appended, not substituted.** Substituting would delete the natural
words, so a loose query like "CFR 314 requirements" would stop matching. Appending
gives exact citation queries a rare, highly discriminative token while leaving
ordinary word matching untouched. Atoms are appended once per distinct identifier
per document, not once per occurrence, so term frequency is not inflated by a
citation that happens to repeat twenty times.

The same encoding runs over queries and documents, which is the only way the
atoms can ever meet.
"""

from __future__ import annotations

import re
from typing import Any

from ragpipe import identifiers as ident_mod

# Prefix marks a token as a synthetic identifier atom, so it can never collide
# with a real word from the corpus.
ATOM_PREFIX = "idz"
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def atomic_token(canonical: str) -> str:
    """Encode a canonical identifier as a single indivisible token.

    Lowercase alphanumeric with `z` separators: no character in the output can be
    split on by a word tokenizer, so the token survives indexing intact.

    Upper-case letters are marked rather than folded. `identifiers.py` deliberately
    preserves subsection case — `(C)` and `(c)` are different subdivisions — and
    lowercasing collided them into one atom. No collision occurs in this corpus, but
    an identifier scheme that silently merges two distinct citations is exactly the
    defect this module exists to prevent.
    """
    marked = "".join(f"q{ch.lower()}" if ch.isupper() else ch for ch in canonical)
    return ATOM_PREFIX + _NON_ALNUM.sub("z", marked.lower())


def atoms_for(text: str, kinds: list[str] | None = None) -> list[str]:
    """Atomic tokens for the distinct identifiers in `text`, in first-seen order.

    Both granularities are emitted — section level and subsection level — so a
    query citing `21 CFR 117.136` matches a document discussing
    `21 CFR 117.136(a)(2)`, while a query citing the subsection still gets an
    exact hit on it. De-duplicated, so a citation repeated twenty times does not
    inflate its own term frequency.
    """
    seen: dict[str, None] = {}
    for ident in ident_mod.extract(text, kinds=kinds):
        for form in (ident.canonical, ident.canonical_detailed):
            seen.setdefault(atomic_token(form), None)
    return list(seen)


def atoms_from_records(records: list[dict[str, Any]]) -> list[str]:
    """Atoms from already-extracted identifier records, avoiding a re-scan.

    Chunks carry their identifiers from Phase 1, so indexing can reuse that work
    instead of running the regexes over the whole corpus again.
    """
    seen: dict[str, None] = {}
    for rec in records:
        for key in ("canonical", "canonical_detailed"):
            value = rec.get(key)
            if value:
                seen.setdefault(atomic_token(value), None)
    return list(seen)


def augment(text: str, atoms: list[str]) -> str:
    """Append identifier atoms to a text for indexing or querying."""
    return f"{text}\n{' '.join(atoms)}" if atoms else text


def augment_query(query: str) -> str:
    """Prepare a query: append atoms for any identifier it mentions."""
    return augment(query, atoms_for(query))
