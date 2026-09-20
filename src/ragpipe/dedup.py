"""Near-duplicate detection over chunks, by MinHash + LSH.

## Why not the obvious approach

The conventional recipe is "embed everything, drop any chunk with cosine
similarity > 0.95 to an existing one." Two problems with it here:

  1. **It needs embeddings.** Deduplication then can't run until the embedding
     model exists, and every re-chunk costs a full re-embed. MinHash over word
     shingles estimates Jaccard overlap directly from the text, needs no model,
     and runs in seconds — so dedup is available in Phase 1 and free to re-run
     during the chunking sweep.
  2. **Cosine similarity is the wrong measurement.** Two chunks can be
     semantically near-identical (high cosine) while being different regulations,
     and near-duplicate boilerplate can sit below 0.95 while being literally the
     same paragraph. Jaccard over shingles measures textual duplication, which is
     what the problem actually is.

## Why cluster instead of skip

This corpus is regulatory documentation, and boilerplate is *legitimately*
near-identical across documents: standard safety-reporting language, identical
regulatory citations, templated protocol sections. Dropping the duplicates would
destroy the ability to answer "which documents impose this requirement?" — a
question the corpus is well suited to and which provenance depends on.

So duplicates are clustered, not deleted. One member is chosen as canonical and
the rest record `duplicate_of` plus their own source location. Retrieval can then
return one result while still reporting every document the text appears in, and
the ablation can measure whether collapsing duplicates helps or hurts.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

# Word 5-grams. Long enough that ordinary sentences don't collide, short enough
# that a reflowed paragraph still overlaps heavily with its twin.
SHINGLE_SIZE = 5

NUM_PERM = 128
# 32 bands x 4 rows. The banding threshold ~ (1/bands)^(1/rows) ~ 0.42, so pairs
# above roughly 0.4 Jaccard become candidates and are then verified exactly.
LSH_BANDS = 32
LSH_ROWS = 4

# Verified Jaccard at or above this is a duplicate. Deliberately below the
# conventional 0.95: PDF extraction reflows text, so genuine duplicates routinely
# differ in whitespace and hyphenation.
DUPLICATE_THRESHOLD = 0.85

# Screening slack below the threshold, so a pair whose *estimate* falls short but
# whose true Jaccard clears it is still checked exactly. Sized at ~3 standard
# errors of a 128-permutation MinHash at J=0.85.
_SCREEN_MARGIN = 0.10

_MERSENNE = (1 << 61) - 1
_MAX_HASH = (1 << 64) - 1

_WORD = re.compile(r"\w+")


def _stable_hash(text: str) -> int:
    """64-bit content hash that is stable across processes.

    Python salts `str.__hash__` per process unless PYTHONHASHSEED is set, so using
    the builtin here made every signature, LSH bucket, cluster membership and
    similarity figure change on each run — while three separate docstrings in this
    module claimed determinism. The committed cluster files were unreproducible.
    """
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


@dataclass(slots=True)
class DuplicateCluster:
    canonical_id: str
    member_ids: list[str]
    size: int
    similarity_floor: float


def _permutations(num_perm: int, seed: int = 20260813) -> list[tuple[int, int]]:
    """Deterministic (a, b) coefficients for the hash family (a*x + b) mod p."""
    import random

    rng = random.Random(seed)
    return [(rng.randrange(1, _MERSENNE), rng.randrange(0, _MERSENNE)) for _ in range(num_perm)]


_PERMS = _permutations(NUM_PERM)


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[int]:
    """Hashed word n-grams. Case-folded, punctuation-stripped, whitespace-agnostic
    so that reflowed text produces the same shingle set."""
    words = _WORD.findall(text.casefold())
    if len(words) < size:
        # Short text: one shingle of everything, so it can still match its twin.
        return {_stable_hash(" ".join(words))} if words else set()
    return {_stable_hash(" ".join(words[i : i + size])) for i in range(len(words) - size + 1)}


def signature(shingle_set: set[int]) -> tuple[int, ...]:
    """MinHash signature: the minimum of each permutation over the shingle set."""
    if not shingle_set:
        return tuple([_MAX_HASH] * NUM_PERM)
    return tuple(min(((a * s + b) % _MERSENNE) for s in shingle_set) for a, b in _PERMS)


def estimated_jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    """MinHash estimate of Jaccard. Used only to *screen* candidate pairs."""
    matches = sum(1 for x, y in zip(sig_a, sig_b, strict=True) if x == y)
    return matches / len(sig_a)


def exact_jaccard(a: set[int], b: set[int]) -> float:
    """True Jaccard over shingle sets.

    The estimate has a standard error of ~0.032 at J=0.85 with 128 permutations,
    which admitted and rejected the wrong pairs — 4 of 46 shipped duplicate pairs
    were actually below the stated threshold. The shingle sets are already in hand,
    so the final decision is exact and the docstrings' "verified exactly" is now
    true rather than aspirational.
    """
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Union by id keeps clustering deterministic across runs.
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def find_duplicates(
    items: list[tuple[str, str]], threshold: float = DUPLICATE_THRESHOLD
) -> tuple[dict[str, str], list[DuplicateCluster]]:
    """Cluster near-duplicate texts.

    `items` is [(id, text)]. Returns (duplicate_of, clusters) where `duplicate_of`
    maps each non-canonical id to its cluster's canonical id. Ids absent from the
    mapping are unique or canonical.

    Exact duplicates are caught first by normalised hash. That pass is nearly free
    but, measured on this corpus, accounts for a small minority of what is found
    (0 of 5 clusters for `fixed`, 4 of 35 for `structural`) — this corpus's
    duplication is overwhelmingly reflowed rather than byte-identical, which is why
    the MinHash pass exists at all.
    """
    signatures: dict[str, tuple[int, ...]] = {}
    shingle_sets: dict[str, set[int]] = {}
    lengths: dict[str, int] = {}
    uf = _UnionFind()

    # Pass 1 — exact duplicates by normalised text.
    by_exact: dict[str, list[str]] = defaultdict(list)
    for item_id, text in items:
        normalised = " ".join(_WORD.findall(text.casefold()))
        by_exact[normalised].append(item_id)
        lengths[item_id] = len(text)
    for group in by_exact.values():
        for other in group[1:]:
            uf.union(group[0], other)

    # Pass 2 — near duplicates by MinHash + LSH banding.
    #
    # Buckets hold one representative per exact-duplicate group, not every item.
    # Pass 1 unions exact duplicates in the union-find but leaves them all in
    # `items`, so bucketing raw ids let a group of N byte-identical chunks put N
    # entries in the same bucket and make the pairwise check quadratic in N for no
    # information gain. Deduplicating by union-find root first is what the previous
    # member cap was reaching for, without a cap that could silently exclude a
    # genuine near-duplicate from every band.
    representatives: dict[str, str] = {}
    for item_id, text in items:
        shingle_sets[item_id] = shingles(text)
        signatures[item_id] = signature(shingle_sets[item_id])
        representatives.setdefault(uf.find(item_id), item_id)

    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for item_id in representatives.values():
        sig = signatures[item_id]
        for band in range(LSH_BANDS):
            key = (band, sig[band * LSH_ROWS : (band + 1) * LSH_ROWS])
            buckets[key].append(item_id)

    checked: set[tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                pair = (a, b) if a < b else (b, a)
                if pair in checked:
                    continue
                checked.add(pair)
                # Screen with the estimate (cheap), decide with exact Jaccard.
                if estimated_jaccard(signatures[a], signatures[b]) < threshold - _SCREEN_MARGIN:
                    continue
                if exact_jaccard(shingle_sets[a], shingle_sets[b]) >= threshold:
                    uf.union(a, b)

    # Build clusters. Canonical = longest text (the most complete version), with
    # id as the tie-break so the choice is reproducible.
    groups: dict[str, list[str]] = defaultdict(list)
    for item_id, _ in items:
        groups[uf.find(item_id)].append(item_id)

    duplicate_of: dict[str, str] = {}
    clusters: list[DuplicateCluster] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = max(sorted(members), key=lambda m: lengths[m])
        # Exact minimum pairwise similarity across the whole cluster, not just
        # against the canonical member: union-find is transitive, so two members
        # may never have been compared directly, and "floor" must mean floor.
        floor = min(
            exact_jaccard(shingle_sets[x], shingle_sets[y])
            for i, x in enumerate(sorted(members))
            for y in sorted(members)[i + 1 :]
        )
        for m in members:
            if m != canonical:
                duplicate_of[m] = canonical
        clusters.append(
            DuplicateCluster(
                canonical_id=canonical,
                member_ids=sorted(members),
                size=len(members),
                similarity_floor=round(floor, 3),
            )
        )

    clusters.sort(key=lambda c: (-c.size, c.canonical_id))
    return duplicate_of, clusters
