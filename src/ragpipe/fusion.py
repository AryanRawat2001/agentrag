"""Rank fusion: three methods that the source guide treats as one.

The guide says "implement RRF ... make the weighting configurable (0.7 dense / 0.3
sparse)". Those are two different algorithms in one sentence, and picking the
difference apart is the whole point of this module.

## Vanilla RRF

    score(d) = sum over lists of 1 / (rank_constant + rank(d))

No per-list weights, and it never looks at a score — only at positions. That is the
entire appeal: BM25 returns unbounded corpus-dependent scores and a cosine returns
[-1, 1], so the two are not on a common scale and no fixed weighting makes them
comparable across queries. Discarding magnitude sidesteps the problem instead of
tuning around it. Its one knob is `rank_constant`, which sets how fast rank
influence decays — the conventional 60 means rank 1 (1/61) is worth only ~1.6x rank
10 (1/70), so RRF is deliberately flat. A document ranked well by *both* retrievers
beats one ranked first by either.

Discarding magnitude is also its cost: RRF cannot tell a runaway top hit from a
narrow win, so on the `exact_identifier` slice — where BM25's top hit is often
correct by a wide margin — it throws away exactly the signal that matters. That
tradeoff is why all three methods get a row rather than an argument.

## Weighted RRF

Same rank-only reciprocal, with a per-list multiplier. Restores the ability to say
"trust dense more" without reintroducing scale sensitivity. This is what people
usually mean by "weighted RRF", and it is *not* what 0.7/0.3 score weighting does.

## Min-max weighted score fusion

Normalize each list's scores to [0, 1], then take a weighted sum. This is what the
guide's 0.7/0.3 actually describes. It keeps magnitude — the thing RRF throws away —
and pays for it with two failure modes that are worth knowing out loud:

1. **The window defines the scale.** Min-max over a truncated top-k list normalizes
   against whatever happens to be in that window, not the corpus. Retrieve 10
   instead of 100 and every score changes, because the divisor changed.

   Precisely: RRF's **scores** are window-invariant — a document at rank 7 scores
   `1/67` whether the window is 10 or 1000 — while min-max's are not. Neither
   method's **ranking** is window-invariant, since widening the window admits
   documents that were not there before; see `HybridRetriever.fetch_k` below, which
   exists for exactly that reason. An earlier version of this note claimed RRF's
   ranking was window-independent, which contradicted the `fetch_k` docstring in
   this same file.
2. **Degenerate spreads.** One hit, or several tied hits, makes `max == min` and the
   normalization undefined. Mapping a single hit to 1.0 flatters it — it was the
   only candidate, not a confident one. This implementation maps a zero-spread list
   to 0.5, so a lone hit carries middling rather than maximal weight.

Weights are normalized to sum to 1 so that a fused score stays in [0, 1] and rows
with different weight pairs stay comparable.

All three take ranked hit lists and return a fused ranking, so any of them drops
into the eval harness as one more retriever.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ragpipe.retrieval import Hit

DEFAULT_RANK_CONSTANT = 60


def _normalized_weights(n: int, weights: Sequence[float] | None) -> list[float]:
    if weights is None:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"expected {n} weights, got {len(weights)}")
    if any(w < 0 for w in weights):
        raise ValueError(f"weights must be non-negative, got {list(weights)}")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must not sum to zero")
    return [float(w) / total for w in weights]


def _ranked(fused: dict[str, float], k: int) -> list[Hit]:
    # Ties broken on chunk_id so a fused ranking is deterministic. Without it, two
    # runs can differ on tied documents and a metric moves for no reason.
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        Hit(chunk_id=cid, score=score, rank=rank)
        for rank, (cid, score) in enumerate(ordered[:k], start=1)
    ]


def rrf(
    hit_lists: Sequence[Sequence[Hit]],
    *,
    k: int = 10,
    rank_constant: int = DEFAULT_RANK_CONSTANT,
) -> list[Hit]:
    """Vanilla Reciprocal Rank Fusion. No weights — that is the point."""
    if rank_constant <= 0:
        raise ValueError(f"rank_constant must be positive, got {rank_constant}")
    fused: dict[str, float] = {}
    for hits in hit_lists:
        for hit in hits:
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (rank_constant + hit.rank)
    return _ranked(fused, k)


def weighted_rrf(
    hit_lists: Sequence[Sequence[Hit]],
    weights: Sequence[float] | None = None,
    *,
    k: int = 10,
    rank_constant: int = DEFAULT_RANK_CONSTANT,
) -> list[Hit]:
    """RRF with per-list weights. Still rank-only, so still scale-free."""
    if rank_constant <= 0:
        raise ValueError(f"rank_constant must be positive, got {rank_constant}")
    ws = _normalized_weights(len(hit_lists), weights)
    fused: dict[str, float] = {}
    for weight, hits in zip(ws, hit_lists, strict=True):
        for hit in hits:
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + weight / (rank_constant + hit.rank)
    return _ranked(fused, k)


def minmax_weighted(
    hit_lists: Sequence[Sequence[Hit]],
    weights: Sequence[float] | None = None,
    *,
    k: int = 10,
    missing: float = 0.0,
) -> list[Hit]:
    """Min-max normalize each list, then weighted-sum. Keeps magnitude.

    `missing` is the score assigned to a document absent from a list. Zero is the
    honest default given the normalization: absent means "did not make this
    retriever's window", which is at best the bottom of it.
    """
    ws = _normalized_weights(len(hit_lists), weights)

    normalized: list[dict[str, float]] = []
    for hits in hit_lists:
        if not hits:
            normalized.append({})
            continue
        scores = [h.score for h in hits]
        lo, hi = min(scores), max(scores)
        spread = hi - lo
        if spread <= 0:
            # Zero spread: one hit, or all tied. Mapping to 1.0 would treat a lone
            # candidate as a confident one. 0.5 says "present, magnitude unknown".
            normalized.append({h.chunk_id: 0.5 for h in hits})
        else:
            normalized.append({h.chunk_id: (h.score - lo) / spread for h in hits})

    all_ids = {cid for norm in normalized for cid in norm}
    fused = {
        cid: sum(w * norm.get(cid, missing) for w, norm in zip(ws, normalized, strict=True))
        for cid in all_ids
    }
    return _ranked(fused, k)


METHODS = {
    "rrf": rrf,
    "weighted_rrf": weighted_rrf,
    "minmax": minmax_weighted,
}


class HybridRetriever:
    """Two or more retrievers, fused into one ranking.

    `fetch_k` is retrieved from each component and deliberately larger than the `k`
    returned. Fusing only the final top-10 of each list caps how far a document can
    be promoted: something ranked 15th by BM25 and 12th by dense — a strong hybrid
    candidate, exactly the case fusion exists to catch — would be invisible to both.
    A wider component window is what gives fusion anything to work with.
    """

    def __init__(
        self,
        components: Sequence[Any],
        *,
        method: str = "rrf",
        weights: Sequence[float] | None = None,
        name: str | None = None,
        fetch_k: int = 100,
        rank_constant: int = DEFAULT_RANK_CONSTANT,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}, got {method!r}")
        if len(components) < 2:
            raise ValueError("fusion needs at least two retrievers")
        self.components = list(components)
        self.method = method
        self.weights = list(weights) if weights is not None else None
        self.fetch_k = fetch_k
        self.rank_constant = rank_constant
        self.name = name or self._default_name()

    def _default_name(self) -> str:
        parts = "+".join(c.name for c in self.components)
        label = f"{parts} {self.method}"
        if self.weights is not None:
            ws = _normalized_weights(len(self.components), self.weights)
            label += "(" + "/".join(f"{w:.2f}".rstrip("0").rstrip(".") for w in ws) + ")"
        return label

    def __len__(self) -> int:
        return max((len(c) for c in self.components), default=0)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        hit_lists = [c.search(query, k=self.fetch_k) for c in self.components]
        if self.method == "rrf":
            return rrf(hit_lists, k=k, rank_constant=self.rank_constant)
        if self.method == "weighted_rrf":
            return weighted_rrf(hit_lists, self.weights, k=k, rank_constant=self.rank_constant)
        return minmax_weighted(hit_lists, self.weights, k=k)
