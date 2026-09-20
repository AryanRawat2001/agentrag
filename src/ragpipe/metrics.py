"""Retrieval metrics.

Deliberately small and dependency-free, because every number in the eventual
ablation table passes through here. A subtle error in nDCG would be invisible —
the table would still look plausible — so each function is short enough to read in
one sitting and is pinned by tests against hand-computed values.

## Which metric answers which question

  `hit@k`       Did *any* relevant chunk make the top k? The right primary metric
                when a query has many relevant chunks (a document-title lookup
                matches every chunk of that document, so recall@10 is bounded far
                below 1 by construction and would read as failure).
  `recall@k`    What share of all relevant chunks made the top k? The right
                primary metric when the relevant set is small and complete
                retrieval matters.
  `precision@k` What share of the top k was relevant? Matters once a reranker or a
                generation context budget is involved.
  `mrr`         How high did the *first* relevant chunk rank? Sensitive to the top
                of the ranking, which is what a user actually sees.
  `ndcg@k`      Rank-discounted gain over the ideal ranking. The one metric that
                credits retrieving several relevant chunks *and* ranking them well.

Queries with no relevant chunks (the unanswerable slice) are excluded from all of
the above — every one of them is undefined on an empty relevant set. They are
scored separately, on whether the retriever's top score is low enough to be
separable from answerable queries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryResult:
    """One query's ranked chunk ids alongside its ground truth."""

    query_id: str
    slice_name: str
    ranked_ids: list[str]  # best first
    relevant_ids: frozenset[str]
    top_score: float

    @property
    def is_answerable(self) -> bool:
        return bool(self.relevant_ids)


def hit_at_k(ranked: list[str], relevant: frozenset[str], k: int) -> float:
    return 1.0 if any(cid in relevant for cid in ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], relevant: frozenset[str], k: int) -> float:
    """Share of relevant items retrieved in the top k.

    Counts *distinct* ids: counting positions let a retriever that returned the same
    chunk twice score above 1.0. No current retriever repeats, but a fused or
    deduplicated one could, and an impossible recall value is the kind of thing that
    would be read as a win.
    """
    if not relevant:
        return 0.0
    found = len({cid for cid in ranked[:k] if cid in relevant})
    return found / len(relevant)


def precision_at_k(ranked: list[str], relevant: frozenset[str], k: int) -> float:
    if k <= 0:
        return 0.0
    # Denominator is k, not len(ranked[:k]): a retriever returning 3 results for
    # k=10 has not earned the precision of one returning 10.
    return sum(1 for cid in ranked[:k] if cid in relevant) / k


def reciprocal_rank(ranked: list[str], relevant: frozenset[str], k: int | None = None) -> float:
    limit = len(ranked) if k is None else k
    for i, cid in enumerate(ranked[:limit], start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: frozenset[str], k: int) -> float:
    """Binary-relevance nDCG@k.

    DCG uses the standard 1/log2(rank+1) discount. The ideal ranking places
    min(len(relevant), k) relevant items first, so a query with more relevant
    chunks than k can still score 1.0 — without that cap, nDCG would penalise a
    perfect ranking for the size of its own ground truth.
    """
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1) for i, cid in enumerate(ranked[:k], start=1) if cid in relevant
    )
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


K_VALUES = (1, 5, 10, 20)


def score_query(result: QueryResult) -> dict[str, float]:
    """All metrics for one answerable query."""
    ranked, relevant = result.ranked_ids, result.relevant_ids
    scores: dict[str, float] = {}
    for k in K_VALUES:
        scores[f"hit@{k}"] = hit_at_k(ranked, relevant, k)
        scores[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        scores[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
    scores["mrr@10"] = reciprocal_rank(ranked, relevant, 10)
    scores["ndcg@10"] = ndcg_at_k(ranked, relevant, 10)
    return scores


def aggregate(results: list[QueryResult]) -> dict[str, float]:
    """Mean of each metric over answerable queries.

    Unanswerable queries are excluded: every metric here is undefined on an empty
    relevant set, and averaging in a structural zero would silently drag the whole
    table down in proportion to how many unanswerable queries the set happens to
    contain.
    """
    answerable = [r for r in results if r.is_answerable]
    if not answerable:
        return {"n_queries": 0}

    per_query = [score_query(r) for r in answerable]
    out: dict[str, float] = {"n_queries": float(len(answerable))}
    for metric in per_query[0]:
        out[metric] = sum(p[metric] for p in per_query) / len(per_query)
    return out


def separability(results: list[QueryResult]) -> dict[str, float]:
    """How well top-1 score separates answerable from unanswerable queries.

    This is the unanswerable slice's actual purpose. It is not a retrieval metric —
    it is the evidence for whether Phase 5 can refuse on a score threshold rather
    than guessing one. Reported as the best achievable accuracy over all candidate
    thresholds, with the threshold that achieves it.
    """
    answerable = [r.top_score for r in results if r.is_answerable]
    unanswerable = [r.top_score for r in results if not r.is_answerable]
    if not answerable or not unanswerable:
        return {"n_answerable": len(answerable), "n_unanswerable": len(unanswerable)}

    # Include a threshold above every observed score, so the degenerate
    # "predict everything unanswerable" classifier is reachable. Without it the
    # search could not represent that corner and could report a worse-than-baseline
    # optimum as the best available.
    candidates = sorted({*answerable, *unanswerable})
    candidates.append(max(candidates) + 1.0)
    best_acc, best_threshold = 0.0, 0.0
    total = len(answerable) + len(unanswerable)
    for threshold in candidates:
        # Predict "answerable" when top score >= threshold.
        correct = sum(1 for s in answerable if s >= threshold) + sum(
            1 for s in unanswerable if s < threshold
        )
        if correct / total > best_acc:
            best_acc, best_threshold = correct / total, threshold

    # The majority-class baseline. Without it a headline accuracy is unreadable:
    # 0.879 against a 0.750 baseline is a far weaker result than it appears.
    total_n = len(answerable) + len(unanswerable)
    baseline = max(len(answerable), len(unanswerable)) / total_n

    return {
        "n_answerable": float(len(answerable)),
        "n_unanswerable": float(len(unanswerable)),
        "majority_baseline": baseline,
        "gain_over_baseline": best_acc - baseline,
        "answerable_mean_top1": sum(answerable) / len(answerable),
        "unanswerable_mean_top1": sum(unanswerable) / len(unanswerable),
        "best_threshold": best_threshold,
        "best_accuracy": best_acc,
    }
