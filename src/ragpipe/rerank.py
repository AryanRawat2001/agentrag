"""Cross-encoder reranking over a first-stage candidate list.

The single highest-leverage component in production RAG, and absent from the source
guide's stack table.

## Why it works, stated precisely

A bi-encoder embeds the query and the document *independently*. Every chunk vector
is computed once, before any query exists, so the model can never condition the
document representation on what was asked. That independence is exactly what makes
dense retrieval cheap — one matrix product serves every query — and exactly what
caps its precision.

A cross-encoder concatenates query and document and runs them through the model
together, so attention operates across the pair and the score can depend on their
interaction. It cannot be precomputed: N documents means N forward passes *per
query*. That is why it reranks a shortlist instead of replacing retrieval.

So the two stages are not competing implementations of one idea; they trade recall
against precision in opposite directions, and the architecture that wins is
"retrieve wide and cheap, then rerank narrow and expensive."

## What the ablation must control for

Reranking **cannot improve recall@k for k >= candidate_k** — it only permutes a list
the first stage already fixed. Its ceiling is that list. A rerank row that appears to
lift recall@10 above its first stage by more than reordering can explain is a bug,
not a result, so `candidate_k` is recorded on every row and the recall ceiling is
reported alongside.

This is also why `candidate_k` is the interesting knob rather than an
implementation detail: it sets the recall ceiling the reranker is permuting within,
and it is the direct cost/quality dial (latency grows linearly in it).

`LLMReranker` is deliberately not implemented here. Per deviation #6 it is worth at
most one comparison row — milliseconds and free versus seconds and metered — and it
belongs with the other metered calls in Phase 5.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ragpipe.retrieval import Hit, _index_text

RERANKERS: dict[str, dict[str, Any]] = {
    "bge-reranker-base": {"hf_id": "BAAI/bge-reranker-base", "window": 512},
    "bge-reranker-v2-m3": {"hf_id": "BAAI/bge-reranker-v2-m3", "window": 8192},
    "minilm-l6": {"hf_id": "cross-encoder/ms-marco-MiniLM-L-6-v2", "window": 512},
}

DEFAULT_RERANKER = "bge-reranker-base"
DEFAULT_CANDIDATE_K = 50


class CrossEncoderReranker:
    """Reranks a first-stage retriever's candidates with a cross-encoder."""

    def __init__(
        self,
        base: Any,
        chunks: list[dict[str, Any]],
        *,
        model: str = DEFAULT_RERANKER,
        name: str | None = None,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        heading_mode: str = "prepend",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if model not in RERANKERS:
            raise ValueError(f"unknown reranker {model!r}; known: {sorted(RERANKERS)}")
        if candidate_k < 1:
            raise ValueError(f"candidate_k must be >= 1, got {candidate_k}")
        self.base = base
        self.model = model
        self.hf_id: str = RERANKERS[model]["hf_id"]
        self.candidate_k = candidate_k
        self.batch_size = batch_size
        self._device = device
        self._ce: Any = None
        # Text is looked up by chunk id: the base retriever returns ids, and the
        # cross-encoder needs the text those ids stand for.
        self._text = {c["chunk_id"]: _index_text(c, heading_mode) for c in chunks}
        self.name = name or f"{base.name} + rerank {model}(c{candidate_k})"
        # The candidate ids from the most recent `search`. Exposed so the harness can
        # compute this row's recall ceiling from the list actually scored, rather
        # than by re-running the base retriever and hoping it returned the same one.
        self.last_candidate_ids: list[str] = []
        # The window actually scored on the most recent search. Equals `candidate_k`
        # unless the caller asked for more results than that, or the corpus held
        # fewer candidates than the window.
        self.effective_candidate_k: int = candidate_k

    @property
    def ce(self) -> Any:
        if self._ce is None:
            from sentence_transformers import CrossEncoder

            from ragpipe.embedding import _best_device

            self._ce = CrossEncoder(
                self.hf_id, device=self._device or _best_device(), max_length=512
            )
        return self._ce

    def __len__(self) -> int:
        return len(self.base)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        # Retrieve at least `k` even if candidate_k is smaller, so the reranker can
        # never shrink the result list below what was asked for. `candidate_k` is
        # therefore a *floor* on the window, not a ceiling on it, and the effective
        # window is what the ceiling must be computed over.
        #
        # Getting this wrong was subtle: scoring `max(candidate_k, k)` while measuring
        # the ceiling over `candidate_k` meant any `--candidate-k` below `max(K_VALUES)`
        # (20) produced rows whose recall legitimately exceeded their own printed
        # ceiling — which the report prose declares to be a wiring bug. Default 50 hid
        # it; the first `--candidate-k 10` sweep row would not have.
        window = max(self.candidate_k, k)
        candidates = self.base.search(query, k=window)
        self.effective_candidate_k = len(candidates)
        self.last_candidate_ids = [h.chunk_id for h in candidates]
        if not candidates:
            return []

        pairs = [(query, self._text.get(h.chunk_id, "")) for h in candidates]
        scores = np.asarray(
            self.ce.predict(pairs, batch_size=self.batch_size, show_progress_bar=False),
            dtype=np.float32,
        ).reshape(-1)

        # Ties broken on chunk_id, matching fusion, so a ranking is reproducible.
        order = sorted(
            range(len(candidates)),
            key=lambda i: (-float(scores[i]), candidates[i].chunk_id),
        )
        return [
            Hit(chunk_id=candidates[i].chunk_id, score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order[:k], start=1)
        ]


def recall_ceiling(base_hits: list[Hit], relevant: set[str], candidate_k: int) -> float:
    """The best recall any reranker could reach on this candidate list.

    Reported so a rerank row is read against what was reachable rather than against
    1.0. If a rerank row exceeds its own ceiling, the wiring is wrong.
    """
    if not relevant:
        return 0.0
    window = {h.chunk_id for h in base_hits[:candidate_k]}
    return len(window & relevant) / len(relevant)


def ceiling_from_ids(candidate_ids: list[str], relevant: set[str], candidate_k: int) -> float:
    """`recall_ceiling` over ids the reranker actually scored.

    The harness uses this rather than re-running the base retriever: re-running it
    would compute a ceiling for a list that might differ from the one reranked, and
    a ceiling that does not describe the row it labels is worse than none.
    """
    if not relevant:
        return 0.0
    return len(set(candidate_ids[:candidate_k]) & relevant) / len(relevant)
