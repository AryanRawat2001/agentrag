"""Dense retrieval over cached embeddings.

Search is **exact**, not approximate: a full matrix product against every chunk
vector. That is a deliberate choice for the eval harness, not a shortcut taken
because the corpus is small.

An ANN index has its own recall curve, so an approximate dense row conflates two
effects — how good the embedding model is, and how much the index gave up to be
fast. When the number in the table is supposed to answer "does dense retrieval
beat BM25 on this corpus", index error is a confound. Exact search removes it, and
at 16k chunks × 384 dims the whole product is a few milliseconds.

Qdrant still earns its place in the plan, but for the *serving* path in Phase 7,
where the question is latency at scale rather than what the model can do. Measuring
the ceiling first means Phase 7's ANN configuration can be checked against a known
exact baseline — the recall lost to approximation becomes its own reportable number
instead of being invisible.

Vectors are unit-norm from the provider, so the dot product *is* cosine similarity
and scores are already in [-1, 1]. That matters for fusion: unlike BM25's unbounded
corpus-dependent scores, dense scores are comparable across queries, which is why
the two need different treatment when their rankings are combined.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ragpipe import embedding as emb_mod
from ragpipe.retrieval import Hit, _index_text


class DenseRetriever:
    """Exact cosine retrieval over chunk embeddings, with a content-hash cache."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embedder: emb_mod.LocalEmbedder,
        *,
        name: str | None = None,
        heading_mode: str = "prepend",
        cache: emb_mod.VectorCache | None = None,
        count_truncation: bool = False,
    ) -> None:
        if heading_mode not in {"prepend", "source", "strip"}:
            raise ValueError(f"heading_mode must be prepend|source|strip, got {heading_mode!r}")
        self.embedder = embedder
        self.heading_mode = heading_mode
        self.name = name or f"dense {embedder.name}"
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        self._query_cache: dict[str, np.ndarray] = {}

        # Prepared once, then used for the cache key, the truncation count, and the
        # encode. All three must describe the same string; deriving them separately
        # is how a key stops matching the vector it names.
        prepared = [embedder.prepare_document(_index_text(c, heading_mode)) for c in chunks]

        # Truncation accounting runs on the *prepared* text — after normalization —
        # because that is what the model actually sees. Counting the raw text would
        # overstate the loss and credit the normalizer with nothing.
        self.truncation: dict[str, Any] | None = None
        if count_truncation and prepared:
            self.truncation = embedder.count_truncation(prepared).as_dict()

        keys = [emb_mod.cache_key(p, embedder) for p in prepared]
        if not chunks:
            # Short-circuit rather than asking the provider to encode nothing: it
            # keeps the matrix correctly shaped (0, dim) instead of (0,), and avoids
            # loading a model to answer a question with no possible answer.
            self.matrix = np.zeros((0, embedder.dim), dtype=np.float32)
        elif cache is None:
            self.matrix = embedder.encode_prepared(prepared)
        else:
            self.matrix = cache.get_or_encode(keys, prepared, embedder.encode_prepared)
        self.matrix = np.ascontiguousarray(self.matrix, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._chunk_ids)

    def _query_vector(self, query: str) -> np.ndarray:
        """Query vectors are memoized per retriever instance.

        The same query is encoded once per row otherwise: every fusion variant calls
        into this retriever, and a reranker calls its base again. Beyond the saved
        forward passes this makes rows *comparable* — each sees a bit-identical query
        vector, so a difference between fusion methods cannot be nondeterminism in
        the encoder.
        """
        vec = self._query_cache.get(query)
        if vec is None:
            vec = self.embedder.encode_queries([query])[0]
            self._query_cache[query] = vec
        return vec

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if not self._chunk_ids:
            return []
        qv = self._query_vector(query)
        scores = self.matrix @ qv
        k = max(1, min(k, len(self._chunk_ids)))

        # argpartition for the top-k window, then sort only that window.
        idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        idx = idx[np.argsort(-scores[idx], kind="stable")]

        return [
            Hit(chunk_id=self._chunk_ids[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(idx.tolist(), start=1)
        ]
