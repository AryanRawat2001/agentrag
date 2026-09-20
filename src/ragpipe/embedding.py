"""Embedding providers, an append-only vector cache, and the text normalization
that keeps a fixed context window from being wasted on punctuation.

Three things live here, and they are separate on purpose.

## 1. The provider interface

Dense retrieval is the first component in this project with a real cost curve, so
the model sits behind `EmbeddingProvider` rather than being called directly. That
buys three things the ablation table needs: a hosted API can be added as one row
without touching the retriever, model size becomes a variable instead of a
commitment, and the query/document asymmetry below can be a property of the
provider instead of a rule the caller has to remember.

## 2. Query/document asymmetry — the detail that silently halves recall

BGE v1.5 models are trained with an instruction prefix on the **query side only**:

    Represent this sentence for searching relevant passages: <query>

Documents get no prefix. Encode both sides the same way and retrieval still
"works" — it returns plausible neighbours and no error — it is just measurably
worse, because the query embedding lands in a different region of the space than
the one the model was trained to match against passages. BGE-M3 dropped the prefix
entirely, so the convention is **per model**, which is exactly why it belongs on
the provider. `query_prefix` is exposed so the ablation can measure it rather than
take the model card's word for it.

## 3. Normalization, and why it is applied to `embed_text` only

A fixed context window is a budget, and this corpus wastes it on punctuation.
Measured against `bge-small-en-v1.5` (512 tokens) before any normalization:

    fixed        33.1% of chunks exceed the window; 16.8% of corpus tokens truncated away
    structural   18.8% of chunks exceed the window; 16.0% of corpus tokens truncated away

Chunk *sizes* are not the cause — no chunk exceeds 2,464 characters against a
2,048 target (2,298 for `fixed` and `structural`). The cause is tokenization density. Median
regulatory text runs 4.54 chars/token, but the 5th percentile is 2.10 and the floor
is **1.16** — and the chunks at that floor are table-of-contents dot leaders, where
runs of `....................` turn one character into roughly one token. A chunk of
navigational filler consumes 1,819 tokens for 2,108 characters and evicts the
substantive text after it.

So truncation was not hitting long content, it was hitting punctuation. Collapsing
leader runs reclaims the window.

This is safe precisely because Phase 1b kept two fields per chunk: `text` is a
byte-exact source slice and is what citations resolve against, `embed_text` is the
retrieval input. Normalization touches only the latter, so character offsets — and
therefore every span in the golden set — are unaffected. That split was built for
heading prefixes; it pays off again here.

Truncation is **counted and reported**, never silent. A model quietly discarding a
sixth of the corpus is the kind of defect that produces a plausible ablation table
and a wrong conclusion.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

# Model registry. `window` is the token limit we hold the model to, and
# `query_prefix` its retrieval convention — both properties of the checkpoint, not
# of the caller.
MODELS: dict[str, dict[str, Any]] = {
    "bge-small": {
        "hf_id": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "window": 512,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "bge-base": {
        "hf_id": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "window": 512,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    # BGE-M3: 8192-token window, so the truncation above effectively disappears.
    # No query prefix — the M3 training recipe dropped it.
    "bge-m3": {
        "hf_id": "BAAI/bge-m3",
        "dim": 1024,
        "window": 8192,
        "query_prefix": "",
    },
}

DEFAULT_MODEL = "bge-small"

# Bump when normalization changes semantics: it is part of the cache key, so an
# old cache must not be silently reused against new preprocessing.
NORMALIZER_VERSION = 1

_LEADER_RUN = re.compile(r"([.·•_\-=~*])\s*(?:\1\s*){3,}")
_HSPACE_RUN = re.compile(r"[ \t]{3,}")
_BLANK_RUN = re.compile(r"(?:\r?\n\s*){3,}")


def normalize_for_embedding(text: str) -> str:
    """Collapse leader/rule runs so the context window holds content, not filler.

    Only ever applied to embedding input. Never to `text`, whose offsets are
    load-bearing for citations.
    """
    out = _LEADER_RUN.sub(r"\1 ", text)
    out = _HSPACE_RUN.sub(" ", out)
    out = _BLANK_RUN.sub("\n\n", out)
    return out.strip()


class EmbeddingProvider(Protocol):
    """Anything that turns text into unit-norm vectors."""

    name: str
    dim: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class TruncationStats:
    """How much text a model's context window discarded. Reported, not assumed."""

    n_texts: int = 0
    n_truncated: int = 0
    tokens_total: int = 0
    tokens_dropped: int = 0

    @property
    def pct_truncated(self) -> float:
        return 100.0 * self.n_truncated / self.n_texts if self.n_texts else 0.0

    @property
    def pct_tokens_dropped(self) -> float:
        return 100.0 * self.tokens_dropped / self.tokens_total if self.tokens_total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_texts": self.n_texts,
            "n_truncated": self.n_truncated,
            "pct_truncated": round(self.pct_truncated, 2),
            "tokens_total": self.tokens_total,
            "tokens_dropped": self.tokens_dropped,
            "pct_tokens_dropped": round(self.pct_tokens_dropped, 2),
        }


class LocalEmbedder:
    """`sentence-transformers` backend, on MPS when available.

    Local rather than hosted so that re-embedding is free. The chunking sweep, the
    heading-mode comparison, and the chunk-size sweep in Phase 4 all re-embed the
    corpus; at API prices each would be a decision about budget instead of a
    measurement. A hosted provider is still worth one ablation row, which is what
    the interface is for.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 64,
        use_query_prefix: bool = True,
        normalize_text: bool = True,
    ) -> None:
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; known: {sorted(MODELS)}")
        spec = MODELS[model]
        self.model = model
        self.hf_id: str = spec["hf_id"]
        self.dim: int = spec["dim"]
        self.window: int = spec["window"]
        self.query_prefix: str = spec["query_prefix"] if use_query_prefix else ""
        self.use_query_prefix = use_query_prefix
        self.normalize_text = normalize_text
        self.batch_size = batch_size
        self._device = device
        self._st: Any = None
        self._tokenizer: Any = None
        self.doc_truncation = TruncationStats()

        bits = [model]
        if not use_query_prefix:
            bits.append("noprefix")
        if not normalize_text:
            bits.append("rawtext")
        self.name = "+".join(bits)

    # -- lazy model load -------------------------------------------------------
    # Deferred so that a fully cached run never pays the load cost, and so that
    # constructing a provider stays cheap enough to do in a config list.

    @property
    def st(self) -> Any:
        if self._st is None:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(self.hf_id, device=self._device or _best_device())
            # Hold the model to the window we report, rather than trusting the
            # checkpoint's default to match the model card.
            self._st.max_seq_length = min(self.window, self._st.max_seq_length or self.window)
        return self._st

    def prepare_document(self, text: str) -> str:
        return normalize_for_embedding(text) if self.normalize_text else text

    def prepare_query(self, text: str) -> str:
        body = normalize_for_embedding(text) if self.normalize_text else text
        return self.query_prefix + body

    @property
    def tokenizer(self) -> Any:
        """The tokenizer alone, without instantiating the model.

        Truncation accounting needs only tokenization, but reading
        `self.st.tokenizer` loads the full `SentenceTransformer` — which defeated the
        lazy-load rationale two properties up, since `cmd_eval` counts truncation on
        every run. For a 33M model that is a second; for the 568M `bge-m3` the plan
        is heading to it is not.
        """
        if self._tokenizer is None:
            if self._st is not None:
                self._tokenizer = self._st.tokenizer
            else:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        return self._tokenizer

    def count_truncation(self, texts: list[str]) -> TruncationStats:
        """Token accounting against the window, without embedding anything."""
        tk = self.tokenizer
        stats = TruncationStats()
        for text in texts:
            n = len(tk(text, add_special_tokens=True, truncation=False)["input_ids"])
            stats.n_texts += 1
            stats.tokens_total += n
            if n > self.window:
                stats.n_truncated += 1
                stats.tokens_dropped += n - self.window
        return stats

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.st.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # unit norm, so dot product *is* cosine
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        prepared = [self.prepare_document(t) for t in texts]
        return self._encode(prepared)

    def encode_prepared(self, texts: list[str]) -> np.ndarray:
        """Encode text that has already been through `prepare_document`.

        Exists so the cache can key on the prepared string and then hand that same
        string to the model. Preparing twice would be harmless (normalization is
        idempotent) but it would mean the key and the encoded text are produced by
        two separate calls, which is exactly the kind of seam that drifts.
        """
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.prepare_query(t) for t in texts])


def _best_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class VectorCache:
    """Append-only content-addressed vector store.

    Keyed on the chunk's `content_hash` plus everything that changes the vector —
    model, prefix convention, normalizer version, and which text variant was
    embedded. Two consequences that matter more than the disk saving:

    * Re-running the eval is free, so the retrieval table can gate every change,
      which is the premise the whole phase ordering rests on.
    * Chunks whose text did not change keep their vectors when the corpus is
      re-indexed. That is the same `content_hash` Phase 1b introduced for
      incremental indexing, doing its second job.

    Rows are only ever appended *in memory*, so a partial run never invalidates
    vectors it already holds. Two caveats worth stating rather than implying:

    * `flush()` rewrites the whole array, so it is O(N) on disk, not an append.
    * That makes it **single-writer**. Two processes sharing one cache file each
      flush their own in-memory state, and the later flush wins outright — the
      other's new vectors are gone. Not reachable from `cmd_eval`, which handles one
      strategy at a time against a per-`{model}-{strategy}` file, but two concurrent
      `make eval` runs would silently lose work.
    """

    path: Path
    dim: int
    _keys: dict[str, int] = field(default_factory=dict)
    _vectors: list[np.ndarray] = field(default_factory=list)
    _dirty: bool = False
    hits: int = 0
    misses: int = 0
    deduped: int = 0  # requested keys that repeated another key in the same call

    @classmethod
    def open(cls, path: Path, dim: int) -> VectorCache:
        cache = cls(path=path, dim=dim)
        vec_path, key_path = cache._paths()
        if vec_path.exists() and key_path.exists():
            try:
                keys = json.loads(key_path.read_text(encoding="utf-8"))
                arr = np.load(vec_path)
                # A dim change means a different model wrote this file: discard
                # rather than mix vector spaces, which would be unrecoverable
                # nonsense rather than an error.
                if arr.ndim == 2 and arr.shape[1] == dim and len(keys) == arr.shape[0]:
                    cache._keys = {k: i for i, k in enumerate(keys)}
                    cache._vectors = [row for row in arr]
            except (OSError, ValueError, json.JSONDecodeError):
                pass  # unreadable cache is a cold cache, not a failure
        return cache

    def _paths(self) -> tuple[Path, Path]:
        # String concatenation, not `with_suffix`: the latter replaces everything
        # after the last dot, so a model or strategy name containing one (`v1.5`)
        # would silently write to the wrong file.
        return (
            self.path.with_name(self.path.name + ".npy"),
            self.path.with_name(self.path.name + ".keys.json"),
        )

    def get_or_encode(self, keys: list[str], texts: list[str], encode: Any) -> np.ndarray:
        """Vectors for `keys`, encoding only what is missing.

        Duplicate keys within one call are encoded once — this corpus has exact
        duplicate chunks (regulatory boilerplate), so that is a real saving.
        """
        if len(keys) != len(texts):
            raise ValueError(f"keys/texts length mismatch: {len(keys)} vs {len(texts)}")

        missing: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in self._keys and key not in missing:
                missing[key] = text

        # Three counters on one denominator: hits + encoded + deduped == len(keys).
        # An earlier version counted `hits` in key *occurrences* and `misses` in
        # *distinct* new keys, so the printed line read as an accounting of every
        # chunk while silently omitting within-call duplicates — 9,409 encoded
        # against 9,413 requested, with nothing explaining the gap.
        self.hits += sum(1 for k in keys if k in self._keys)
        self.misses += len(missing)
        self.deduped += len(keys) - sum(1 for k in keys if k in self._keys) - len(missing)

        if missing:
            new_keys = list(missing)
            new_vecs = encode([missing[k] for k in new_keys])
            if new_vecs.shape[0] != len(new_keys):
                raise ValueError(
                    f"encoder returned {new_vecs.shape[0]} vectors for {len(new_keys)} texts"
                )
            # Dimension is checked here, not only on reload. Accepting a
            # wrong-width vector writes a file that `open()` then discards on every
            # subsequent run — so the corpus is silently re-embedded forever, which
            # breaks the "re-running the eval is free" premise without raising.
            if new_vecs.ndim != 2 or new_vecs.shape[1] != self.dim:
                raise ValueError(
                    f"encoder returned vectors of shape {new_vecs.shape}, expected (*, {self.dim})"
                )
            for key, vec in zip(new_keys, new_vecs, strict=True):
                self._keys[key] = len(self._vectors)
                self._vectors.append(np.asarray(vec, dtype=np.float32))
            self._dirty = True

        return np.stack([self._vectors[self._keys[k]] for k in keys])

    def flush(self) -> None:
        if not self._dirty:
            return
        vec_path, key_path = self._paths()
        vec_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self._keys.items(), key=lambda kv: kv[1])
        arr = np.stack(self._vectors) if self._vectors else np.zeros((0, self.dim), np.float32)
        # Atomic: a crash mid-write must not leave vectors and keys disagreeing.
        tmp_vec = vec_path.with_name(vec_path.name + ".tmp")
        tmp_key = key_path.with_name(key_path.name + ".tmp")
        # Write through an open handle. `np.save` appends `.npy` to a *path* that
        # lacks it, so passing `foo.npy.tmp` silently creates `foo.npy.tmp.npy` and
        # the rename below then fails on a file that was never written.
        with open(tmp_vec, "wb") as fh:
            np.save(fh, arr)
        tmp_key.write_text(json.dumps([k for k, _ in ordered]), encoding="utf-8")
        tmp_vec.replace(vec_path)
        tmp_key.replace(key_path)
        self._dirty = False

    def __len__(self) -> int:
        return len(self._keys)


def cache_key(prepared_text: str, embedder: LocalEmbedder) -> str:
    """Key on the exact text being embedded, plus the model that will embed it.

    Takes the **prepared** text — post-normalization, post-heading — rather than the
    chunk's `content_hash`, and that distinction was a real defect. `content_hash`
    hashes `text`, but under `heading_mode="prepend"` the embedded string is
    `heading + "\\n\\n" + body`, and the heading appeared nowhere in the key: only the
    *mode name* did. Two chunks with byte-identical bodies under different headings
    therefore collided, and the second silently received the first's vector. Two such
    groups exist in `structural`.

    `content_hash` also case-folds and collapses whitespace, which is invisible under
    `bge-small-en-v1.5` (uncased WordPiece) but **not** under `bge-m3`, whose
    SentencePiece tokenizer is cased — and `bge-m3` is the plan's destination model.
    35 groups in `semantic` differ only by case or whitespace, so that latent bug
    would have surfaced as wrong vectors the moment the intended model was used.

    Hashing the prepared text removes all of it: the key is exactly what is embedded.
    Incremental re-indexing still works, since unchanged text prepares identically.
    `content_hash` keeps its Phase 1b job of change detection; it was never a safe
    embedding key.
    """
    parts = (
        embedder.model,
        f"norm{NORMALIZER_VERSION if embedder.normalize_text else 0}",
        f"w{embedder.window}",
        f"prefix{int(embedder.use_query_prefix)}",
        prepared_text,
    )
    # NUL-delimited: it cannot occur in the prepared text, so no combination of
    # field values can be made to collide with a different combination.
    return hashlib.blake2b("\x00".join(parts).encode("utf-8"), digest_size=16).hexdigest()
