"""Embedding-layer tests: normalization, the vector cache, and dense ranking.

No model is loaded here. A fake provider stands in, so these run in milliseconds
and test the parts that can be wrong without being obviously broken — cache keying,
resumability, and whether the retriever's ranking actually reflects the vectors.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ragpipe import embedding as em
from ragpipe.dense import DenseRetriever


class FakeEmbedder:
    """Deterministic vectors from text, so ranking is predictable.

    Mirrors the real provider's contract: unit-norm output, a query prefix applied
    on the query side only, and normalization applied to both.
    """

    def __init__(self, dim: int = 8, *, query_prefix: str = "Q: ", normalize_text: bool = True):
        self.dim = dim
        self.model = "fake"
        self.name = "fake"
        self.window = 16
        self.query_prefix = query_prefix
        self.use_query_prefix = bool(query_prefix)
        self.normalize_text = normalize_text
        self.calls: list[list[str]] = []

    def prepare_document(self, text: str) -> str:
        return em.normalize_for_embedding(text) if self.normalize_text else text

    def prepare_query(self, text: str) -> str:
        return self.query_prefix + self.prepare_document(text)

    def _vec(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.stack([self._vec(self.prepare_document(t)) for t in texts])

    def encode_prepared(self, texts: list[str]) -> np.ndarray:
        """Already prepared — encode verbatim. Preparing again here would hide a
        mismatch between the string that was keyed and the string that was encoded."""
        self.calls.append(list(texts))
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(self.prepare_query(t)) for t in texts])


@pytest.fixture
def real_tokenizer():
    """A genuinely loaded model, or a skip.

    Most tests here use `FakeEmbedder` and need no model at all. Token accounting is
    the exception: the whole claim is about how a *specific* tokenizer fragments
    regulatory text, so faking it would assert nothing. Loading it needs the model
    on disk and a working TLS trust store, neither of which is guaranteed in a
    sandboxed or offline run — so skip rather than fail, and keep the rest of the
    suite hermetic.
    """
    embedder = em.LocalEmbedder("bge-small")
    try:
        embedder.st  # noqa: B018 — triggers the lazy load
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"bge-small unavailable ({type(exc).__name__}); needs model + TLS access")
    return embedder


@pytest.fixture
def real_tokenizer_only():
    """An embedder whose tokenizer has loaded but whose model has not."""
    embedder = em.LocalEmbedder("bge-small")
    try:
        embedder.tokenizer  # noqa: B018 — triggers the tokenizer-only load
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"bge-small tokenizer unavailable ({type(exc).__name__})")
    return embedder


def chunk(cid: str, text: str, *, heading: str | None = None, chash: str | None = None) -> dict:
    return {
        "chunk_id": cid,
        "doc_id": "d1",
        "text": text,
        "embed_text": f"{heading}\n\n{text}" if heading else text,
        "section_heading": heading,
        "content_hash": chash or f"h-{cid}",
        "identifiers": [],
        "strategy": "fixed",
    }


# -- normalization ----------------------------------------------------------


def test_normalize_collapses_toc_dot_leaders():
    """The measured cause of truncation: leader runs cost ~1 token per character."""
    raw = "4. INTRODUCTION ......................................................22"
    out = em.normalize_for_embedding(raw)
    assert "..." not in out
    assert "INTRODUCTION" in out and "22" in out
    assert len(out) < len(raw) / 2


@pytest.mark.parametrize("fill", [".", "_", "-", "=", "~", "*", "·", "•"])
def test_normalize_collapses_every_leader_character(fill):
    out = em.normalize_for_embedding(f"Section {fill * 40} 12")
    assert fill * 4 not in out
    assert "Section" in out and "12" in out


def test_normalize_preserves_short_runs_and_real_punctuation():
    """Ellipses and hyphenation are meaning, not filler. Only runs are collapsed."""
    assert em.normalize_for_embedding("wait... really?") == "wait... really?"
    assert em.normalize_for_embedding("a well-known dose-response curve") == (
        "a well-known dose-response curve"
    )


def test_normalize_collapses_runs_of_spaces_and_blank_lines():
    assert em.normalize_for_embedding("a      b") == "a b"
    assert em.normalize_for_embedding("a\n\n\n\n\nb") == "a\n\nb"


def test_normalize_is_idempotent():
    once = em.normalize_for_embedding("Title ........... 4\n\n\n\nBody     text")
    assert em.normalize_for_embedding(once) == once


def test_normalize_never_applied_to_citation_text():
    """The guarantee that makes normalization safe: `text` offsets are load-bearing
    for citations, so only embedding input may be rewritten.

    An earlier version of this test ended in `or True`, so it asserted nothing at
    all — it would have passed even if the retriever had mutated the chunk.
    """
    body = "Heading ....... 3\nThe requirement applies."
    c = chunk("c1", body, heading="Heading")
    e = FakeEmbedder()
    DenseRetriever([c], e, heading_mode="source")

    # The chunk dict is untouched, offsets and all.
    assert c["text"] == body
    assert "......." in c["text"]
    # What reached the model was normalized.
    assert e.calls, "expected the embedder to be called"
    assert "......." not in e.calls[0][0]
    assert "The requirement applies." in e.calls[0][0]


# -- query/document asymmetry ----------------------------------------------


def test_query_prefix_applied_to_queries_only():
    """BGE v1.5 prefixes queries and not documents. Getting this backwards degrades
    retrieval silently — no error, just worse numbers."""
    e = FakeEmbedder(query_prefix="Represent this sentence: ")
    assert e.prepare_document("aspirin") == "aspirin"
    assert e.prepare_query("aspirin") == "Represent this sentence: aspirin"


def test_real_model_specs_declare_prefix_convention():
    """M3 dropped the prefix; v1.5 requires it. It is per-model, so it lives on the
    provider rather than in caller code."""
    assert em.MODELS["bge-small"]["query_prefix"]
    assert em.MODELS["bge-base"]["query_prefix"]
    assert em.MODELS["bge-m3"]["query_prefix"] == ""
    assert em.MODELS["bge-m3"]["window"] > em.MODELS["bge-small"]["window"]


def test_disabling_prefix_changes_provider_name():
    """An ablation row must be distinguishable in the table, or two rows collide."""
    a = em.LocalEmbedder("bge-small")
    b = em.LocalEmbedder("bge-small", use_query_prefix=False)
    c = em.LocalEmbedder("bge-small", normalize_text=False)
    assert len({a.name, b.name, c.name}) == 3
    assert b.query_prefix == ""


# -- cache keying -----------------------------------------------------------


def test_cache_key_changes_with_every_vector_affecting_input():
    base = em.LocalEmbedder("bge-small")
    keys = {
        em.cache_key("some text", base),
        em.cache_key("other text", base),  # different text
        em.cache_key("some text", em.LocalEmbedder("bge-base")),  # different model
        em.cache_key("some text", em.LocalEmbedder("bge-small", normalize_text=False)),
        em.cache_key("some text", em.LocalEmbedder("bge-small", use_query_prefix=False)),
    }
    assert len(keys) == 5


def test_cache_key_distinguishes_text_differing_only_by_heading():
    """The defect this signature exists to prevent.

    Keying on the chunk's `content_hash` hashed `text` only, so two chunks with
    identical bodies under different headings shared a key and the second silently
    received the first's vector — two such groups exist in the real `structural`
    chunks. Keying on the prepared string cannot collide.
    """
    e = em.LocalEmbedder("bge-small")
    body = "The applicant shall submit the report within fifteen days."
    a = em.cache_key(f"4.2.5.3 Data Handling\n\n{body}", e)
    b = em.cache_key(f"4.4.6.3 Data Handling\n\n{body}", e)
    assert a != b


def test_cache_key_is_case_sensitive():
    """`content_hash` case-folds, which is invisible under bge-small's uncased
    tokenizer but wrong under bge-m3's cased one — the plan's destination model.
    35 groups in the real semantic chunks differ only by case or whitespace."""
    e = em.LocalEmbedder("bge-m3")
    assert em.cache_key("Aspirin dose", e) != em.cache_key("aspirin dose", e)
    assert em.cache_key("a  b", e) != em.cache_key("a b", e)


def test_cache_key_is_stable_across_instances():
    a = em.cache_key("text", em.LocalEmbedder("bge-small"))
    b = em.cache_key("text", em.LocalEmbedder("bge-small"))
    assert a == b


def test_cache_key_does_not_change_with_irrelevant_settings():
    """Batch size and device affect speed, not vectors. Including either would
    needlessly cold-start the cache."""
    a = em.cache_key("t", em.LocalEmbedder("bge-small", batch_size=8, device="cpu"))
    b = em.cache_key("t", em.LocalEmbedder("bge-small", batch_size=256, device="mps"))
    assert a == b


def test_cache_key_fields_cannot_be_confused_with_each_other():
    """NUL-delimited, so no combination of field values can be rearranged into
    another combination's key."""
    e = em.LocalEmbedder("bge-small")
    assert em.cache_key("a\x00b", e) != em.cache_key("a", e)


# -- VectorCache ------------------------------------------------------------


def test_cache_encodes_only_misses(tmp_path):
    cache = em.VectorCache.open(tmp_path / "c", dim=4)
    seen: list[list[str]] = []

    def encode(texts):
        seen.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    cache.get_or_encode(["a", "b"], ["ta", "tb"], encode)
    cache.get_or_encode(["a", "b", "c"], ["ta", "tb", "tc"], encode)
    assert seen == [["ta", "tb"], ["tc"]]
    assert cache.misses == 3


def test_cache_deduplicates_repeated_keys_within_one_call():
    """Regulatory boilerplate produces exact duplicate chunks; encoding them once
    is a real saving, and the returned matrix must still align to the input."""
    cache = em.VectorCache(path=None, dim=4)  # type: ignore[arg-type]
    calls: list[int] = []

    def encode(texts):
        calls.append(len(texts))
        return np.stack([np.full(4, float(i), dtype=np.float32) for i in range(len(texts))])

    out = cache.get_or_encode(["x", "y", "x"], ["tx", "ty", "tx"], encode)
    assert calls == [2]
    assert out.shape == (3, 4)
    assert np.array_equal(out[0], out[2])


def test_cache_round_trips_through_disk(tmp_path):
    def encode(texts):
        return np.stack([np.full(4, float(len(t)), dtype=np.float32) for t in texts])

    first = em.VectorCache.open(tmp_path / "c", dim=4)
    want = first.get_or_encode(["k1", "k2"], ["a", "bb"], encode)
    first.flush()

    reopened = em.VectorCache.open(tmp_path / "c", dim=4)
    assert len(reopened) == 2
    got = reopened.get_or_encode(["k1", "k2"], ["a", "bb"], _no_encode)
    assert np.array_equal(want, got)
    assert reopened.misses == 0


def _no_encode(texts):
    raise AssertionError(f"should not re-encode: {texts}")


def test_cache_preserves_key_to_vector_mapping_across_reload(tmp_path):
    """Row order on disk must not scramble which vector belongs to which key — a
    silent version of this would mis-rank everything while raising nothing."""

    def encode(texts):
        return np.stack([np.full(3, float(t), dtype=np.float32) for t in texts])

    c = em.VectorCache.open(tmp_path / "c", dim=3)
    keys = [f"k{i}" for i in range(10)]
    c.get_or_encode(keys, [str(i) for i in range(10)], encode)
    c.flush()

    reopened = em.VectorCache.open(tmp_path / "c", dim=3)
    shuffled = ["k7", "k2", "k9", "k0"]
    got = reopened.get_or_encode(shuffled, ["7", "2", "9", "0"], _no_encode)
    assert [v[0] for v in got] == [7.0, 2.0, 9.0, 0.0]


def test_cache_discards_file_written_by_a_different_model(tmp_path):
    """A dim mismatch means another model wrote this file. Mixing vector spaces
    would produce confident nonsense rather than an error, so refuse to load."""
    c = em.VectorCache.open(tmp_path / "c", dim=4)
    c.get_or_encode(["a"], ["ta"], lambda t: np.ones((len(t), 4), dtype=np.float32))
    c.flush()

    other = em.VectorCache.open(tmp_path / "c", dim=8)
    assert len(other) == 0


def test_cache_tolerates_corrupt_files(tmp_path):
    (tmp_path / "c.npy").write_bytes(b"not a numpy file")
    (tmp_path / "c.keys.json").write_text("{{{ not json")
    cache = em.VectorCache.open(tmp_path / "c", dim=4)
    assert len(cache) == 0  # cold cache, not a crash


def test_cache_detects_key_vector_length_disagreement(tmp_path):
    c = em.VectorCache.open(tmp_path / "c", dim=4)
    c.get_or_encode(["a", "b"], ["ta", "tb"], lambda t: np.ones((len(t), 4), dtype=np.float32))
    c.flush()
    (tmp_path / "c.keys.json").write_text(json.dumps(["a", "b", "c"]))
    assert len(em.VectorCache.open(tmp_path / "c", dim=4)) == 0


def test_cache_rejects_mismatched_inputs():
    cache = em.VectorCache(path=None, dim=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length mismatch"):
        cache.get_or_encode(["a", "b"], ["only-one"], lambda t: np.ones((len(t), 4)))


def test_cache_rejects_encoder_returning_wrong_count():
    """A provider that drops a row would otherwise misalign every later key."""
    cache = em.VectorCache(path=None, dim=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vectors for"):
        cache.get_or_encode(["a", "b"], ["ta", "tb"], lambda t: np.ones((1, 4), dtype=np.float32))


def test_flush_is_a_noop_when_nothing_changed(tmp_path):
    cache = em.VectorCache.open(tmp_path / "c", dim=4)
    cache.flush()
    assert not (tmp_path / "c.npy").exists()


# -- DenseRetriever ---------------------------------------------------------


def test_dense_ranks_by_cosine_and_returns_contiguous_ranks():
    chunks = [chunk(f"c{i}", f"body {i}") for i in range(6)]
    r = DenseRetriever(chunks, FakeEmbedder())
    hits = r.search("body 3", k=4)
    assert len(hits) == 4
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_dense_finds_the_exact_text_it_indexed():
    """A query identical to a document's prepared text must rank it first — the
    weakest possible sanity check, and the one that catches a prefix or
    normalization mismatch between the two encode paths."""
    e = FakeEmbedder(query_prefix="")
    chunks = [chunk(f"c{i}", f"unique body {i}") for i in range(5)]
    r = DenseRetriever(chunks, e, heading_mode="source")
    assert r.search("unique body 2", k=1)[0].chunk_id == "c2"


def test_dense_respects_heading_mode():
    c = chunk("c1", "Heading\n\nThe body text.", heading="Heading")
    prepend = DenseRetriever([c], FakeEmbedder(), heading_mode="prepend")
    strip = DenseRetriever([c], FakeEmbedder(), heading_mode="strip")
    assert not np.array_equal(prepend.matrix, strip.matrix)


def test_dense_rejects_unknown_heading_mode():
    with pytest.raises(ValueError, match="heading_mode"):
        DenseRetriever([chunk("c1", "x")], FakeEmbedder(), heading_mode="nope")


def test_dense_k_is_clamped_to_corpus_size():
    r = DenseRetriever([chunk("c1", "a"), chunk("c2", "b")], FakeEmbedder())
    assert len(r.search("q", k=99)) == 2


def test_dense_handles_empty_corpus():
    assert DenseRetriever([], FakeEmbedder()).search("q", k=5) == []


def test_dense_encodes_identical_text_once_and_shares_the_vector():
    """Genuinely identical embedding input is encoded once; both chunks get it."""
    e = FakeEmbedder()
    cache = em.VectorCache(path=None, dim=e.dim)  # type: ignore[arg-type]
    chunks = [chunk("c1", "same text"), chunk("c2", "same text")]
    r = DenseRetriever(chunks, e, cache=cache)
    assert cache.misses == 1
    assert cache.deduped == 1
    assert r.matrix.shape == (2, e.dim)
    assert np.array_equal(r.matrix[0], r.matrix[1])


def test_dense_does_not_share_a_vector_across_different_headings():
    """The regression for the cache-key defect, at the retriever level: identical
    bodies under different headings must get different vectors under `prepend`."""
    e = FakeEmbedder()
    cache = em.VectorCache(path=None, dim=e.dim)  # type: ignore[arg-type]
    chunks = [
        chunk("c1", "identical body text here", heading="Section A", chash="H"),
        chunk("c2", "identical body text here", heading="Section B", chash="H"),
    ]
    r = DenseRetriever(chunks, e, cache=cache, heading_mode="prepend")
    assert cache.misses == 2, "same content_hash must not collapse different headings"
    assert not np.array_equal(r.matrix[0], r.matrix[1])


def test_dense_keys_the_string_it_encodes():
    """Key and encoded text must come from one preparation, not two."""
    e = FakeEmbedder()
    cache = em.VectorCache(path=None, dim=e.dim)  # type: ignore[arg-type]
    c = chunk("c1", "Heading ....... 4\nBody text follows here.", heading=None)
    DenseRetriever([c], e, cache=cache, heading_mode="source")
    encoded = e.calls[0][0]
    assert "......." not in encoded, "encoder received un-normalized text"
    assert em.cache_key(encoded, e) in cache._keys


def test_cache_counters_account_for_every_requested_key():
    """hits + misses + deduped == len(keys). An earlier version mixed occurrence
    and distinct denominators, so the printed line was short by the duplicates."""
    cache = em.VectorCache(path=None, dim=4)  # type: ignore[arg-type]
    enc = lambda t: np.ones((len(t), 4), dtype=np.float32)  # noqa: E731
    cache.get_or_encode(["a", "a", "b"], ["ta", "ta", "tb"], enc)
    assert (cache.hits, cache.misses, cache.deduped) == (0, 2, 1)
    cache.get_or_encode(["a", "b", "c"], ["ta", "tb", "tc"], enc)
    assert cache.hits + cache.misses + cache.deduped == 6


def test_cache_rejects_encoder_returning_wrong_dimension():
    """A wrong-width vector accepted here writes a file that every later run
    discards — the corpus is re-embedded forever, silently."""
    cache = em.VectorCache(path=None, dim=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected"):
        cache.get_or_encode(["a"], ["ta"], lambda t: np.ones((len(t), 8), dtype=np.float32))


def test_count_truncation_does_not_load_the_model(real_tokenizer_only):
    """Truncation accounting needs tokenization, not a model. Reading
    `st.tokenizer` loaded the full SentenceTransformer on every eval run."""
    e = real_tokenizer_only
    e.count_truncation(["hello world"])
    assert e._st is None, "model was instantiated for a tokenization-only operation"


def test_dense_truncation_counted_on_normalized_text(real_tokenizer):
    """Counting raw text would overstate the loss and give the normalizer no credit."""
    e = real_tokenizer
    dotted = "Heading " + "." * 3000 + " 42"
    raw = e.count_truncation([dotted])
    normalized = e.count_truncation([e.prepare_document(dotted)])
    assert raw.n_truncated == 1
    assert normalized.n_truncated == 0
    assert normalized.tokens_total < raw.tokens_total


def test_truncation_stats_percentages_and_empty_case():
    s = em.TruncationStats(n_texts=200, n_truncated=50, tokens_total=1000, tokens_dropped=250)
    assert s.pct_truncated == pytest.approx(25.0)
    assert s.pct_tokens_dropped == pytest.approx(25.0)
    empty = em.TruncationStats()
    assert empty.pct_truncated == 0.0 and empty.pct_tokens_dropped == 0.0


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        em.LocalEmbedder("gpt-embeddings-9000")
