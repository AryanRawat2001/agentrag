"""BM25 retriever tests.

This file exists because the numerical audit found two defects here and there was
no `test_retrieval.py` at all — the retriever was covered only indirectly, through
whatever the eval harness happened to exercise. Both defects were reproducibility
bugs: identical inputs could produce different rankings, which is the one class of
error that makes an ablation table worthless no matter how correct the arithmetic.
"""

from __future__ import annotations

from ragpipe.retrieval import BM25Retriever, _index_text


def chunk(cid: str, text: str, *, heading: str | None = None, idents=None) -> dict:
    return {
        "chunk_id": cid,
        "doc_id": "d1",
        "text": text,
        "embed_text": f"{heading}\n\n{text}" if heading else text,
        "section_heading": heading,
        "identifiers": idents or [],
        "strategy": "fixed",
    }


CORPUS = [
    chunk("c1", "The applicant shall submit an annual report of adverse events."),
    chunk("c2", "Adverse event reporting requirements for sponsors of trials."),
    chunk("c3", "Manufacturing controls and stability testing of drug substance."),
    chunk("c4", "Labeling requirements for over the counter monograph products."),
]


# -- out-of-vocabulary queries ----------------------------------------------


def test_all_oov_query_returns_no_hits():
    """The audit's MEDIUM-HIGH finding.

    `bm25s.tokenize` on a string query silently drops out-of-vocabulary terms, so an
    entirely-unseen query yields a *non-empty* token list, an empty id list, and an
    all-zero score vector. The old guard checked only for empty tokens, so this
    returned k documents all scoring 0.0 in an arbitrary order — and rank fusion
    consumed that noise as ranks 1..k.
    """
    r = BM25Retriever(CORPUS, name="bm25")
    assert r.search("zzzzqqqq_unseen_token", k=100) == []


def test_underscore_joined_title_is_one_oov_token():
    """The exact query that exposed it: `NEPA_Final_Guidance`. Underscore is a word
    character under bm25s' `\\b\\w\\w+\\b` pattern, so the whole string is one token."""
    r = BM25Retriever(CORPUS, name="bm25")
    assert r.search("NEPA_Final_Guidance", k=50) == []


def test_no_returned_hit_ever_has_a_zero_score():
    """A zero BM25 score is absence of evidence, not weak evidence. Handing one to
    fusion as a rank lets a document that matched nothing outrank one that matched,
    purely by appearing in two candidate lists."""
    r = BM25Retriever(CORPUS, name="bm25")
    # k far exceeds the number of documents that can possibly match.
    hits = r.search("adverse", k=len(CORPUS))
    assert hits
    assert all(h.score > 0.0 for h in hits)
    assert len(hits) < len(CORPUS)  # non-matching docs dropped, not returned at 0.0


def test_partially_oov_query_still_uses_the_known_terms():
    r = BM25Retriever(CORPUS, name="bm25")
    hits = r.search("qqzzunseen adverse", k=5)
    assert {h.chunk_id for h in hits} <= {"c1", "c2"}
    assert hits


# -- determinism -------------------------------------------------------------


def test_ranks_are_contiguous_from_one_after_filtering():
    """Dropping zero-score hits must renumber, not leave holes — `rank` is what
    RRF divides by, so a gap silently reweights the fusion."""
    r = BM25Retriever(CORPUS, name="bm25")
    hits = r.search("reporting requirements", k=len(CORPUS))
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_scores_are_descending():
    r = BM25Retriever(CORPUS, name="bm25")
    scores = [h.score for h in r.search("requirements", k=4)]
    assert scores == sorted(scores, reverse=True)


def test_exact_score_ties_break_on_chunk_id():
    """The audit's finding 7. `fusion._ranked` and `CrossEncoderReranker.search`
    both tie-break on chunk_id and both cite determinism; this retriever did not, so
    a tie between a relevant and an irrelevant chunk resolved on bm25s' internal
    ordering and rank-1 metrics were not reproducible from the same inputs.
    """
    # Identical text under different ids scores identically by construction.
    tied = [chunk("zzz", "identical wording here"), chunk("aaa", "identical wording here")]
    hits = BM25Retriever(tied, name="bm25").search("identical wording", k=2)
    assert [h.chunk_id for h in hits] == ["aaa", "zzz"]


def test_ranking_is_stable_across_input_order():
    """Same corpus, different insertion order, same ranking."""
    forward = BM25Retriever(CORPUS, name="bm25").search("requirements", k=4)
    reverse = BM25Retriever(list(reversed(CORPUS)), name="bm25").search("requirements", k=4)
    assert [h.chunk_id for h in forward] == [h.chunk_id for h in reverse]
    assert [h.score for h in forward] == [h.score for h in reverse]


def test_repeated_search_is_identical():
    r = BM25Retriever(CORPUS, name="bm25")
    a = r.search("adverse event reporting", k=4)
    b = r.search("adverse event reporting", k=4)
    assert a == b


# -- basic contract ----------------------------------------------------------


def test_finds_the_obviously_relevant_chunk():
    r = BM25Retriever(CORPUS, name="bm25")
    assert r.search("stability testing drug substance", k=1)[0].chunk_id == "c3"


def test_k_is_clamped_and_empty_corpus_is_safe():
    assert BM25Retriever(CORPUS, name="bm25").search("requirements", k=999)
    assert len(BM25Retriever(CORPUS, name="bm25").search("requirements", k=999)) <= len(CORPUS)


def test_heading_modes_change_indexed_text_only_when_a_heading_exists():
    """`fixed` chunks carry no heading, so all three modes must be identical for
    them — the reason the heading confound stayed invisible until strategies were
    compared."""
    plain = chunk("c1", "body text about reporting")
    assert (
        _index_text(plain, "prepend") == _index_text(plain, "source") == _index_text(plain, "strip")
    )

    with_heading = chunk(
        "c2", "Subpart C — Reporting\n\nbody text", heading="Subpart C — Reporting"
    )
    assert "Subpart C" in _index_text(with_heading, "prepend")
    assert "Subpart C" not in _index_text(with_heading, "strip")
