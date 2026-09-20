"""Tests for golden-set construction.

New in the post-quarantine regeneration. There was no test module for `evalset`
before this, which is part of why the pool filter was missing for five phases.
"""

from __future__ import annotations

from ragpipe import evalset


class TestQuarantinedDocsAreExcludedFromThePool:
    """The substance of the post-quarantine regeneration.

    Ground truth being chunking-independent does not make it quarantine-independent.
    A span in a document no strategy indexes resolves to no chunk under every
    retriever, so the query is unscorable by construction and silently shrinks its
    slice. Before the filter, 7 of 180 answerable queries pointed at the three
    unreadable documents — and regenerating the set made it *worse* (2 → 7), because
    quarantining more documents means more queries aimed at them.
    """

    def _docs(self):
        """Three documents, one of which the chunker would refuse to index."""
        return [
            {
                "doc_id": "good-1",
                "text_quality": "ok",
                "single_char_token_share": 0.03,
                "control_char_share": 0.0,
                "sections": [
                    {
                        "heading": "1. Scope of This Guidance",
                        "idx": 0,
                        "level": 0,
                        "page": 1,
                        "start": 0,
                        "end": 200,
                    }
                ],
                "title": "A Readable Guidance",
                "n_chars": 400,
                "source": "fda_guidance",
                "metadata": {},
                "distinct_identifiers_detailed": {},
            },
            {
                "doc_id": "glyph-noise",
                "text_quality": "broken_encoding",
                "single_char_token_share": 0.07,
                "control_char_share": 0.13,
                "sections": [
                    {
                        "heading": "2. Glyph Noise Section",
                        "idx": 0,
                        "level": 0,
                        "page": 1,
                        "start": 0,
                        "end": 200,
                    }
                ],
                "title": "Unreadable Glyph Codes",
                "n_chars": 400,
                "source": "fda_guidance",
                "metadata": {},
                "distinct_identifiers_detailed": {},
            },
            {
                "doc_id": "spaced",
                "text_quality": "character_spaced",
                "single_char_token_share": 0.72,
                "control_char_share": 0.0,
                "sections": [
                    {
                        "heading": "3. Character Spaced Section",
                        "idx": 0,
                        "level": 0,
                        "page": 1,
                        "start": 0,
                        "end": 200,
                    }
                ],
                "title": "Character Spaced",
                "n_chars": 400,
                "source": "fda_guidance",
                "metadata": {},
                "distinct_identifiers_detailed": {},
            },
        ]

    def test_the_chunker_predicate_rejects_exactly_the_unreadable_two(self):
        """Pins the coupling: the eval set and the index must not disagree about which
        corpus exists, so both consult the same predicate."""
        from ragpipe.cli import _is_quarantined

        verdicts = {d["doc_id"]: _is_quarantined(d) for d in self._docs()}
        assert verdicts == {"good-1": False, "glyph-noise": True, "spaced": True}

    def test_no_query_targets_a_quarantined_document(self):
        from ragpipe.cli import _is_quarantined

        docs = self._docs()
        pool = [d for d in docs if not _is_quarantined(d)]
        excluded = {d["doc_id"] for d in docs if _is_quarantined(d)}
        queries = evalset.build_all(pool, seed=1, per_slice=3)
        for q in queries:
            assert not (set(q.relevant_doc_ids or []) & excluded), q.query_id

    def test_filtering_is_what_prevents_it_not_luck(self):
        """Guard against a false pass: unfiltered, the same seed *does* produce
        queries against the excluded documents."""
        from ragpipe.cli import _is_quarantined

        docs = self._docs()
        excluded = {d["doc_id"] for d in docs if _is_quarantined(d)}
        unfiltered = evalset.build_all(docs, seed=1, per_slice=3)
        leaked = [q for q in unfiltered if set(q.relevant_doc_ids or []) & excluded]
        assert leaked, "fixture must be able to leak, or the test above proves nothing"
