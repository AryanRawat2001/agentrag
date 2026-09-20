"""Tests for tier-2 citation judging.

Weighted toward the calibration machinery, because that is what decides whether any
faithfulness number this module produces means anything. An uncalibrated judge emits
confident verdicts either way, and the failure is invisible in the output.
"""

from __future__ import annotations

from ragpipe import judge


def _pair(pid, doc, answer, quote):
    from ragpipe.golden import STATUS_ACCEPTED, GoldenPair

    return GoldenPair(
        pair_id=pid,
        question="q" * 50,
        answer=answer,
        evidence=({"chunk_id": f"{doc}::c1", "quote": quote},),
        doc_id=doc,
        shape="requirement",
        status=STATUS_ACCEPTED,
    )


def _pairs(n=8):
    return [
        _pair(f"d{i}::gold::000", f"d{i}", f"answer number {i} asserting something", f"quote {i}")
        for i in range(n)
    ]


class TestControlSet:
    def test_builds_both_kinds_of_control(self):
        items = judge.build_control_set(_pairs(), n_positive=4, n_negative=4)
        assert sum(1 for i in items if i.control == "positive") == 4
        assert sum(1 for i in items if i.control == "negative") == 4

    def test_negatives_pair_a_claim_with_another_documents_evidence(self):
        """The whole point: the evidence provably cannot support the claim."""
        items = judge.build_control_set(_pairs(), 0, 4)
        for item in items:
            claim_n = item.claim.split()[2]
            quote_n = item.quote.split()[1]
            assert claim_n != quote_n, f"{item.item_id} paired a claim with its own evidence"

    def test_negatives_never_cross_within_one_document(self):
        pairs = [
            _pair(f"same::gold::00{i}", "same-doc", f"answer {i}", f"quote {i}") for i in range(4)
        ]
        assert judge.build_control_set(pairs, 0, 4) == []

    def test_control_set_is_deterministic(self):
        a = judge.build_control_set(_pairs(), 3, 3)
        b = judge.build_control_set(_pairs(), 3, 3)
        assert [i.item_id for i in a] == [i.item_id for i in b]

    def test_rejected_pairs_are_never_used_as_controls(self):
        from ragpipe.golden import STATUS_REJECTED, GoldenPair

        bad = GoldenPair(
            pair_id="x::gold::000",
            question="q",
            answer="a",
            evidence=({"chunk_id": "c", "quote": "q"},),
            doc_id="x",
            shape="requirement",
            status=STATUS_REJECTED,
        )
        assert judge.build_control_set([bad], 5, 5) == []

    def test_item_ids_encode_the_label(self):
        items = judge.build_control_set(_pairs(), 2, 2)
        assert all(i.item_id.startswith("pos::") for i in items if i.control == "positive")
        assert all(i.item_id.startswith("neg::") for i in items if i.control == "negative")


class TestAgreementScoring:
    def _items(self):
        # Four per side: `MIN_CONTROLS_PER_SIDE` rejects a calibration built on fewer,
        # because a rate off one or two items is not a calibration.
        return [judge.JudgeItem(f"pos::{i}", "c", "q", control="positive") for i in range(4)] + [
            judge.JudgeItem(f"neg::{i}", "c", "q", control="negative") for i in range(4)
        ]

    def test_a_perfect_judge_scores_one_on_both(self):
        j = {i.item_id: judge.Judgment(i.item_id, judge.SUPPORTED) for i in self._items()[:4]}
        j.update(
            {i.item_id: judge.Judgment(i.item_id, judge.UNSUPPORTED) for i in self._items()[4:]}
        )
        a = judge.score_agreement(self._items(), j)
        assert a.positive_rate == 1.0 and a.negative_rate == 1.0 and a.usable

    def test_a_judge_that_says_supported_to_everything_is_caught(self):
        """The failure a positives-only calibration cannot see: perfect on positives,
        zero on negatives."""
        j = {i.item_id: judge.Judgment(i.item_id, judge.SUPPORTED) for i in self._items()}
        a = judge.score_agreement(self._items(), j)
        assert a.positive_rate == 1.0
        assert a.negative_rate == 0.0
        assert not a.usable

    def test_a_judge_that_rejects_everything_is_also_caught(self):
        """Gating on negatives alone let this through: it rejects every constructed
        negative perfectly. A low floor on positives closes the hole."""
        j = {i.item_id: judge.Judgment(i.item_id, judge.UNSUPPORTED) for i in self._items()}
        a = judge.score_agreement(self._items(), j)
        assert a.negative_rate == 1.0 and a.positive_rate == 0.0 and not a.usable

    def test_partial_counts_as_rejecting_a_negative(self):
        j = {
            i.item_id: judge.Judgment(i.item_id, judge.PARTIAL)
            for i in self._items()
            if i.control == "negative"
        }
        a = judge.score_agreement([i for i in self._items() if i.control == "negative"], j)
        assert a.negative_rate == 1.0

    def test_partial_does_not_count_as_supporting_a_positive(self):
        j = {"pos::0": judge.Judgment("pos::0", judge.PARTIAL)}
        a = judge.score_agreement([self._items()[0]], j)
        assert a.positive_rate == 0.0

    def test_missing_judgments_are_counted_not_ignored(self):
        a = judge.score_agreement(self._items(), {})
        assert a.unjudged == 8
        assert a.positive_rate is None and a.negative_rate is None
        assert not a.usable

    def test_non_control_items_are_excluded_from_agreement(self):
        items = self._items() + [judge.JudgeItem("real::1", "c", "q")]
        j = {"real::1": judge.Judgment("real::1", judge.SUPPORTED)}
        a = judge.score_agreement(items, j)
        assert a.n_positive == 0 and a.n_negative == 0 and a.unjudged == 8

    def test_usable_is_gated_on_negatives_alone(self):
        """The asymmetry is the point. Negatives are constructed and cannot be supported
        under any reading, so failing them means a broken judge. Positives are only
        *presumed* supported, so a low rate may mean defective reference data — the first
        real calibration scored 0.750 there with a judge that was right every time."""
        for pos, neg, ok in (
            (1.0, 1.0, True),
            (0.75, 1.0, True),  # the real case: good judge, weak golden pairs
            (0.2, 1.0, False),  # rejects nearly everything -- broken the other way
            (1.0, 0.5, False),  # rubber stamp
            (1.0, 0.0, False),
        ):
            a = judge.Agreement(
                n_positive=20,
                n_negative=20,
                positive_supported=int(pos * 20),
                negative_rejected=int(neg * 20),
            )
            assert a.usable is ok, (pos, neg)

    def test_low_positive_rate_points_at_the_golden_set(self):
        a = judge.Agreement(
            n_positive=16, n_negative=16, positive_supported=12, negative_rejected=16
        )
        d = a.as_dict()
        assert d["usable"] is True
        assert d["inspect_golden_set"] is True

    def test_healthy_calibration_raises_no_golden_set_flag(self):
        a = judge.Agreement(
            n_positive=16, n_negative=16, positive_supported=16, negative_rejected=16
        )
        assert a.as_dict()["inspect_golden_set"] is False


class TestPromptAndParsing:
    def test_prompt_echoes_every_item_id(self):
        items = [judge.JudgeItem(f"i{n}", f"claim {n}", f"quote {n}") for n in range(3)]
        p = judge.build_prompt(items)
        for i in items:
            assert i.item_id in p and i.claim in p and i.quote in p

    def test_prompt_states_the_expected_count(self):
        assert "exactly 3 judgments" in judge.build_prompt(
            [judge.JudgeItem(f"i{n}", "c", "q") for n in range(3)]
        )

    def test_judgments_are_keyed_not_positional(self):
        """A short response must not shift verdicts onto the wrong claims."""
        payload = {
            "judgments": [
                {"item_id": "i2", "verdict": "unsupported", "reason": "r"},
                {"item_id": "i0", "verdict": "supported", "reason": "r"},
            ]
        }
        out = judge.parse_judgments(payload)
        assert out["i0"].verdict == judge.SUPPORTED
        assert out["i2"].verdict == judge.UNSUPPORTED
        assert "i1" not in out

    def test_invalid_verdicts_are_dropped(self):
        payload = {"judgments": [{"item_id": "i0", "verdict": "probably", "reason": "r"}]}
        assert judge.parse_judgments(payload) == {}

    def test_malformed_entries_do_not_abort_the_batch(self):
        payload = {
            "judgments": ["garbage", None, {"item_id": "i0", "verdict": "supported", "reason": "r"}]
        }
        assert set(judge.parse_judgments(payload)) == {"i0"}

    def test_missing_key_yields_nothing(self):
        assert judge.parse_judgments({}) == {}

    def test_prompt_names_the_hl7_failure_mode(self):
        """The concrete case tier 1 could not catch, carried into the instructions."""
        assert "HL7" in judge.JUDGE_SYSTEM_PROMPT

    def test_prompt_tells_the_judge_to_read_through_pdf_artifacts(self):
        assert "line numbers" in judge.JUDGE_SYSTEM_PROMPT


class TestBatching:
    def test_batches_respect_the_size(self):
        items = [judge.JudgeItem(f"i{n}", "c", "q") for n in range(20)]
        b = judge.batches(items, 8)
        assert [len(x) for x in b] == [8, 8, 4]

    def test_every_item_appears_exactly_once(self):
        items = [judge.JudgeItem(f"i{n}", "c", "q") for n in range(20)]
        flat = [i.item_id for b in judge.batches(items, 7) for i in b]
        assert sorted(flat) == sorted(i.item_id for i in items)

    def test_controls_are_spread_not_grouped(self):
        """One bad response must not wipe out the calibration."""
        pairs = [_pair(f"d{i}::gold::000", f"d{i}", f"answer {i}", f"quote {i}") for i in range(16)]
        items = judge.build_control_set(pairs, 8, 8)
        first = judge.batches(items, 8)[0]
        kinds = {i.control for i in first}
        assert kinds == {"positive", "negative"}, "controls grouped into one batch"

    def test_judge_default_differs_from_the_generation_default(self):
        """A model must not grade its own homework."""
        from ragpipe.generation import DEFAULT_GENERATOR

        assert judge.DEFAULT_JUDGE != DEFAULT_GENERATOR


class TestCalibrationIsAuditable:
    """An aggregate that cannot be recomputed has to be trusted.

    The first calibration needed re-scoring after the `usable` rule was corrected. The
    per-item reasons are what distinguished "broken judge" from "golden answers outrun
    their evidence" — a distinction no aggregate number carries.
    """

    def test_per_item_verdicts_and_reasons_are_persisted(self, tmp_path):
        import json

        items = [
            judge.JudgeItem("pos::1", "claim one", "quote one", control="positive"),
            judge.JudgeItem("neg::1", "claim two", "quote two", control="negative"),
        ]
        judgments = {
            "pos::1": judge.Judgment("pos::1", judge.SUPPORTED, "states it directly"),
            "neg::1": judge.Judgment("neg::1", judge.UNSUPPORTED, "about something else"),
        }
        a = judge.score_agreement(items, judgments)
        out = judge.write_agreement(a, tmp_path / "cal.json", items, judgments)
        data = json.loads(out.read_text())
        assert len(data["items"]) == 2
        by_id = {i["item_id"]: i for i in data["items"]}
        assert by_id["pos::1"]["reason"] == "states it directly"
        assert by_id["neg::1"]["verdict"] == judge.UNSUPPORTED
        assert by_id["neg::1"]["control"] == "negative"

    def test_agreement_is_recomputable_from_the_file(self, tmp_path):
        """The property that matters: the rates can be re-derived from disk without
        re-spending quota. An earlier version of this test ended in `if False else True`
        and so could not fail — the same defect this suite keeps catching elsewhere.
        """
        import json

        items = [judge.JudgeItem(f"pos::{i}", "c", "q", control="positive") for i in range(4)]
        items += [judge.JudgeItem(f"neg::{i}", "c", "q", control="negative") for i in range(4)]
        judgments = {i.item_id: judge.Judgment(i.item_id, judge.SUPPORTED) for i in items[:3]}
        judgments.update(
            {i.item_id: judge.Judgment(i.item_id, judge.UNSUPPORTED) for i in items[4:]}
        )
        original = judge.score_agreement(items, judgments)
        out = judge.write_agreement(original, tmp_path / "c.json", items, judgments)

        data = json.loads(out.read_text())
        rebuilt_items = [
            judge.JudgeItem(d["item_id"], d["claim"], d["quote"], control=d["control"])
            for d in data["items"]
        ]
        rebuilt_judgments = {
            d["item_id"]: judge.Judgment(d["item_id"], d["verdict"], d["reason"])
            for d in data["items"]
            if d["verdict"]
        }
        rescored = judge.score_agreement(rebuilt_items, rebuilt_judgments)
        assert rescored.as_dict() == original.as_dict()
        # Non-trivial: one positive went unjudged, and unjudged items are excluded from
        # the denominator rather than counted as failures.
        assert original.n_positive == 3
        assert original.positive_rate == 1.0
        assert original.unjudged == 1

    def test_unjudged_items_persist_with_a_null_verdict(self, tmp_path):
        import json

        items = [judge.JudgeItem("pos::1", "c", "q", control="positive")]
        out = judge.write_agreement(
            judge.score_agreement(items, {}), tmp_path / "c.json", items, {}
        )
        assert json.loads(out.read_text())["items"][0]["verdict"] is None


class TestMostlyMissingJudgments:
    """Rates divide by judged items only, so a mostly-failed batch could otherwise
    report a perfect rate from a couple of survivors."""

    def test_a_mostly_unjudged_calibration_is_not_usable(self):
        a = judge.Agreement(
            n_positive=1, n_negative=1, positive_supported=1, negative_rejected=1, unjudged=30
        )
        assert a.positive_rate == 1.0 and a.negative_rate == 1.0
        assert not a.usable, "a perfect rate off 2 of 32 items is not a calibration"

    def test_a_few_missing_judgments_are_tolerated(self):
        a = judge.Agreement(
            n_positive=15, n_negative=15, positive_supported=15, negative_rejected=15, unjudged=2
        )
        assert a.usable

    def test_the_real_calibration_still_passes(self):
        """16 positive + 16 negative, none missing, 12 supported."""
        a = judge.Agreement(
            n_positive=16, n_negative=16, positive_supported=12, negative_rejected=16
        )
        assert a.usable and a.as_dict()["inspect_golden_set"]


class TestRealItemsArePersistedToo:
    """Controls alone are not enough.

    `_run_tier2` originally wrote only the control items, so the per-answer verdicts that
    produce the faithfulness figure were absent from disk and the number could not be
    re-derived. It surfaced when regenerating the report yielded 0 of 8 supported.
    """

    def test_non_control_items_round_trip(self, tmp_path):
        import json

        items = [
            judge.JudgeItem("pos::1", "c", "q", control="positive"),
            judge.JudgeItem("neg::1", "c", "q", control="negative"),
            judge.JudgeItem("query-7", "the answer", "the evidence"),
        ]
        judgments = {
            "pos::1": judge.Judgment("pos::1", judge.SUPPORTED),
            "neg::1": judge.Judgment("neg::1", judge.UNSUPPORTED),
            "query-7": judge.Judgment("query-7", judge.SUPPORTED, "states it"),
        }
        out = judge.write_agreement(
            judge.score_agreement(items, judgments), tmp_path / "c.json", items, judgments
        )
        data = json.loads(out.read_text())
        real = [i for i in data["items"] if i["control"] is None]
        assert len(real) == 1
        assert real[0]["item_id"] == "query-7"
        assert real[0]["verdict"] == judge.SUPPORTED
        assert real[0]["reason"] == "states it"

    def test_controls_and_real_items_are_distinguishable_on_disk(self, tmp_path):
        import json

        items = [
            judge.JudgeItem("pos::1", "c", "q", control="positive"),
            judge.JudgeItem("q1", "c", "q"),
        ]
        out = judge.write_agreement(
            judge.score_agreement(items, {}), tmp_path / "c.json", items, {}
        )
        controls = [i for i in json.loads(out.read_text())["items"] if i["control"]]
        assert len(controls) == 1, "agreement must be recomputable from controls alone"


class TestControlsSpreadAcrossDocuments:
    """The Phase 6 audit gate's most consequential finding.

    `build_control_set` sorted by `pair_id` and took a prefix. Golden pair_ids are
    prefixed by doc_id, so every control came from ONE document — and it was
    `ctgov-NCT00567567-Prot_SAP_000`, the document Phase 6a singled out for spurious
    intra-word spaces and a 10/10 -> 5/10 drafting regression. The 0.750 positive rate
    was therefore a measurement of the corpus's worst document, extrapolated to "golden
    pairs" generally.

    `golden.sample_for_review` already spread across documents for exactly this reason.
    The lesson did not travel between modules, which is why this test names it.
    """

    def _pairs(self, docs=6, per_doc=4):
        from ragpipe.golden import STATUS_ACCEPTED, GoldenPair

        out = []
        for d in range(docs):
            for i in range(per_doc):
                out.append(
                    GoldenPair(
                        pair_id=f"doc{d}::gold::{i:03d}",
                        question="q" * 50,
                        answer=f"answer {d}-{i} asserting something specific",
                        evidence=({"chunk_id": f"doc{d}::c1", "quote": f"quote {d}-{i}"},),
                        doc_id=f"doc{d}",
                        shape="requirement",
                        status=STATUS_ACCEPTED,
                    )
                )
        return out

    def _docs_of(self, items, kind):
        return {i.source_id.split("|")[0].split("::")[0] for i in items if i.control == kind}

    def test_positives_span_many_documents(self):
        items = judge.build_control_set(self._pairs(), 6, 0)
        assert len(self._docs_of(items, "positive")) == 6, "a prefix would give 2"

    def test_a_prefix_would_have_failed_this(self):
        """Guard against a false pass: the fixture must be able to concentrate."""
        pairs = sorted(self._pairs(), key=lambda p: p.pair_id)
        prefix_docs = {p.doc_id for p in pairs[:6]}
        assert len(prefix_docs) < 6, "fixture cannot demonstrate the defect"

    def test_negatives_also_span_documents(self):
        items = judge.build_control_set(self._pairs(), 0, 6)
        assert len(self._docs_of(items, "negative")) >= 4

    def test_still_deterministic(self):
        a = judge.build_control_set(self._pairs(), 5, 5)
        b = judge.build_control_set(self._pairs(), 5, 5)
        assert [i.item_id for i in a] == [i.item_id for i in b]

    def test_uneven_document_sizes_do_not_break_the_round_robin(self):
        from ragpipe.golden import STATUS_ACCEPTED, GoldenPair

        pairs = self._pairs(docs=3, per_doc=2)
        pairs += [
            GoldenPair(
                pair_id=f"big::gold::{i:03d}",
                question="q" * 50,
                answer=f"big answer {i}",
                evidence=({"chunk_id": "big::c1", "quote": f"big quote {i}"},),
                doc_id="big",
                shape="requirement",
                status=STATUS_ACCEPTED,
            )
            for i in range(10)
        ]
        items = judge.build_control_set(pairs, 8, 0)
        assert len(items) == 8
        assert len(self._docs_of(items, "positive")) == 4, "one large document must not dominate"

    def test_every_positive_is_used_at_most_once(self):
        items = judge.build_control_set(self._pairs(), 24, 0)
        ids = [i.item_id for i in items if i.control == "positive"]
        assert len(ids) == len(set(ids))


class TestNegativeConstructionIsOrderingIndependent:
    """The bug the document-spread fix introduced, and its own test caught.

    The old code offset by `len(usable) // 2` and skipped same-document collisions. Under
    round-robin ordering with D documents interleaved, that offset is a multiple of D
    whenever the length is — so every candidate collided and the function returned **zero
    negatives**. Zero negatives makes `negative_rate` None and `usable` False, so it fails
    safe, but silently: the calibration simply stops existing.
    """

    def _pairs(self, docs, per_doc):
        from ragpipe.golden import STATUS_ACCEPTED, GoldenPair

        return [
            GoldenPair(
                pair_id=f"doc{d}::gold::{i:03d}",
                question="q" * 50,
                answer=f"answer {d}-{i}",
                evidence=({"chunk_id": f"doc{d}::c1", "quote": f"quote {d}-{i}"},),
                doc_id=f"doc{d}",
                shape="requirement",
                status=STATUS_ACCEPTED,
            )
            for d in range(docs)
            for i in range(per_doc)
        ]

    def test_uniform_document_sizes_still_produce_negatives(self):
        """6 documents x 4 pairs = 24, len//2 = 12, 12 % 6 == 0 — the exact collision."""
        items = judge.build_control_set(self._pairs(6, 4), 0, 6)
        negs = [i for i in items if i.control == "negative"]
        assert len(negs) == 6, "zero negatives means no calibration at all"

    def test_every_negative_crosses_a_document_boundary(self):
        for docs, per_doc in ((6, 4), (4, 4), (3, 5), (8, 2), (2, 8)):
            items = judge.build_control_set(self._pairs(docs, per_doc), 0, 6)
            for i in (x for x in items if x.control == "negative"):
                claim_doc, ev_doc = (part.split("::")[0] for part in i.source_id.split("|"))
                assert claim_doc != ev_doc, f"{docs}x{per_doc}: {i.source_id}"

    def test_a_single_document_corpus_yields_no_negatives(self):
        """Correct behaviour, not a bug: there is no other document to draw from."""
        assert judge.build_control_set(self._pairs(1, 8), 0, 6) == []

    def test_negatives_are_requested_count_when_available(self):
        for n in (1, 3, 8, 12):
            negs = [
                i
                for i in judge.build_control_set(self._pairs(5, 4), 0, n)
                if i.control == "negative"
            ]
            assert len(negs) == min(n, 20), n
