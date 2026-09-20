"""Tests for the golden-set curation triage.

Every test in the first class pins a bug this module shipped with and that judging real
pairs exposed. They are worth reading as a group, because all four are the same species:
**a check that reported confidently about text it had not actually matched.**

1. `not` without word boundaries matched inside "notified", so "the FDA must be notified"
   was read as a negated obligation. Same class as the `\\bcomment\\b` regex Phase 6 found
   — that one had a boundary it should not have, this one lacked one it needed.
2. The polarity window looked only *forward*, so "are not required" — where the negation
   precedes the modal — read as an assertion of obligation. The check was blind to the
   exact inversion it exists to detect.
3. `max_modal_strength` counted negated modals, so "bookmarks are not required" scored
   mandatory.
4. `check_modal_strength` re-derived the maximum itself instead of calling
   `max_modal_strength`, so fixing (3) changed nothing until the call site was fixed too.

The second theme is **false-positive control**. A triage check that fires on most of the
data is worse than no check, because it teaches a reviewer to skip it. The first negation
check fired on 96 of 159 pairs; these tests hold the rebuilt version to the shape of the
defect instead of the shape of the vocabulary.
"""

from __future__ import annotations

import json

import pytest

from ragpipe import curate


class TestBugsFoundByJudgingRealPairs:
    def test_not_is_word_bounded(self):
        """ "must be notified" is not a negated obligation."""
        assert curate.modal_polarities("must be notified") == {("must", False)}
        assert curate.max_modal_strength("must be notified") == 4
        for word in ("notified", "notice", "annotation", "nothing"):
            assert curate.count_negations(f"shall be {word}") == 0, word

    def test_negation_before_the_modal_is_seen(self):
        """English negates backward as often as forward."""
        assert curate.modal_polarities("are not required") == {("required", True)}
        assert curate.modal_polarities("shall not submit") == {("shall", True)}

    def test_a_negated_strong_modal_asserts_no_obligation(self):
        assert curate.max_modal_strength("bookmarks are not required") == 0
        assert curate.max_modal_strength("bookmarks are required") == 4

    def test_the_check_uses_the_shared_rule_rather_than_its_own(self):
        """The regression that made fixing the rule useless. "not required" paraphrasing
        a source's "not needed" must not read as a strengthening."""
        flags = curate.check_modal_strength(
            "Bookmarks are not required in SPL files",
            "Bookmarks are not needed because the tags may provide this functionality",
        )
        assert flags == []

    def test_a_modal_split_by_pdf_extraction_is_still_found(self):
        """One source reads "the requir ed reporting". Missing it made the source's
        strongest modal fall back to an unrelated clause's "will", and a correct answer
        saying "mandatory" was flagged as inventing an obligation."""
        assert curate.max_modal_strength("the requir ed reporting by hospital staff") == 4
        assert (
            curate.check_modal_strength(
                "mandatory reporting applies", "the requir ed reporting by hospital staff"
            )
            == []
        )

    def test_the_bare_infinitive_of_require_is_ranked(self):
        """8 of 159 sources use it. An incomplete lexicon does not fail gently -- it
        produces confident false positives."""
        assert "require" in curate.MODAL_STRENGTH
        assert curate.max_modal_strength("MR sessions require participants to lie flat") == 4


class TestModalStrength:
    def test_should_to_must_is_flagged_high(self):
        """The defect this module exists for: FDA guidance uses "should" for a
        recommendation, so asserting "must" manufactures a legal obligation."""
        (flag,) = curate.check_modal_strength(
            "The treatment must be discontinued immediately.",
            "the study treatment should be discontinued immediately",
        )
        assert flag.kind == "modal_strengthened"
        assert flag.severity == "high"
        assert "should" in flag.detail and "must" in flag.detail

    def test_weakening_is_deliberately_not_flagged(self):
        """A `modal_weakened` branch fired on 19 of 159 pairs, contributed 45% of all
        flags, and contained **zero** real defects: a cited chunk is a thousand characters
        of regulatory prose and nearly always contains a "must" somewhere, so an answer
        correctly paraphrasing one permissive sentence looks weaker than the chunk. It was
        detecting that regulations contain obligations. Removed, and pinned removed --
        re-adding it would silently restore a 45% noise floor to the worksheet."""
        assert (
            curate.check_modal_strength(
                "The sponsor may submit the report.", "the sponsor shall submit the report"
            )
            == []
        )
        assert "modal_weakened" not in curate.render_worksheet(
            [],
            {
                "n_pairs": 0,
                "n_flagged": 0,
                "flagged_share": None,
                "by_kind": {},
                "by_severity": {"high": 0, "medium": 0, "low": 0},
            },
            {},
        )

    def test_equal_strength_is_not_flagged(self):
        assert curate.check_modal_strength("must do X", "shall do X") == []

    def test_a_source_with_no_modal_is_a_different_finding(self):
        """Imperatives and bare specifications oblige without modals and score 0.
        Conflating "source states no obligation" with "source states a weaker one"
        produced most of this check's false positives."""
        (flag,) = curate.check_modal_strength(
            "Samples must be shipped overnight.", "Ship via Federal Express Priority Overnight."
        )
        assert flag.kind == "modal_added_to_unmodalised_source"
        assert flag.severity == "medium"

    def test_an_answer_with_no_modal_is_never_flagged(self):
        assert curate.check_modal_strength("The deadline is 60 days.", "you shall file it") == []


class TestPolarity:
    def test_an_inversion_is_flagged_when_the_source_is_unambiguous(self):
        (flag,) = curate.check_negation(
            "The sponsor shall submit before approval.",
            "the sponsor shall not submit before approval",
        )
        assert flag.kind == "polarity_inverted"
        assert flag.severity == "high"

    def test_a_source_stating_both_polarities_is_not_flagged(self):
        """Then the answer picked one of two readings the document genuinely supports,
        and only a reader can judge which."""
        assert (
            curate.check_negation(
                "the sponsor shall submit",
                "the sponsor shall not submit before X; the sponsor shall submit after Y",
            )
            == []
        )

    def test_matching_polarity_is_not_flagged(self):
        assert curate.check_negation("shall not submit", "shall not submit the form") == []

    def test_negation_counts_are_not_used_as_the_signal(self):
        """The rebuilt check must not fire merely because a long source contains more
        negations than a short answer -- the asymmetry that fired on 96 of 159 pairs."""
        answer = "The report is due in 60 days."
        source = (
            "This is not the only route. No extension applies. Nothing here is optional. "
            "The report is due in 60 days unless otherwise agreed."
        )
        assert curate.check_negation(answer, source) == []


class TestAttribution:
    def test_only_fires_on_an_obligation_with_a_novel_actor(self):
        (flag,) = curate.check_attribution(
            "The sponsor must notify the committee.", "the agency shall notify the committee"
        )
        assert flag.kind == "actor_not_in_source"

    def test_does_not_fire_without_an_obligation(self):
        assert (
            curate.check_attribution(
                "The sponsor may notify the committee.", "the agency shall notify"
            )
            == []
        )

    def test_does_not_fire_when_the_source_names_the_actor(self):
        assert (
            curate.check_attribution(
                "The sponsor must notify.", "the sponsor shall notify the agency"
            )
            == []
        )


class TestReviewPlumbing:
    @staticmethod
    def _pair(answer, chunk_text, quote="q"):
        return (
            {
                "pair_id": "p1",
                "doc_id": "d1",
                "question": "Q?",
                "answer": answer,
                "evidence": [{"chunk_id": "c1", "quote": quote}],
            },
            {"c1": {"chunk_id": "c1", "doc_id": "d1", "text": chunk_text}},
        )

    def test_a_clean_pair_raises_nothing(self):
        pair, index = self._pair("The sponsor shall submit.", "the sponsor shall submit the form")
        assert curate.review_pair(pair, index).needs_review is False

    def test_an_unresolvable_chunk_is_flagged_not_silently_passed(self):
        """Otherwise a pair whose evidence has vanished looks identical to a clean one."""
        pair, _ = self._pair("anything", "")
        review = curate.review_pair(pair, {})
        assert [f.kind for f in review.flags] == ["no_source"]
        assert review.severity == "high"

    def test_severity_is_the_worst_flag_present(self):
        pair, index = self._pair(
            "The treatment must stop.", "the treatment should stop; the agency may act"
        )
        assert curate.review_pair(pair, index).severity == "high"

    def test_the_relevant_source_sentence_is_chosen_by_content_not_by_modal(self):
        """The worksheet's first version showed the window around the source's strongest
        modal, which is often an unrelated clause: a pair about central venous lines was
        presented beside a sentence about glomerular filtration rate."""
        answer = "All patients must have a double lumen central venous line placed."
        source = (
            "Patients with GFR below 60 will be taken off protocol therapy. "
            "All patients will have a double lumen central venous line placed."
        )
        sentence = curate.relevant_source_sentence(answer, source)
        assert "central venous line" in sentence
        assert "GFR" not in sentence

    def test_summarise_counts_pairs_not_flags_for_severity(self):
        pair, index = self._pair(
            "The treatment must stop and the sponsor must act.",
            "the treatment should stop; the agency may act",
        )
        reviews = [curate.review_pair(pair, index)]
        summary = curate.summarise(reviews)
        assert summary["n_pairs"] == 1
        assert sum(summary["by_severity"].values()) == 1


class TestVerdicts:
    def test_missing_verdict_file_is_empty_not_an_error(self, tmp_path):
        assert curate.load_verdicts(tmp_path / "nope.json") == {}

    def test_verdicts_load_from_either_shape(self, tmp_path):
        wrapped = tmp_path / "w.json"
        wrapped.write_text(json.dumps({"verdicts": {"p1": {"verdict": "keep", "reason": "r"}}}))
        assert curate.load_verdicts(wrapped)["p1"]["verdict"] == "keep"

    def test_non_dict_entries_are_ignored(self):
        """A hand-edited file is the expected input, so a stray string must not crash a
        report build."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.json"
            p.write_text(json.dumps({"verdicts": {"p1": "oops", "p2": {"verdict": "keep"}}}))
            loaded = curate.load_verdicts(p)
            assert set(loaded) == {"p2"}

    def test_the_worksheet_shows_which_pairs_are_still_unjudged(self):
        pair = {
            "pair_id": "p1",
            "doc_id": "d1",
            "question": "Q?",
            "answer": "The treatment must stop.",
            "evidence": [{"chunk_id": "c1", "quote": "q"}],
        }
        index = {"c1": {"chunk_id": "c1", "doc_id": "d1", "text": "the treatment should stop"}}
        reviews = [curate.review_pair(pair, index)]
        summary = curate.summarise(reviews)
        assert "not yet judged" in curate.render_worksheet(reviews, summary, {})
        judged = curate.render_worksheet(
            reviews, summary, {"p1": {"verdict": "reject", "reason": "should -> must"}}
        )
        assert "Verdict: reject" in judged
        assert "not yet judged" not in judged


class TestShippedGoldenSet:
    """The triage run against the real set, checked the way an audit gate would."""

    @staticmethod
    def _report():
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "reports" / "golden_curation.json"
        if not path.exists():
            pytest.skip("needs reports/golden_curation.json — run `ragpipe curate`")
        return json.loads(path.read_text())

    def test_the_flag_rate_stays_in_a_useful_triage_range(self):
        """Not an accuracy claim — a usability one. A check firing on most of the data
        trains a reviewer to skip it, and one firing on none is indistinguishable from
        being switched off."""
        share = self._report()["summary"]["flagged_share"]
        assert 0.02 < share < 0.45, f"flag rate {share:.1%} is not usable as triage"

    def test_every_flagged_pair_carries_the_evidence_for_its_flag(self):
        """A flag without its excerpt is a to-do item, not a finding."""
        for review in self._report()["reviews"]:
            for flag in review["flags"]:
                if flag["kind"] == "no_source":
                    continue
                assert flag["detail"], review["pair_id"]
                assert flag["answer_excerpt"] or flag["source_excerpt"], review["pair_id"]

    def test_high_severity_flags_are_a_readable_number(self):
        by_sev = self._report()["summary"]["by_severity"]
        assert by_sev["high"] <= 40, f"{by_sev['high']} high-severity pairs is not a review pass"


class TestCurationSurvivesRevalidation:
    """The interaction that nearly erased the whole curation pass.

    `ragpipe revalidate` re-runs the five machine checks from each pair's *drafted* state,
    deliberately, so that relaxing a threshold can re-accept a pair rather than only ever
    rejecting more. But the pairs a human rejects are exactly the ones that **pass** every
    machine check -- a misstated obligation level is lexically invisible, which is why it
    needed a person. So a plain re-validation reported `152 -> 159` and `--write` would
    have silently undone the curation and called it a clean run.

    Two commands, each correct alone, whose interaction destroys work.
    """

    def test_a_curation_reject_is_not_machine_detectable(self):
        """The premise. If a machine check could see these, curation would be unnecessary
        and this interaction would be harmless."""
        flags = curate.check_modal_strength(
            "The treatment must be discontinued immediately.",
            "the study treatment should be discontinued immediately",
        )
        # The curation detector sees it...
        assert flags and flags[0].kind == "modal_strengthened"
        # ...but it is a *triage* signal, not one of the five validation checks, and the
        # answer is fully supported lexically, which is what validation measures.
        from ragpipe.golden import answer_evidence_coverage

        coverage = answer_evidence_coverage(
            "The treatment must be discontinued immediately.",
            ["the study treatment should be discontinued immediately"],
        )
        assert coverage >= 0.45, (
            "if coverage rejected this, validation would already catch obligation errors"
        )

    def test_revalidate_holds_curation_rejects(self):
        """Asserted against the shipped artifacts rather than a fixture, because the bug
        was in how the two real files interact."""
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        if not (root / "corpus" / "curation_verdicts.json").exists():
            pytest.skip("needs corpus/curation_verdicts.json")
        proc = subprocess.run(
            ["uv", "run", "ragpipe", "revalidate"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, proc.stderr[-1500:]
        assert "held rejected by human curation verdicts" in proc.stderr
        # And the accepted count must not go up, which is the symptom the bug produced.
        import re

        match = re.search(r"\[revalidate\] (\d+) -> (\d+) accepted", proc.stderr)
        assert match, proc.stderr[-800:]
        before, after = int(match.group(1)), int(match.group(2))
        assert after <= before, f"revalidation re-accepted {after - before} curated-out pairs"

    def test_every_recorded_reject_is_actually_rejected_on_disk(self):
        import json as _json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        verdicts_path = root / "corpus" / "curation_verdicts.json"
        golden_path = root / "corpus" / "golden.jsonl"
        if not (verdicts_path.exists() and golden_path.exists()):
            pytest.skip("needs the curated golden set")
        verdicts = curate.load_verdicts(verdicts_path)
        rejected = {p for p, v in verdicts.items() if v.get("verdict") == "reject"}
        status = {
            _json.loads(line)["pair_id"]: _json.loads(line)["status"]
            for line in golden_path.open()
            if line.strip()
        }
        still_accepted = [p for p in rejected if status.get(p) == "accepted"]
        assert not still_accepted, f"judged reject but still accepted: {still_accepted}"

    def test_every_verdict_carries_a_reason(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "corpus" / "curation_verdicts.json"
        if not path.exists():
            pytest.skip("needs corpus/curation_verdicts.json")
        for pair_id, verdict in curate.load_verdicts(path).items():
            assert verdict.get("verdict") in curate.VERDICT_VALUES, pair_id
            assert len(verdict.get("reason", "")) > 40, (
                f"{pair_id}: a verdict without a substantive reason is not auditable"
            )


class TestDefectsFoundByThePhase8Gates:
    """Each of these was a live defect the two Phase 8 gates found. All are the project's
    recurring class: a claim that outran the code, or a check in the wrong place."""

    def test_no_source_fires_when_several_chunks_fail_to_resolve(self):
        """`" ".join` over two unresolvable ids yields `" "` — truthy — so the guard was
        skipped and three checks ran against whitespace, producing confident false
        positives on a pair whose evidence had vanished."""
        for n_evidence in (1, 2, 3):
            pair = {
                "pair_id": "p1",
                "doc_id": "d1",
                "question": "Q?",
                "answer": "The sponsor must submit the report within 30 days.",
                "evidence": [{"chunk_id": f"gone{i}", "quote": "q"} for i in range(n_evidence)],
            }
            review = curate.review_pair(pair, {})
            kinds = [f.kind for f in review.flags]
            assert kinds == ["no_source"], (n_evidence, kinds)

    def test_the_recommendation_frame_is_ranked(self):
        """`recommend`/`recommendations` were absent, so "The following recommendations
        apply" scored 0 and routed to the *lenient* branch — which cost a wrong keep on
        `fda-78268::gold::094`. Same defect class as the missing bare `require`."""
        for frame in (
            "The following recommendations apply to amendments, updates, and supplements",
            "We recommend you provide the following",
            "What special recommendations apply to the submission",
        ):
            assert curate.max_modal_strength(frame) == 2, frame
        (flag,) = curate.check_modal_strength(
            "Each filename must begin with the date.",
            "The following recommendations apply. Begin each filename with the date.",
        )
        assert flag.kind == "modal_strengthened"
        assert flag.severity == "high"

    def test_every_live_flag_kind_has_an_explanation_in_the_worksheet(self):
        """The `meanings` table listed a deleted check and omitted three live ones, so the
        shipped worksheet's "what it means" column was blank for most rows."""
        pair = {
            "pair_id": "p1",
            "doc_id": "d1",
            "question": "Q?",
            "answer": "The sponsor must submit.",
            "evidence": [{"chunk_id": "c1", "quote": "q"}],
        }
        index = {"c1": {"chunk_id": "c1", "doc_id": "d1", "text": "the sponsor should submit"}}
        reviews = [curate.review_pair(pair, index)]
        summary = curate.summarise(reviews)
        # Every kind the checks can actually emit must be explained.
        emitted = {
            "modal_strengthened",
            "modal_added_to_unmodalised_source",
            "modal_strengthened_in_context",
            "polarity_inverted",
            "actor_not_in_source",
            "unsupported_number",
            "no_source",
        }
        forced = {**summary, "by_kind": dict.fromkeys(emitted, 1)}
        worksheet = curate.render_worksheet(reviews, forced, {})
        for kind in emitted:
            row = next((ln for ln in worksheet.splitlines() if f"`{kind}`" in ln and "|" in ln), "")
            assert row, f"{kind} has no row in the meanings table"
            assert row.rstrip().rstrip("|").rsplit("|", 1)[-1].strip(), (
                f"{kind}'s explanation cell is blank"
            )

    def test_the_deleted_check_is_not_advertised(self):
        """`negation_asymmetry` is the removed count-based check and can never fire."""
        pair = {
            "pair_id": "p1",
            "doc_id": "d1",
            "question": "Q?",
            "answer": "The sponsor must submit.",
            "evidence": [{"chunk_id": "c1", "quote": "q"}],
        }
        index = {"c1": {"chunk_id": "c1", "doc_id": "d1", "text": "the sponsor should submit"}}
        reviews = [curate.review_pair(pair, index)]
        out = curate.render_worksheet(reviews, curate.summarise(reviews), {})
        assert "negation_asymmetry" not in out

    def test_the_weakened_removal_guard_is_not_vacuous(self):
        """Half of the original guard asserted a string was absent from a worksheet whose
        `by_kind` fixture was empty — so the meanings table rendered no rows at all and the
        assertion could not fail. Re-checked with the key actually present."""
        pair = {
            "pair_id": "p1",
            "doc_id": "d1",
            "question": "Q?",
            "answer": "The sponsor may submit.",
            "evidence": [{"chunk_id": "c1", "quote": "q"}],
        }
        index = {"c1": {"chunk_id": "c1", "doc_id": "d1", "text": "the sponsor shall submit"}}
        reviews = [curate.review_pair(pair, index)]
        summary = curate.summarise(reviews)
        forced = {**summary, "by_kind": {"modal_weakened": 1, "modal_strengthened": 1}}
        out = curate.render_worksheet(reviews, forced, {})
        # The check emits nothing for weakening...
        assert curate.check_modal_strength("may submit", "shall submit") == []
        # ...and even when a caller forces the key in, no explanation exists for it,
        # because the kind is gone rather than merely unused.
        row = next((ln for ln in out.splitlines() if "modal_weakened" in ln), "")
        if row:
            cell = row.rstrip().rstrip("|").rsplit("|", 1)[-1].strip()
            assert not cell, "modal_weakened still has an explanation, implying it is live"

    def test_an_orphan_verdict_is_reported_not_silently_dropped(self):
        """A one-character typo in a hand-written verdict file deletes a human judgement.

        Tested against the module's logic rather than by driving the CLI. The first version
        wrote a synthetic verdict into the **real** `corpus/curation_verdicts.json`, ran
        `ragpipe curate` in the **real** repo, and restored only the verdicts file — so it
        silently rewrote `reports/golden_curation.{md,json}` with a fixture in them, and
        `reports/failure_modes.json` was then generated *from* that, committing
        `fda-99999::gold::999` into two published artifacts. A test that mutates committed
        state is worse than a missing test: the suite stayed green while the reports rotted.
        """
        known = {"fda-1::gold::001", "fda-1::gold::002"}
        verdicts = {
            "fda-1::gold::001": {"verdict": "reject", "reason": "x" * 60},
            "fda-1::gold::16": {"verdict": "reject", "reason": "typo for ::016"},
            "fda-99999::gold::999": {"verdict": "keep", "reason": "does not exist"},
        }
        orphans = sorted(set(verdicts) - known)
        assert orphans == ["fda-1::gold::16", "fda-99999::gold::999"]

    def test_the_cli_warns_about_orphans_without_touching_committed_reports(self):
        """The end-to-end half, with every file it can write snapshotted and restored.

        `cmd_curate` writes `reports/golden_curation.{md,json}` unconditionally, so any
        test that invokes it has to put those back — including on failure.
        """
        import json as _json
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        touched = [
            root / "corpus" / "curation_verdicts.json",
            root / "reports" / "golden_curation.json",
            root / "reports" / "golden_curation.md",
        ]
        if not all(p.exists() for p in touched):
            pytest.skip("needs the curated golden set and its report")

        snapshot = {p: p.read_bytes() for p in touched}
        try:
            payload = _json.loads(snapshot[touched[0]].decode())
            payload["verdicts"]["fda-99999::gold::999"] = {
                "verdict": "reject",
                "reason": "synthetic orphan, asserted and reverted by this test",
            }
            touched[0].write_text(_json.dumps(payload, indent=2) + "\n")
            proc = subprocess.run(
                ["uv", "run", "ragpipe", "curate"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=600,
            )
            assert proc.returncode == 0, proc.stderr[-800:]
            assert "absent from" in proc.stderr
            assert "fda-99999::gold::999" in proc.stderr
        finally:
            for path, blob in snapshot.items():
                path.write_bytes(blob)

        # And the restore has to be real, not assumed.
        for path, blob in snapshot.items():
            assert path.read_bytes() == blob, f"{path.name} was not restored"
        assert "fda-99999" not in (root / "reports" / "golden_curation.json").read_text()

    def test_no_verdict_in_the_shipped_file_is_an_orphan(self):
        import json as _json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        verdicts_path = root / "corpus" / "curation_verdicts.json"
        golden_path = root / "corpus" / "golden.jsonl"
        if not (verdicts_path.exists() and golden_path.exists()):
            pytest.skip("needs the curated golden set")
        known = {_json.loads(line)["pair_id"] for line in golden_path.open() if line.strip()}
        orphans = sorted(set(curate.load_verdicts(verdicts_path)) - known)
        assert not orphans, f"verdicts naming pairs that do not exist: {orphans}"


class TestDeterminism:
    """The worksheet must be byte-identical across processes.

    `modal_polarities` returns a **set**, so a `max(..., key=...)` tie-break followed set
    iteration order — which depends on `PYTHONHASHSEED`. Four seeds produced four
    different `reports/golden_curation.md` files, so the artifact was not reproducible by
    the command that writes it, and a flag's `source_excerpt` sometimes pointed at FDA
    page-header boilerplate rather than the clause under dispute. Adding four `recommend*`
    spellings at strength 2 turned ties from rare into routine.
    """

    def test_the_tie_break_is_stable_and_prefers_the_specific_spelling(self):
        tied = {"should", "recommended", "recommendations", "recommend"}
        assert curate._pick(tied, 2) == "recommendations"
        # Same answer every call, and independent of insertion order.
        assert curate._pick(set(reversed(sorted(tied))), 2) == "recommendations"

    def test_an_empty_candidate_set_yields_empty_not_an_exception(self):
        assert curate._pick(set(), 4) == ""
        assert curate._pick({"may"}, 4) == ""

    def test_both_checks_use_the_same_rule(self):
        """Two reports of the same data must not disagree about which modal to name."""
        answer = "The treatment must be discontinued."
        source = "the treatment should be discontinued; we recommend it be discontinued"
        (strength_flag,) = curate.check_modal_strength(answer, source)
        context = curate.check_modal_in_context(answer, source)
        named = [f for f in (strength_flag, *context)]
        quoted = {
            f.detail.split('source says "')[-1].split('"')[0]
            for f in named
            if 'source says "' in f.detail
        }
        # Whatever they name, they must agree on it.
        assert len(quoted) <= 1, f"the two checks disagree on the source modal: {quoted}"

    def test_repeated_renders_of_the_same_pair_are_identical(self):
        pair = {
            "pair_id": "p1",
            "doc_id": "d1",
            "question": "Q?",
            "answer": "The treatment must be discontinued immediately.",
            "evidence": [{"chunk_id": "c1", "quote": "q"}],
        }
        index = {
            "c1": {
                "chunk_id": "c1",
                "doc_id": "d1",
                "text": "we recommend the treatment should be discontinued; recommendations apply",
            }
        }
        renders = set()
        for _ in range(8):
            reviews = [curate.review_pair(pair, index)]
            renders.add(curate.render_worksheet(reviews, curate.summarise(reviews), {}))
        assert len(renders) == 1, f"{len(renders)} distinct renders of one pair"

    def test_the_shipped_worksheet_is_reproducible_across_hash_seeds(self):
        """End-to-end, because the unit tests above cannot see a set that leaks into the
        renderer by some other path."""
        import hashlib
        import os
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        report = root / "reports" / "golden_curation.md"
        if not report.exists():
            pytest.skip("needs reports/golden_curation.md")
        snapshot = report.read_bytes()
        json_path = root / "reports" / "golden_curation.json"
        json_snapshot = json_path.read_bytes() if json_path.exists() else None
        digests = set()
        try:
            for seed in ("0", "1", "7"):
                env = {**os.environ, "PYTHONHASHSEED": seed}
                proc = subprocess.run(
                    ["uv", "run", "ragpipe", "curate"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=env,
                )
                assert proc.returncode == 0, proc.stderr[-500:]
                digests.add(hashlib.md5(report.read_bytes()).hexdigest())
        finally:
            report.write_bytes(snapshot)
            if json_snapshot is not None:
                json_path.write_bytes(json_snapshot)
        assert len(digests) == 1, f"{len(digests)} distinct worksheets across hash seeds"


class TestJudgedCountIsNotARatioAboveOne:
    """`Judged: 25 of 24` -- the numerator counted every verdict on file, the denominator
    only what the *current* detectors flag. Once the detector fixes retired a flag on an
    already-judged pair, the header printed a progress fraction greater than 1."""

    @staticmethod
    def _summary(**over):
        base = {
            "n_pairs": 159,
            "n_flagged": 24,
            "by_kind": {},
            "by_severity": {"high": 20, "medium": 4, "low": 0},
            "verdicts_recorded": 25,
            "flagged_unjudged": 0,
            "verdicts_retired_by_detector_fixes": 1,
        }
        base.update(over)
        return base

    def test_the_numerator_never_exceeds_the_flag_count(self):
        out = curate.render_worksheet([], self._summary())
        assert "**Judged: 24 of 24 flagged pairs.**" in out
        assert "25 of 24" not in out

    def test_unread_pairs_reduce_the_numerator(self):
        out = curate.render_worksheet([], self._summary(flagged_unjudged=5))
        assert "**Judged: 19 of 24 flagged pairs.**" in out
        assert "5 still unread" in out

    def test_a_retired_verdict_is_reported_separately_and_explained(self):
        out = curate.render_worksheet([], self._summary())
        assert "1 further verdict is kept" in out
        assert "no longer" in out and "revalidate" in out

    def test_retired_verdicts_pluralise(self):
        out = curate.render_worksheet([], self._summary(verdicts_retired_by_detector_fixes=3))
        assert "3 further verdicts are kept" in out

    def test_no_retired_verdicts_means_no_paragraph(self):
        out = curate.render_worksheet([], self._summary(verdicts_retired_by_detector_fixes=0))
        assert "further verdict" not in out

    def test_the_judged_line_is_absent_when_no_verdicts_are_loaded(self):
        summary = self._summary()
        del summary["verdicts_recorded"]
        assert "Judged:" not in curate.render_worksheet([], summary)


class TestVerdictFileValidation:
    """`corpus/curation_verdicts.json` is hand-edited and is the one artifact here that
    cannot be regenerated. It was loaded with no validation at all, and the failure mode is
    silent in the dangerous direction: only `verdict == "reject"` is acted on, so a
    misspelled rejection leaves the pair in the accepted set while the worksheet still
    counts it as judged."""

    @staticmethod
    def _write(tmp_path, payload):
        import json

        p = tmp_path / "verdicts.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    def test_a_missing_file_is_empty_not_an_error(self, tmp_path):
        assert curate.load_verdicts(tmp_path / "absent.json") == {}

    def test_a_valid_file_loads(self, tmp_path):
        path = self._write(tmp_path, {"verdicts": {"a::gold::1": {"verdict": "reject"}}})
        assert curate.load_verdicts(path)["a::gold::1"]["verdict"] == "reject"

    def test_a_misspelled_rejection_is_refused_rather_than_ignored(self, tmp_path):
        path = self._write(tmp_path, {"verdicts": {"a::gold::1": {"verdict": "rejct"}}})
        with pytest.raises(ValueError, match="unrecognised value"):
            curate.load_verdicts(path)

    def test_a_capitalised_verdict_is_refused(self, tmp_path):
        """`"Reject"` is not `"reject"`, and the comparison downstream is exact."""
        path = self._write(tmp_path, {"verdicts": {"a::gold::1": {"verdict": "Reject"}}})
        with pytest.raises(ValueError, match="unrecognised value"):
            curate.load_verdicts(path)

    def test_a_missing_verdict_key_is_refused(self, tmp_path):
        path = self._write(tmp_path, {"verdicts": {"a::gold::1": {"reason": "why"}}})
        with pytest.raises(ValueError, match="unrecognised value"):
            curate.load_verdicts(path)

    def test_the_error_names_the_offending_pair(self, tmp_path):
        path = self._write(tmp_path, {"verdicts": {"a::gold::1": {"verdict": "nope"}}})
        with pytest.raises(ValueError, match="a::gold::1"):
            curate.load_verdicts(path)

    def test_every_allowed_value_is_accepted(self, tmp_path):
        payload = {
            "verdicts": {
                f"a::gold::{i}": {"verdict": v} for i, v in enumerate(curate.VERDICT_VALUES)
            }
        }
        assert len(curate.load_verdicts(self._write(tmp_path, payload))) == len(
            curate.VERDICT_VALUES
        )

    def test_an_amendment_disagreeing_with_the_verdict_is_refused(self, tmp_path):
        """The audit trail claiming a rejection the pipeline never received. Nothing reads
        `amendments`, which is precisely why it can drift from the verdict it describes."""
        path = self._write(
            tmp_path,
            {
                "verdicts": {"a::gold::1": {"verdict": "keep"}},
                "amendments": [
                    {
                        "pair_id": "a::gold::1",
                        "from": "keep",
                        "to": "reject",
                        "why": "w",
                        "date": "2026-01-01",
                    }
                ],
            },
        )
        with pytest.raises(ValueError, match="audit trail and the verdict disagree"):
            curate.load_verdicts(path)

    def test_a_consistent_amendment_loads(self, tmp_path):
        path = self._write(
            tmp_path,
            {
                "verdicts": {"a::gold::1": {"verdict": "reject"}},
                "amendments": [
                    {
                        "pair_id": "a::gold::1",
                        "from": "keep",
                        "to": "reject",
                        "why": "w",
                        "date": "2026-01-01",
                    }
                ],
            },
        )
        assert curate.load_verdicts(path)["a::gold::1"]["verdict"] == "reject"

    def test_an_amendment_missing_its_reasoning_is_refused(self, tmp_path):
        """An amendment without `why` is the one field that makes the trail worth keeping."""
        path = self._write(
            tmp_path,
            {
                "verdicts": {"a::gold::1": {"verdict": "reject"}},
                "amendments": [{"pair_id": "a::gold::1", "from": "keep", "to": "reject"}],
            },
        )
        with pytest.raises(ValueError, match="missing"):
            curate.load_verdicts(path)

    def test_an_amendment_naming_an_absent_pair_is_refused(self, tmp_path):
        path = self._write(
            tmp_path,
            {
                "verdicts": {},
                "amendments": [
                    {
                        "pair_id": "ghost::gold::9",
                        "from": "keep",
                        "to": "reject",
                        "why": "w",
                        "date": "2026-01-01",
                    }
                ],
            },
        )
        with pytest.raises(ValueError, match="disagree"):
            curate.load_verdicts(path)

    def test_the_committed_file_passes_its_own_validation(self):
        """The point of all of the above."""
        from pathlib import Path

        real = Path(__file__).resolve().parents[1] / "corpus" / "curation_verdicts.json"
        if not real.exists():
            pytest.skip("needs corpus/curation_verdicts.json")
        assert len(curate.load_verdicts(real)) == 25
