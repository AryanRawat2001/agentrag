"""Tests for the failure analysis.

The module's whole claim is that **no figure in it was written from memory**, so the
central test is a mutation guard: render from the real payload and from one whose every
numeric leaf is a sentinel, and any number identical in both outputs is by construction
not derived from the data.

That claim needed enforcing rather than asserting. Writing this report is what caught the
corpus line-number exposure — quoted in three live source files as "7,229 of 13,706
chunks (52.7%)" — having been computed before the post-quarantine regeneration, against a
corpus that now holds 13,423 chunks. The conclusion survived; the arithmetic had been
wrong for three phases because the figure lived only in prose.

The second theme is **not overstating what the artifacts support**. `wrong_chunk` and
`line_number_ambiguous` are both zero in the shipped generation run and both have been
seen live, so the report must present them as anecdotes. A failure analysis that promotes
an `n=1` observation to a rate is doing the thing this project exists to avoid.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import pytest

from ragpipe import failures

ROOT = Path(__file__).resolve().parents[1]


def _payload(**over):
    payload = {
        "identifier_collapse": [
            {
                "chunking": "fixed",
                "sparse": 0.9553,
                "dense": 0.0128,
                "fused": 0.1628,
                "ratio": 74.6,
            },
            {
                "chunking": "structural",
                "sparse": 0.9508,
                "dense": 0.0222,
                "fused": 0.2019,
                "ratio": 42.8,
            },
        ],
        "refusal": {
            "per_slice": {
                "exact_identifier": {"n": 4, "refused": 3},
                "section_lookup": {"n": 4, "refused": 1},
                "title_lookup": {"n": 4, "refused": 0},
                "unanswerable": {"n": 4, "refused": 4},
            },
            "answerable_n": 12,
            "answerable_refused": 4,
            "honest_n": 8,
            "honest_refused": 1,
            "refused_queries": [
                {
                    "query_id": "ident-0000",
                    "slice": "exact_identifier",
                    "query": "What are the requirements of 21 CFR 1.980?",
                },
                {
                    "query_id": "section-0003",
                    "slice": "section_lookup",
                    "query": "3.7.4 Risk mitigation strategy",
                },
            ],
        },
        "obligations": {
            "n_judged": 25,
            "n_rejected": 2,
            "rejected": {
                "ctgov-NCT02942264-Prot_SAP_000::gold::016": {
                    "verdict": "reject",
                    "reason": 'source: "should be discontinued". Answer: "MUST be discontinued".',
                },
                "fda-75426::gold::087": {
                    "verdict": "reject",
                    "reason": 'source: "should be read aloud". Answer: "MUST be read aloud".',
                },
            },
        },
        "line_numbers": {
            "n_chunks": 13423,
            "n_affected": 7141,
            "share_affected": 0.532,
            "digits_before": 392163,
            "digits_after": 305932,
            "digits_removed": 86231,
        },
        "citation_buckets": {
            "exact": 18,
            "normalized": 4,
            "line_number_ambiguous": 0,
            "wrong_chunk": 0,
            "unverified": 0,
            "too_short": 0,
        },
        "faithfulness": {
            "n_claimed": 22,
            "n_located": 22,
            "answers_with_citations": 8,
            "answers_fully_supported": 7,
            "n_partial": 1,
            "answer_level_rate": 0.875,
        },
        "judge": {"positive_rate": 0.875, "negative_rate": 1.0},
        "generation_n": 16,
        "curation": {"n_flagged": 25, "n_pairs": 159},
        "missing_artifacts": [],
    }
    payload.update(over)
    return payload


#: Containers whose *keys* are data rather than schema: pair ids and slice names. They
#: carry digits (`::gold::016`, `fda-75426`), so leaving them fixed made the guard flag
#: its own fixture; perturbing every key instead renamed the payload schema the renderer
#: indexes by. Only these two get their keys replaced.
#: Only `obligations.rejected`. `refusal.per_slice` is keyed by *slice name*, which
#: the renderer looks up by name (`per_slice.get("unanswerable")`); perturbing those
#: keys made the lookup miss and print its `0` fallback, which then registered as a
#: hardcoded zero rather than as the graceful degradation it is.
_ID_KEYED = (("obligations", "rejected"),)


#: Markdown constructs whose digits are structure, not measurement: the five `## N.`
#: section numbers and the `N.` ordered-list markers. Stripped before extraction rather
#: than allow-listed, because allow-listing "1".."5" also excuses a hardcoded count of 1
#: to 5 -- and the Phase 8 code-review gate found exactly that: `n_judged`,
#: `answerable_refused`, `honest_refused`, `n_rejected` and `n_partial` could all be
#: hardcoded without the guard noticing.
_STRUCTURAL = (
    # The trailing `\s` is not cosmetic. Without it, `^#+\s*\d+\.` eats the integer part
    # *and the decimal point* of any figure at the start of a heading: "## 74.6x ..."
    # became "6x ...", and the residue "6" is in `ALLOWED` — so a hardcoded 74.6x in a
    # heading shipped green. That is the single most staleness-prone figure in this
    # project, and the strip added to close a hole opened a bigger one.
    re.compile(r"^#+\s*\d+\.\s", re.M),  # "## 3. The over-refusal rate..."
    re.compile(r"^\s*\d+\.\s", re.M),  # "1. Dense retrieval collapses..."
)


def _numbers(text):
    for pattern in _STRUCTURAL:
        text = pattern.sub("", text)
    text = _NAMES.sub("", text)
    return set(re.findall(r"\d[\d,]*(?:\.\d+)?", text))


def _perturb_payload(payload, counter):
    """Perturb every leaf, plus the keys that are data rather than structure."""
    out = _perturb(payload, counter)
    for outer, inner in _ID_KEYED:
        container = out.get(outer, {}).get(inner)
        if isinstance(container, dict):
            out[outer][inner] = {_perturb(key, counter): value for key, value in container.items()}
    return out


def _perturb(obj, counter):
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int):
        # Distinct by a large prime stride, so no two sentinels are close enough for
        # their *ratio* to round to 1.0. Consecutive sentinels made the renderer print
        # "100.0%" for a perturbed count over a perturbed total, colliding with the real
        # payload's 100.0% negative-judge rate and registering as a hardcoded figure.
        return 100_003 + 7_919 * next(counter)
    if isinstance(obj, float):
        # Large and prime-strided, not small and consecutive. Small sentinels (0.1, 0.2,
        # ...) collide with structural digits once the renderer formats them: `ratio` is a
        # float printed with `:.0f`, so 0.3 rendered as "0" and a pair of them rendered as
        # a ratio of "1x" -- both indistinguishable from the table's own small integers,
        # which the guard then reported as leaks.
        return 14_281.0 + 977.31 * next(counter)
    if isinstance(obj, str):
        n = next(counter)
        return "vX" + "".join(chr(ord("a") + int(d)) for d in str(n))
    if isinstance(obj, list):
        return [_perturb(v, counter) for v in obj]
    if isinstance(obj, dict):
        return {k: _perturb(v, counter) for k, v in obj.items()}
    return obj


#: Numbers the prose uses structurally rather than as measurements: the five section
#: numbers, the "20 requests per day" quota that is a fact about the provider, the "0.1"
#: dense fusion weight that names a configuration, and the CFR/NCT identifiers quoted as
#: examples of what an identifier looks like.
ALLOWED = {
    # Facts about the provider and the corpus vocabulary, not measurements of this run.
    "20",  # free-tier requests per day
    "0.1",  # the dense fusion weight that names a working configuration
    "40",
    "0.45",  # min-max weight discussion and the coverage threshold
    # Identifiers quoted as *examples* of what an identifier looks like.
    "21",
    "314.50",
    "02942264",
    "1.980",
    # `recall@10` in the table header, and the "60 days / 30 days" pair quoted to show
    # what a bag-of-words check cannot distinguish.
    "10",
    "60",
    "30",
    # "slowest 5%" and percentile labels.
    "95",
    "50",
    # Names and labels that merely contain digits rather than reporting a figure. "25" is
    # gone: it was admitted for "BM25" and also excused a hardcoded `n_judged` of 25, so
    # the table header is now stripped by `_BM25` instead.
    "6",  # "Phase 6"
    "0000",  # `ident-0000`, the worked example of a mislabelled query
}

#: Names that contain digits. Removed before extraction rather than allow-listed, for the
#: same reason as the section numbers: admitting "25" would excuse a real count of 25, and
#: admitting "1"/"2" for the verifier's two tiers would excuse a real count of 1 or 2 --
#: which is precisely how `n_rejected`, `n_partial` and three refusal counts slipped past
#: the first version of this guard.
_NAMES = re.compile(
    r"\bBM25\b"  # a model name
    r"|\b[Tt]ier [12]\b"  # the verifier's two tiers
    r"|\b(?:finding|mode) [1-5]\b"  # cross-references between the five sections
)


class TestNothingIsWrittenFromMemory:
    def test_no_number_survives_payload_mutation(self):
        real = failures.render_report(_payload())
        mutated = failures.render_report(_perturb_payload(_payload(), itertools.count(1)))
        leaked = (_numbers(real) & _numbers(mutated)) - ALLOWED
        assert not leaked, f"figures not derived from the payload: {sorted(leaked)}"

    def test_a_figure_at_the_start_of_a_heading_is_not_swallowed(self):
        """The `_STRUCTURAL` strip must remove section numbers without eating decimals.

        Break-tested at the three shapes that broke: a ratio, a threshold and a
        percentage, each as the first token of a heading.
        """
        for heading, figure in (
            ("## 74.6x — dense retrieval collapses", "74.6"),
            ("## 0.95 cosine threshold", "0.95"),
            ("### 53.2% of the corpus", "53.2"),
        ):
            original = failures.render_report

            def patched(payload, _h=heading, _o=original):
                return f"{_o(payload)}\n\n{_h}\n"

            failures.render_report = patched
            try:
                real = _numbers(failures.render_report(_payload()))
                mutated = _numbers(
                    failures.render_report(_perturb_payload(_payload(), itertools.count(1)))
                )
                leaked = (real & mutated) - ALLOWED
                assert figure in leaked, (
                    f"{figure} hidden by the heading strip; leaked={sorted(leaked)}"
                )
            finally:
                failures.render_report = original

    def test_section_numbers_are_still_stripped(self):
        """The other half: the strip has to keep working, or every section number leaks."""
        assert _numbers("## 3. The over-refusal rate") == set()
        assert _numbers("1. Dense retrieval collapses") == set()

    def test_the_guard_catches_a_hardcoded_figure(self):
        """Otherwise the test above passes when the prose quotes nothing at all."""
        original = failures.render_report

        def patched(payload, _orig=original):
            return _orig(payload) + "\n\nExposure is 53.2% of 13,423 chunks.\n"

        failures.render_report = patched
        try:
            real = failures.render_report(_payload())
            mutated = failures.render_report(_perturb_payload(_payload(), itertools.count(1)))
            leaked = (_numbers(real) & _numbers(mutated)) - ALLOWED
            assert {"53.2", "13,423"} <= leaked, leaked
        finally:
            failures.render_report = original

    def test_changing_the_corpus_size_changes_the_prose(self):
        base = failures.render_report(_payload())
        moved = _payload()
        moved["line_numbers"] = {**moved["line_numbers"], "n_chunks": 99999}
        assert "99,999" in failures.render_report(moved)
        assert failures.render_report(moved) != base


class TestDoesNotOverstate:
    def test_zero_count_failure_modes_are_presented_as_anecdotes(self):
        """`wrong_chunk` is 0 in the shipped run and has been seen live exactly once.
        The report must say the project has no measured rate for it."""
        report = failures.render_report(_payload())
        assert "Observed once, and therefore not a rate" in report
        assert "no measured rate" in report
        section = report.split("Observed once")[1]
        assert "wrong_chunk" in section

    def test_the_small_sample_is_named_with_its_cause(self):
        report = failures.render_report(_payload())
        assert "n=16" in report
        assert "20 " in report and "per day" in report

    def test_a_missing_artifact_is_declared_not_estimated(self):
        report = failures.render_report(_payload(missing_artifacts=["judge", "verdicts"]))
        assert "Incomplete" in report
        assert "`judge`" in report and "`verdicts`" in report

    def test_the_refusal_finding_separates_the_mislabelled_slice(self):
        """The headline 33% is a property of the ruler, not the model, and the report has
        to say which figure is which."""
        report = failures.render_report(_payload())
        assert "4 of 12" in report and "1 of 8" in report
        assert "measures the dataset" in report

    def test_each_of_the_five_modes_states_a_consequence(self):
        """A failure analysis that only lists failures is a defect register."""
        report = failures.render_report(_payload())
        assert report.count("**Consequence") >= 4


class TestComputations:
    def test_identifier_ratio_is_computed_not_stored(self):
        retrieval = {
            "runs": [
                {
                    "slice_name": "exact_identifier",
                    "chunking": "fixed",
                    "retriever": "bm25",
                    "metrics": {"recall@10": 0.96},
                },
                {
                    "slice_name": "exact_identifier",
                    "chunking": "fixed",
                    "retriever": "dense bge-small",
                    "metrics": {"recall@10": 0.02},
                },
            ]
        }
        (row,) = failures.identifier_collapse(retrieval)
        assert row["ratio"] == pytest.approx(48.0)

    def test_a_dense_recall_of_zero_does_not_produce_infinity(self):
        """An infinity in a report is a division nobody checked."""
        retrieval = {
            "runs": [
                {
                    "slice_name": "exact_identifier",
                    "chunking": "fixed",
                    "retriever": "bm25",
                    "metrics": {"recall@10": 0.96},
                },
                {
                    "slice_name": "exact_identifier",
                    "chunking": "fixed",
                    "retriever": "dense bge-small",
                    "metrics": {"recall@10": 0.0},
                },
            ]
        }
        (row,) = failures.identifier_collapse(retrieval)
        assert row["ratio"] is None
        report = failures.render_report(_payload(identifier_collapse=[row]))
        assert "a factor of —" in report

    def test_refusal_breakdown_excludes_the_mention_labelled_slice(self):
        generation = {
            "outcomes": [
                {"slice_name": "exact_identifier", "refused": True, "query_id": "i0"},
                {"slice_name": "section_lookup", "refused": True, "query_id": "s0"},
                {"slice_name": "section_lookup", "refused": False, "query_id": "s1"},
                {"slice_name": "unanswerable", "refused": True, "query_id": "u0"},
            ]
        }
        out = failures.refusal_breakdown(generation)
        assert out["answerable_n"] == 3 and out["answerable_refused"] == 2
        assert out["honest_n"] == 2 and out["honest_refused"] == 1

    def test_line_number_exposure_counts_chunks_that_lose_digits(self):
        chunks = [
            {"text": "plain sentence with no numbers at all"},
            {"text": "12 the sponsor shall submit\n13 within 60 days"},
        ]
        out = failures.line_number_exposure(chunks)
        assert out["n_chunks"] == 2
        assert out["n_affected"] == 1
        assert out["digits_removed"] > 0

    def test_obligation_errors_counts_only_rejections(self):
        out = failures.obligation_errors(
            {
                "verdicts": {
                    "a": {"verdict": "reject", "reason": "x"},
                    "b": {"verdict": "keep", "reason": "y"},
                    "c": "not a dict",
                }
            }
        )
        assert out["n_judged"] == 3 and out["n_rejected"] == 1

    def test_document_labels_keep_the_identifying_part(self):
        """A fixed-width tail slice produced `4-Prot_SAP_000`, which drops the one part
        that names the document and keeps the part that does not."""
        assert failures._short_doc("ctgov-NCT02942264-Prot_SAP_000::gold::016") == "NCT02942264"
        assert failures._short_doc("fda-75426::gold::085") == "fda-75426"

    def test_clipping_is_visible_and_on_a_word_boundary(self):
        out = failures._clip("one two three four five six seven", 15)
        assert out.endswith("…")
        assert not out.rstrip("…").endswith(" ")


class TestShippedArtifact:
    @staticmethod
    def _payload():
        path = ROOT / "reports" / "failure_modes.json"
        if not path.exists():
            pytest.skip("needs reports/failure_modes.json — run `ragpipe failures`")
        return json.loads(path.read_text())

    def test_the_markdown_matches_the_json(self):
        md = ROOT / "reports" / "failure_modes.md"
        if not md.exists():
            pytest.skip("needs reports/failure_modes.md")
        assert md.read_text() == failures.render_report(self._payload()), (
            "failure_modes.md is not what the renderer produces from failure_modes.json"
        )

    def test_the_line_number_figure_matches_the_corpus_on_disk(self):
        """The regression this whole module was written to prevent: a corpus-derived
        figure quoted in prose that the corpus no longer supports."""
        from ragpipe import CHUNKS_DIR, retrieval

        chunk_path = CHUNKS_DIR / "structural.jsonl"
        if not chunk_path.exists():
            pytest.skip("needs data/chunks/structural.jsonl")
        fresh = failures.line_number_exposure(retrieval.load_chunks(chunk_path))
        stored = self._payload()["line_numbers"]
        assert fresh["n_chunks"] == stored["n_chunks"]
        assert fresh["n_affected"] == stored["n_affected"]

    def test_no_source_file_still_quotes_the_superseded_figure(self):
        """52.7% / 13,706 predates the corpus regeneration.

        Allowed to remain in `docs/progress.md`, which is a dated log of what each phase
        measured, and in a comment that explicitly labels it as the superseded value.
        Not allowed anywhere a reader would take it as current.
        """
        # Live prose only: source docstrings, the README, and the plan all describe the
        # system as it is now. Generated reports are dated snapshots of the run that
        # produced them -- `chunk_size_sweep.md` legitimately carries 13,706 because that
        # sweep ran on the pre-quarantine corpus, and rewriting its table by hand would
        # forge a measurement. Regenerating it is a `make sweep` away and is recorded as
        # carried-forward work instead.
        for path in [
            *(ROOT / "src" / "ragpipe").glob("*.py"),
            ROOT / "README.md",
            ROOT / "docs" / "plan.md",
            ROOT / "reports" / "failure_modes.md",
        ]:
            text = path.read_text(encoding="utf-8")
            for stale in ("13,706", "52.7%"):
                doc = text.splitlines()
                for i, line in enumerate(doc):
                    if stale not in line:
                        continue
                    # A figure inside quotation marks is a citation of a past value, and a
                    # retraction usually sits in the surrounding sentence rather than on
                    # the same physical line -- so the window is the paragraph around it.
                    # Neither form is a live claim.
                    quoted = re.search(rf'"[^"]*{re.escape(stale)}[^"]*"', line) is not None
                    context = " ".join(doc[max(0, i - 3) : i + 9]).lower()
                    labelled = any(
                        marker in context
                        for marker in ("previous", "superseded", "pre-quarantine", "predates")
                    )
                    assert quoted or labelled, (
                        f"{path.name} quotes {stale} as current: {line.strip()[:90]}"
                    )

    def test_every_listed_mode_has_an_artifact_behind_it(self):
        payload = self._payload()
        assert payload["identifier_collapse"], "mode 1 has no data"
        assert payload["obligations"]["n_judged"], "mode 2 has no data"
        assert payload["refusal"]["per_slice"], "mode 3 has no data"
        assert payload["faithfulness"], "mode 4 has no data"
        assert payload["line_numbers"]["n_chunks"], "mode 5 has no data"


class TestNonBindingClaimIsCheckedNotAsserted:
    """The report named the document contributing the most obligation errors and asserted
    that it "marks itself non-binding". That was hardcoded prose riding a `max()`: true of
    today's leader, false for the protocols one rejection behind it, and false for 23 of
    the corpus's 115 FDA documents."""

    FDA_CHUNKS = [
        {"doc_id": "fda-75426", "text": "Contains Nonbinding\nRecommendations\nDraft guidance."},
        {"doc_id": "ctgov-NCT04125745-Prot_SAP_000", "text": "Statistical analysis plan."},
    ]

    @staticmethod
    def _verdicts(pairs):
        return {"verdicts": {p: {"verdict": "reject", "reason": "r"} for p in pairs}}

    def test_the_marking_is_detected_across_a_pdf_line_break(self):
        assert failures.has_nonbinding_marking(self.FDA_CHUNKS, "fda-75426")

    def test_a_protocol_without_the_marking_is_not_credited_with_it(self):
        assert not failures.has_nonbinding_marking(
            self.FDA_CHUNKS, "ctgov-NCT04125745-Prot_SAP_000"
        )

    def test_a_document_absent_from_the_chunks_is_not_credited(self):
        assert not failures.has_nonbinding_marking(self.FDA_CHUNKS, "fda-00000")

    def test_the_clause_appears_for_a_marked_document(self):
        payload = _payload()
        payload["obligations"] = failures.obligation_errors(
            self._verdicts(["fda-75426::gold::001", "fda-75426::gold::002"]), self.FDA_CHUNKS
        )
        out = failures.render_report(payload)
        assert "Contains Nonbinding Recommendations" in out
        assert "fda-75426" in out

    def test_the_clause_vanishes_when_the_leading_document_is_a_protocol(self):
        """The defect this closes: the same sentence, with the same `max()` behind it,
        would have called a ClinicalTrials.gov protocol non-binding."""
        pairs = [
            "ctgov-NCT04125745-Prot_SAP_000::gold::021",
            "ctgov-NCT04125745-Prot_SAP_000::gold::023",
        ]
        payload = _payload()
        payload["obligations"] = failures.obligation_errors(self._verdicts(pairs), self.FDA_CHUNKS)
        out = failures.render_report(payload)
        assert "2 of them come from a single document" in out
        assert "Nonbinding" not in out
        assert "non-binding" not in out

    def test_chunks_omitted_means_the_claim_is_not_made(self):
        obligations = failures.obligation_errors(
            self._verdicts(["fda-75426::gold::001", "fda-75426::gold::002"])
        )
        assert obligations["worst_doc_nonbinding"] is False

    def test_the_named_document_does_not_depend_on_verdict_file_ordering(self):
        """A tie used to resolve by dict insertion order, so the document the report names
        depended on the order a hand-edited JSON file happened to list its keys in."""
        a = ["fda-11111::gold::001", "fda-11111::gold::002"]
        b = ["ctgov-XX-Prot::gold::001", "ctgov-XX-Prot::gold::002"]
        first = failures.obligation_errors(self._verdicts(a + b))
        second = failures.obligation_errors(self._verdicts(b + a))
        assert first["worst_doc"] == second["worst_doc"]
        assert first["worst_doc_n"] == second["worst_doc_n"] == 2

    def test_a_single_document_error_is_not_described_as_a_concentration(self):
        payload = _payload()
        payload["obligations"] = failures.obligation_errors(
            self._verdicts(["fda-75426::gold::001"]), self.FDA_CHUNKS
        )
        assert "come from a single document" not in failures.render_report(payload)

    def test_n_judged_is_never_called_a_flag_count(self):
        """`n_judged` counts recorded verdicts; the current flag count is smaller. Calling
        it "flagged pairs" published 25 as a flag count when 24 pairs were flagged."""
        assert "judged pairs" in failures.render_report(_payload())
        assert "flagged pairs" not in failures.render_report(_payload())
