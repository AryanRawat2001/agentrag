"""Tests for near-duplicate detection.

This file exists because its absence was itself a finding: five defects in
`dedup.py` were unguarded, and the first one below — non-determinism across
processes — would have been caught by a single test comparing two subprocess runs.

The determinism test deliberately shells out. Python salts `str.__hash__` per
process, so a same-process test cannot detect the bug it exists to prevent.
"""

from __future__ import annotations

import subprocess
import sys

from ragpipe import dedup

BOILERPLATE = (
    "FDA's guidance documents, including this guidance, do not establish legally "
    "enforceable responsibilities. Instead, guidances describe the Agency's current "
    "thinking on a topic and should be viewed only as recommendations, unless "
    "specific regulatory or statutory requirements are cited."
)


class TestDeterminism:
    def test_signatures_are_identical_across_processes(self):
        """The bug this file was created for.

        `shingles()` used Python's builtin `hash()`, which is salted per process, so
        every signature, LSH bucket, cluster and similarity figure changed on each
        run — while three docstrings claimed determinism and the committed cluster
        files could not be reproduced.
        """
        script = (
            "from ragpipe.dedup import shingles, signature;"
            f"print(signature(shingles({BOILERPLATE!r}))[:6])"
        )
        outputs = {
            subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, check=True
            ).stdout.strip()
            for _ in range(3)
        }
        assert len(outputs) == 1, f"non-deterministic across processes: {outputs}"

    def test_shingles_are_identical_across_processes(self):
        script = f"from ragpipe.dedup import shingles;print(sorted(shingles({BOILERPLATE!r}))[:3])"
        outputs = {
            subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, check=True
            ).stdout.strip()
            for _ in range(3)
        }
        assert len(outputs) == 1

    def test_clustering_is_reproducible(self):
        items = [(f"c{i}", BOILERPLATE + f" Variation {i}.") for i in range(8)]
        first = dedup.find_duplicates(items)
        second = dedup.find_duplicates(list(items))
        assert first[0] == second[0]
        assert [c.member_ids for c in first[1]] == [c.member_ids for c in second[1]]


class TestJaccard:
    def test_exact_jaccard_matches_hand_computation(self):
        a, b = {1, 2, 3, 4}, {3, 4, 5, 6}
        assert dedup.exact_jaccard(a, b) == 2 / 6

    def test_identical_sets(self):
        assert dedup.exact_jaccard({1, 2}, {1, 2}) == 1.0

    def test_disjoint_sets(self):
        assert dedup.exact_jaccard({1}, {2}) == 0.0

    def test_two_empty_sets_are_identical(self):
        assert dedup.exact_jaccard(set(), set()) == 1.0

    def test_empty_versus_nonempty(self):
        assert dedup.exact_jaccard(set(), {1, 2}) == 0.0

    def test_estimate_tracks_the_true_value(self):
        """The estimate is allowed to be approximate — it only screens candidates —
        but it must correlate, or screening drops real duplicates."""
        base = BOILERPLATE
        for suffix_words in (0, 5, 20, 60):
            other = base + " extra" * suffix_words
            sa, sb = dedup.shingles(base), dedup.shingles(other)
            true = dedup.exact_jaccard(sa, sb)
            est = dedup.estimated_jaccard(dedup.signature(sa), dedup.signature(sb))
            assert abs(true - est) < 0.15, f"true={true} est={est}"

    def test_decision_uses_exact_not_estimated_jaccard(self):
        """`find_duplicates` must never cluster a pair whose true Jaccard is below
        the threshold. The estimate's standard error at J=0.85 is ~0.032, which
        previously admitted 4 of 46 shipped pairs below the stated bar.
        """
        items = [
            ("a", BOILERPLATE),
            ("b", BOILERPLATE + " " + " ".join(f"word{i}" for i in range(40))),
        ]
        duplicate_of, clusters = dedup.find_duplicates(items, threshold=0.85)
        true = dedup.exact_jaccard(dedup.shingles(items[0][1]), dedup.shingles(items[1][1]))
        if true < 0.85:
            assert not duplicate_of, f"clustered a pair at true Jaccard {true:.3f}"
        for cluster in clusters:
            assert cluster.similarity_floor >= 0.85


class TestShingles:
    def test_reflowed_whitespace_produces_identical_shingles(self):
        a = "the applicant shall submit the report within fifteen days"
        b = "the  applicant\nshall submit   the report\nwithin fifteen days"
        assert dedup.shingles(a) == dedup.shingles(b)

    def test_case_and_punctuation_are_ignored(self):
        assert dedup.shingles("The Applicant, shall submit!") == dedup.shingles(
            "the applicant shall submit"
        )

    def test_text_shorter_than_shingle_size_yields_one_shingle(self):
        assert len(dedup.shingles("two words")) == 1

    def test_empty_text_yields_no_shingles(self):
        assert dedup.shingles("") == set()

    def test_punctuation_only_text_yields_no_shingles(self):
        assert dedup.shingles("... !!! ---") == set()


class TestFindDuplicates:
    def test_exact_duplicates_are_clustered(self):
        items = [("a", BOILERPLATE), ("b", BOILERPLATE), ("c", "something else entirely")]
        duplicate_of, clusters = dedup.find_duplicates(items)
        assert len(clusters) == 1
        assert set(clusters[0].member_ids) == {"a", "b"}
        assert "c" not in duplicate_of

    def test_unique_texts_produce_no_clusters(self):
        items = [(f"c{i}", f"Wholly distinct sentence number {i} about nothing.") for i in range(5)]
        duplicate_of, clusters = dedup.find_duplicates(items)
        assert duplicate_of == {}
        assert clusters == []

    def test_canonical_is_the_longest_member(self):
        long_text = BOILERPLATE + " With an additional trailing clause for length."
        items = [("short", BOILERPLATE), ("long", long_text)]
        duplicate_of, clusters = dedup.find_duplicates(items, threshold=0.5)
        assert clusters
        assert clusters[0].canonical_id == "long"
        assert duplicate_of == {"short": "long"}

    def test_similarity_floor_covers_every_pair_not_just_the_canonical(self):
        """Union-find is transitive, so two members can be clustered without ever
        being compared directly. A floor measured only against the canonical member
        reported 0.891 for a cluster whose true minimum pairwise was 0.781."""
        items = [
            ("a", BOILERPLATE),
            ("b", BOILERPLATE + " One extra sentence here."),
            ("c", BOILERPLATE + " One extra sentence here. And a second one too."),
        ]
        _, clusters = dedup.find_duplicates(items, threshold=0.5)
        for cluster in clusters:
            members = cluster.member_ids
            texts = dict(items)
            worst = min(
                dedup.exact_jaccard(dedup.shingles(texts[x]), dedup.shingles(texts[y]))
                for i, x in enumerate(members)
                for y in members[i + 1 :]
            )
            # similarity_floor is rounded to 3 decimals for reporting.
            assert abs(cluster.similarity_floor - worst) <= 5e-4

    def test_empty_input(self):
        assert dedup.find_duplicates([]) == ({}, [])

    def test_single_item(self):
        assert dedup.find_duplicates([("a", BOILERPLATE)]) == ({}, [])

    def test_punctuation_only_texts_are_not_clustered_together(self):
        """Two texts with no word characters both hash to the empty shingle set, and
        `signature(set())` returns an all-max signature — so their estimated Jaccard
        is 1.0. The exact check must stop them being called duplicates of each other
        on that basis alone."""
        items = [("a", "... ... ..."), ("b", "!!! !!! !!!"), ("c", BOILERPLATE)]
        duplicate_of, _ = dedup.find_duplicates(items)
        assert "c" not in duplicate_of
