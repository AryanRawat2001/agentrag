"""Tests for exact-identifier extraction.

`distinct` identifier counts size the exact-identifier eval slice, so
canonicalisation is load-bearing rather than cosmetic: if "21CFR 312.32" and
"21 CFR 312.32" count as two citations, the slice looks bigger than it is.
"""

from __future__ import annotations

from ragpipe import identifiers as I


class TestCanonicalCfr:
    """Every spelling of the same citation must collapse to one value."""

    def test_variants_of_one_citation_collapse(self):
        variants = [
            "21 CFR 312.32",
            "21CFR 312.32",
            "21 C.F.R. 312.32",
            "21 CFR Part 312.32",
            "21 CFR §312.32",
            "21 CFR\n312.32",
            "21 cfr 312.32",
            "21  CFR  312.32",
        ]
        assert len({I.canonical("cfr", v) for v in variants}) == 1
        assert I.canonical("cfr", variants[0]) == "21 CFR 312.32"

    def test_part_keyword_is_dropped(self):
        assert I.canonical("cfr", "21 CFR PART 50") == "21 CFR 50"

    def test_distinct_sections_stay_distinct(self):
        assert I.canonical("cfr", "21 CFR 312.32") != I.canonical("cfr", "21 CFR 312.33")

    def test_distinct_titles_stay_distinct(self):
        assert I.canonical("cfr", "21 CFR 50") != I.canonical("cfr", "42 CFR 50")


class TestCanonicalOtherKinds:
    def test_usc(self):
        for v in ["42 U.S.C. 262", "42 USC 262", "42 U.S.C. §262", "42 USC\n262"]:
            assert I.canonical("usc", v) == "42 USC 262"

    def test_usc_subsection_moves_to_the_detailed_form(self):
        """`canonical` is section level for *every* class; subsections live in
        `canonical_detailed`.

        This test previously asserted that `canonical` preserved the USC
        subsection. That was the defect: CFR discarded subsections while USC kept
        them, so the two classes' distinct counts were measured at different
        resolutions inside one reported column. Both granularities are now
        explicit and consistent.
        """
        section, detailed = I.canonical_pair("usc", "21 U.S.C. 355(b)(1)")
        assert section == "21 USC 355"
        assert detailed == "21 USC 355(b)(1)"

    def test_cfr_subsection_also_available_in_detailed_form(self):
        section, detailed = I.canonical_pair("cfr", "21 CFR 117.136(a)(2)(i)")
        assert section == "21 CFR 117.136"
        assert detailed == "21 CFR 117.136(a)(2)(i)"

    def test_subsection_case_is_preserved_in_detailed_form(self):
        # (C) and (c) are different subdivisions; lowercasing them would collide.
        _, detailed = I.canonical_pair("usc", "21 U.S.C. 353(g)(1)(C)")
        assert detailed == "21 USC 353(g)(1)(C)"

    def test_federal_register(self):
        assert I.canonical("fed_register", "85 F.R. 12345") == "85 FR 12345"

    def test_registry_id_space_removed(self):
        assert I.canonical("registry", "NCT 01234567") == "NCT01234567"
        assert I.canonical("registry", "NCT01234567") == "NCT01234567"

    def test_ich_with_and_without_revision(self):
        assert I.canonical("ich", "ICH E6(R2)") == "ICH E6(R2)"
        assert I.canonical("ich", "ich e6 (r2)") == "ICH E6(R2)"
        assert I.canonical("ich", "ICH Q3C") == "ICH Q3C"

    def test_docket_dashes_normalised(self):
        # FDA Federal Register typography uses en-dashes.
        assert I.canonical("docket", "FDA–2022–D–0814") == "FDA-2022-D-0814"
        assert I.canonical("docket", "FDA-2016-\nD-0012") == "FDA-2016-D-0012"


class TestExtract:
    def test_finds_each_class(self):
        text = (
            "Per 21 CFR 312.32 and 42 U.S.C. 262, see 85 FR 12345, "
            "docket FDA-2016-D-0734, study NCT01234567, guideline ICH E6(R2)."
        )
        kinds = {i.kind for i in I.extract(text)}
        assert kinds == {"cfr", "usc", "fed_register", "docket", "registry", "ich"}

    def test_offsets_locate_the_match(self):
        text = "xxxx 21 CFR 312.32 yyyy"
        (found,) = [i for i in I.extract(text) if i.kind == "cfr"]
        assert text[found.start : found.end] == found.raw
        assert "312.32" in found.raw

    def test_results_are_position_sorted(self):
        text = "NCT01234567 then 21 CFR 50 then 42 USC 262"
        found = I.extract(text)
        assert [i.start for i in found] == sorted(i.start for i in found)

    def test_line_wrapped_identifier_is_found(self):
        found = I.extract("Docket No. FDA-1996-\nD-0012 applies")
        assert any(i.kind == "docket" for i in found)

    def test_empty_text(self):
        assert I.extract("") == []

    def test_kind_filter(self):
        text = "21 CFR 50 and NCT01234567"
        assert {i.kind for i in I.extract(text, kinds=["cfr"])} == {"cfr"}


class TestDensityAndDistinct:
    def test_density_counts_occurrences_not_distinct_values(self):
        text = "21 CFR 50, 21 CFR 50, 21CFR 50"
        found = I.extract(text)
        assert I.density(found)["cfr"] == 3
        assert I.distinct(found, "cfr") == {"21 CFR 50"}

    def test_density_reports_zero_for_absent_kinds(self):
        counts = I.density(I.extract("21 CFR 50"))
        assert counts["cfr"] == 1
        assert counts["registry"] == 0
        assert set(counts) == set(I.PATTERNS)

    def test_distinct_across_all_kinds(self):
        found = I.extract("21 CFR 50 and NCT01234567")
        assert I.distinct(found) == {"21 CFR 50", "NCT01234567"}

    def test_primary_kind_is_a_real_pattern(self):
        # The eval harness and the report both key off this constant.
        assert I.PRIMARY_KIND in I.PATTERNS
