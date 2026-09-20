"""Tests for FDA index field parsing.

The office taxonomy is the trap in this source, and it has now bitten twice in
opposite directions: first by stripping HTML before splitting (which concatenated
distinct offices), then by splitting on every comma (which shredded office names
that legitimately contain commas). Both directions are pinned here.
"""

from __future__ import annotations

from ragpipe.sources.fda import (
    _parse_date,
    _split_list,
    _split_offices,
    normalise_title,
)


class TestSplitOffices:
    def test_single_office(self):
        assert _split_offices("Center for Veterinary Medicine") == [
            "Center for Veterinary Medicine"
        ]

    def test_br_delimited_offices_are_separated(self):
        """Strip the tags first and these silently concatenate into
        'Center for Drug Evaluation and ResearchCenter for Biologics...'."""
        raw = (
            "Center for Devices and Radiological Health<br><br>"
            "Center for Biologics Evaluation and Research"
        )
        assert _split_offices(raw) == [
            "Center for Devices and Radiological Health",
            "Center for Biologics Evaluation and Research",
        ]

    def test_comma_without_space_separates_offices(self):
        raw = "Center for Drug Evaluation and Research,Office of Regulatory Policy"
        assert _split_offices(raw) == [
            "Center for Drug Evaluation and Research",
            "Office of Regulatory Policy",
        ]

    def test_comma_with_space_is_part_of_the_office_name(self):
        """The defect this test exists for.

        Real value from fda-75334. Splitting on every comma produced
        ['Office of the Commissioner', 'Office of Policy', 'Legislation',
         'and International Affairs', ...] — three fake offices from one real one,
        which also inflated the multi-office data-quality count.
        """
        raw = (
            "Office of the Commissioner,"
            "Office of Policy, Legislation, and International Affairs,"
            "Office of Policy"
        )
        assert _split_offices(raw) == [
            "Office of the Commissioner",
            "Office of Policy, Legislation, and International Affairs",
            "Office of Policy",
        ]

    def test_combined_br_and_comma_forms(self):
        raw = (
            "Office of the Commissioner,Office of the Chief Scientist<br><br>"
            "Center for Devices and Radiological Health,"
            "Office of Communication, Information Disclosure, Training and Education"
        )
        assert _split_offices(raw) == [
            "Office of the Commissioner",
            "Office of the Chief Scientist",
            "Center for Devices and Radiological Health",
            "Office of Communication, Information Disclosure, Training and Education",
        ]

    def test_empty_and_none(self):
        assert _split_offices(None) == []
        assert _split_offices("") == []
        assert _split_offices("<br><br>") == []

    def test_duplicates_are_dropped(self):
        raw = "Human Foods Program<br><br>Human Foods Program"
        assert _split_offices(raw) == ["Human Foods Program"]


class TestParseDate:
    def test_valid_date(self):
        assert _parse_date("09/12/2017") == "2017-09-12"

    def test_placeholder_year_is_rejected(self):
        """The index bottoms out at 1900, which is a placeholder, not an issue date."""
        assert _parse_date("01/01/1900") is None

    def test_boundary_year_is_accepted(self):
        # The comment said "at or before"; the code uses `<`. 1938 is accepted.
        assert _parse_date("06/25/1938") == "1938-06-25"

    def test_malformed_and_empty(self):
        assert _parse_date("not a date") is None
        assert _parse_date("") is None
        assert _parse_date(None) is None
        assert _parse_date("2017-09-12") is None  # wrong format for this source


class TestSplitList:
    def test_comma_delimited(self):
        assert _split_list("Drugs, Biologics, Medical Devices") == [
            "Drugs",
            "Biologics",
            "Medical Devices",
        ]

    def test_br_delimited_does_not_concatenate(self):
        """These fields are comma-delimited in practice, but the concatenation
        trap is a property of the source index, not of one field."""
        assert _split_list("Drugs<br><br>Biologics") == ["Drugs", "Biologics"]

    def test_empty(self):
        assert _split_list(None) == []
        assert _split_list("") == []


class TestNormaliseTitle:
    def test_draft_and_final_of_same_guidance_normalise_together(self):
        a = "Bioequivalence Studies: Draft Guidance for Industry"
        b = "Bioequivalence Studies: Final Guidance for Industry"
        assert normalise_title(a) == normalise_title(b)

    def test_revision_markers_are_stripped(self):
        a = "Electronic Records; Revision 2"
        b = "Electronic Records"
        assert normalise_title(a) == normalise_title(b)

    def test_distinct_titles_stay_distinct(self):
        assert normalise_title("Nasal Sprays") != normalise_title("Oral Tablets")
