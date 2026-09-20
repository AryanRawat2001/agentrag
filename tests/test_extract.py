"""Tests for text extraction and section detection.

The outline-validation tests exist because of a finding from this corpus: a PDF
exposing an accessibility *structure tree* through the bookmark API produced 1,618
"outline entries" for a 25-page document, with a median heading length of 3
characters. An outline's existence is not evidence that it is a table of contents.
"""

from __future__ import annotations

from ragpipe.extract import (
    _heading_from_line,
    _looks_like_toc,
    outline_is_usable,
)


class TestOutlineValidation:
    def test_real_table_of_contents_is_accepted(self):
        outline = [(f"{i}. Study Design Section", 1, i) for i in range(20)]
        usable, reason = outline_is_usable(outline, n_pages=25)
        assert usable, reason

    def test_tag_tree_is_rejected_by_entry_density(self):
        """The real failure: fda-71536, 1,618 entries over 25 pages."""
        outline = [("Guidance for Industry", 1, 0) for _ in range(1618)]
        usable, reason = outline_is_usable(outline, n_pages=25)
        assert not usable
        assert "entries per page" in reason

    def test_rejected_when_titles_are_not_substantive(self):
        # Single characters and empty strings, as produced by a tag tree.
        outline = [("-", 1, i % 20) for i in range(40)]
        usable, reason = outline_is_usable(outline, n_pages=100)
        assert not usable
        assert "word characters" in reason

    def test_rejected_when_everything_resolves_to_one_page(self):
        outline = [(f"Heading Number {i}", 1, 0) for i in range(30)]
        usable, reason = outline_is_usable(outline, n_pages=100)
        assert not usable
        assert "single page" in reason

    def test_rejected_when_too_few_entries(self):
        usable, reason = outline_is_usable([("Only One", 1, 0)], n_pages=10)
        assert not usable
        assert "fewer than 2" in reason

    def test_empty_outline_is_rejected(self):
        assert not outline_is_usable([], n_pages=10)[0]


class TestHeadingDetection:
    def test_decimal_numbered_heading(self):
        assert _heading_from_line("4.2.1 Study Design") == ("4.2.1 Study Design", 3)

    def test_top_level_numbered_heading(self):
        assert _heading_from_line("1 Introduction") == ("1 Introduction", 1)

    def test_roman_numeral_heading(self):
        assert _heading_from_line("II. Considerations") == ("II. Considerations", 1)

    def test_lettered_heading(self):
        assert _heading_from_line("A. Endpoints") == ("A. Endpoints", 2)

    def test_all_caps_heading(self):
        assert _heading_from_line("CLINICAL STUDY PROTOCOL") == (
            "CLINICAL STUDY PROTOCOL",
            1,
        )

    def test_citation_line_is_not_a_heading(self):
        """The most common false positive: a CFR citation matches the
        numbered-heading shape but is a reference."""
        assert _heading_from_line("21 CFR 820.30, Subpart C - Design Controls") is None
        assert _heading_from_line("42 U.S.C. 262 applies here") is None

    def test_sentence_is_not_a_heading(self):
        assert _heading_from_line("1 This is a sentence that ends with a period.") is None

    def test_long_line_is_not_a_heading(self):
        long_line = "2 " + " ".join(["word"] * 30)
        assert _heading_from_line(long_line) is None

    def test_lowercase_continuation_is_not_a_heading(self):
        assert _heading_from_line("2 continued from the previous page") is None

    def test_toc_line_is_not_a_heading(self):
        assert _heading_from_line("4.2 Study Design .............. 17") is None

    def test_single_word_caps_is_not_a_heading(self):
        # Page furniture and figure labels.
        assert _heading_from_line("FIGURE") is None

    def test_caps_without_vowels_is_not_a_heading(self):
        assert _heading_from_line("XX YY ZZ") is None

    def test_blank_and_whitespace(self):
        assert _heading_from_line("") is None
        assert _heading_from_line("    ") is None


class TestTocPageDetection:
    def test_toc_page_is_detected(self):
        page = "\n".join(
            [
                "TABLE OF CONTENTS",
                "1 Introduction ............ 3",
                "2 Background ............. 5",
                "3 Methods ................ 9",
                "4 Results ................ 14",
            ]
        )
        assert _looks_like_toc(page)

    def test_body_page_is_not_a_toc(self):
        page = "This is ordinary body text.\nIt continues for several lines.\n"
        assert not _looks_like_toc(page)

    def test_page_with_one_dotted_line_is_not_a_toc(self):
        page = "Some text\n1 Introduction ......... 3\nMore body text here.\n"
        assert not _looks_like_toc(page)
