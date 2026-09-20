"""Tests for PDF triage.

Every test here corresponds to a defect found during the Phase 0 verification
gate. The point is regression, not coverage theatre: each one fails against the
version of the code that shipped before the review.
"""

from __future__ import annotations

from pypdf import PdfWriter

from ragpipe import pdfcheck


class TestClassifyPages:
    """Threshold logic, previously only exercised by whatever was in the corpus."""

    def test_all_pages_blank_is_image_only(self):
        assert pdfcheck.classify_pages([0] * 10) == (10, "image_only")

    def test_all_pages_text_is_digital_native(self):
        assert pdfcheck.classify_pages([5000] * 10) == (0, "digital_native")

    def test_empty_input_is_unreadable(self):
        assert pdfcheck.classify_pages([]) == (0, "unreadable")

    def test_exact_mixed_threshold_is_inclusive(self):
        # 2/10 == 0.20 == MIXED_THRESHOLD -> mixed, not digital_native.
        blank, verdict = pdfcheck.classify_pages([0, 0] + [5000] * 8)
        assert (blank, verdict) == (2, "mixed")

    def test_exact_image_only_threshold_is_inclusive(self):
        blank, verdict = pdfcheck.classify_pages([0] * 8 + [5000] * 2)
        assert (blank, verdict) == (8, "image_only")

    def test_scanned_appendix_below_threshold_reads_as_digital_native(self):
        """The documented behaviour, pinned so the docstring cannot drift again.

        A real corpus document — 16 blank pages of 93 (17%) — sits just under the
        20% bar. An earlier docstring used "a 200-page protocol with 30 scanned
        appendix pages" as its example of `mixed`; that is 15% and would classify
        as digital_native. The verdict is coarse on purpose; `image_only_pages` is
        the per-document signal.
        """
        blank, verdict = pdfcheck.classify_pages([0] * 16 + [5000] * 77)
        assert blank == 16
        assert verdict == "digital_native"

    def test_page_just_under_min_chars_counts_as_blank(self):
        blank, _ = pdfcheck.classify_pages([pdfcheck.MIN_CHARS_PER_TEXT_PAGE - 1] * 5)
        assert blank == 5

    def test_page_at_min_chars_counts_as_text(self):
        blank, _ = pdfcheck.classify_pages([pdfcheck.MIN_CHARS_PER_TEXT_PAGE] * 5)
        assert blank == 0


class TestIdentifierPatterns:
    """Identifier detectors. Both understated their metric before the review."""

    def test_plain_ascii_docket(self):
        assert pdfcheck.DOCKET_TOKEN_RE.findall("see FDA-2016-D-0734 for detail")

    def test_line_wrapped_docket_is_matched(self):
        # Real extraction output from fda-134845; the strict pattern missed it.
        assert pdfcheck.DOCKET_TOKEN_RE.findall("Docket No. FDA-1996-\nD-0012")

    def test_en_dash_docket_is_matched(self):
        # FDA Federal Register typography uses en-dashes (U+2013).
        assert pdfcheck.DOCKET_TOKEN_RE.findall("Docket No. FDA–2022–D–0814")

    def test_registry_id_is_matched_by_its_own_pattern(self):
        assert pdfcheck.REGISTRY_TOKEN_RE.findall("Study NCT07027878 protocol")

    def test_docket_pattern_cannot_match_a_registry_id(self):
        """Why the two are counted separately.

        Bundling them into one metric implied protocol coverage that could not
        exist: an NCT ID has no separators, so the docket pattern can never match.
        """
        assert not pdfcheck.DOCKET_TOKEN_RE.findall("NCT07027878")

    def test_docket_pattern_rejects_near_misses(self):
        assert not pdfcheck.DOCKET_TOKEN_RE.findall("FDA-16-D-0734")  # 2-digit year
        assert not pdfcheck.DOCKET_TOKEN_RE.findall("FDA-2016-DD-0734")  # 2-letter type


class TestInspect:
    """End-to-end inspection against real files."""

    def test_blank_pages_classify_as_image_only(self, tmp_path):
        """A PDF that parses cleanly but yields no text — the practical failure.

        This is the case that would silently poison an index, and it is the reason
        the quarantine path exists.
        """
        writer = PdfWriter()
        for _ in range(6):
            writer.add_blank_page(width=612, height=792)
        path = tmp_path / "blank.pdf"
        with path.open("wb") as fh:
            writer.write(fh)

        summary = pdfcheck.inspect(path)
        assert summary["pages"] == 6
        assert summary["image_only_pages"] == 6
        assert summary["text_layer"] == "image_only"
        assert not pdfcheck.is_indexable(summary["text_layer"])

    def test_corrupt_file_degrades_to_unreadable_and_does_not_raise(self, tmp_path):
        """pypdf's own base class does not cover every failure mode, so `inspect`
        catches broadly. One bad document must never abort a 160-document run."""
        path = tmp_path / "not-a-pdf.pdf"
        path.write_bytes(b"%PDF-1.4\nthis is not a valid pdf body\n")

        summary = pdfcheck.inspect(path)
        assert summary["text_layer"] == "unreadable"
        assert summary["unreadable_reason"]
        assert not pdfcheck.is_indexable(summary["text_layer"])

    def test_missing_file_degrades_to_unreadable(self, tmp_path):
        summary = pdfcheck.inspect(tmp_path / "absent.pdf")
        assert summary["text_layer"] == "unreadable"


class TestIsIndexable:
    def test_mixed_is_indexable(self):
        # A protocol with a scanned appendix is still mostly useful text;
        # excluding it would bias the corpus toward tidy documents.
        assert pdfcheck.is_indexable("mixed")

    def test_image_only_and_unreadable_are_not(self):
        assert not pdfcheck.is_indexable("image_only")
        assert not pdfcheck.is_indexable("unreadable")
        assert not pdfcheck.is_indexable(None)


class TestCharacterSpacing:
    """Text-layer *quality*, as distinct from presence.

    A glyph-positioned PDF yields a full text layer of unusable single characters. It
    passed every emptiness check in this module and sat in the corpus for four phases,
    carrying 5.8% of `structural` chunks — the largest document in the corpus.
    """

    # Verbatim from the corrupt document: "Protocol BP40234 version 9 has been amended".
    CORRUPT = (
        "Pr ot o c ol B P 4 0 2 3 4 v er si o n 9 h a s b e e n a m e n d e d a n d "
        "c h a n g e s t o t h e pr ot o c ol, al o n g wit h a r ati o n al e f or "
        "e a c h c h a n g e, ar e s u m m ari z e d b el o w. "
    ) * 8
    CLEAN = (
        "The applicant shall submit an annual report of adverse events to the Agency "
        "within fifteen calendar days of becoming aware of the information. "
    ) * 12

    def test_detects_the_real_corrupt_text(self):
        assert pdfcheck.is_character_spaced(self.CORRUPT)
        assert pdfcheck.single_char_token_share(self.CORRUPT) > 0.5

    def test_clean_prose_is_not_flagged(self):
        assert not pdfcheck.is_character_spaced(self.CLEAN)
        assert pdfcheck.single_char_token_share(self.CLEAN) < 0.05

    def test_threshold_clears_the_worst_legitimate_document(self):
        """`fda-75894` is an index page of short entries and numbers and scores 0.161 —
        legitimate. The threshold sits 2.5x above it, so this is not finely balanced."""
        assert pdfcheck.CHARACTER_SPACING_THRESHOLD > 0.161 * 2

    def test_short_text_is_never_flagged(self):
        """A couple of list markers must not quarantine a stub."""
        assert pdfcheck.single_char_token_share("a b c d e") == 0.0
        assert not pdfcheck.is_character_spaced("a b c d e")

    def test_list_markers_alone_do_not_trip_it(self):
        """Legitimate documents enumerate with bare letters; that is not corruption."""
        listy = " ".join(f"{chr(97 + i % 26)} item number {i} follows here" for i in range(80))
        assert pdfcheck.single_char_token_share(listy) < pdfcheck.CHARACTER_SPACING_THRESHOLD

    def test_empty_and_whitespace_are_safe(self):
        for t in ("", "   ", "\n\n\t"):
            assert pdfcheck.single_char_token_share(t) == 0.0
            assert not pdfcheck.is_character_spaced(t)

    def test_threshold_is_a_parameter(self):
        """Tuning the threshold must actually change the verdict. `CLEAN` has *zero*
        single-character tokens, so it can never be flagged at any threshold — the
        probe needs text with a small but non-zero share."""
        mild = " ".join(f"a item number {i} follows on this line" for i in range(60))
        share = pdfcheck.single_char_token_share(mild)
        assert 0.0 < share < pdfcheck.CHARACTER_SPACING_THRESHOLD
        assert not pdfcheck.is_character_spaced(mild)
        assert pdfcheck.is_character_spaced(mild, threshold=share / 2)


class TestBrokenEncoding:
    """Phase 5 found two PDFs with no usable ToUnicode map. Fixtures below are
    verbatim extraction output from `ctgov-NCT04506164-Prot_002`, the worst case
    (control-char share 0.130). It is a glyph-code shift: `J==FAF?` decodes to
    `REENING`, and `\\x01` sits where spaces belong."""

    CIPHER = (
        "J==FAF?\x01)\x1f%/\x01OADD\x01:=?AF\x01OAL@\x019\x01\x11\x0bEGFL@\x01H=JAG<\x01\n"
        "L@9L\x01OADD\x01AF;DM<=\x01=/;J==FAF?\x01KG>LO9J=\x01HJGNAKAGF\n"
        "\x01LJ9AFAF?\n\x01.,%3\n\x019F<\x01:D=F<=<\x01>9;ADAL9LAGF\x01>GDDGO=<\x01:Q\x01\x14\x01\n\n"
        "EGFL@K\x01G>\x01GF?GAF?\x01:D=F<=<\x01>9;ADAL9LAGF\x01\x07\x17\x01EGFL@K\x01LGL9D\x08\x0c\x01"
        "\x1d>L=J\x01L@=\x019;LAN=\x01AEHD=E=FL9LAGF\n\x019DD\x01KAL=K\x01OADD\x01@9N=\x01\n"
        "9\x01\x17\x0bEGFL@\x01KMKL9AFE=FL\x01H=JAG<\x0c\x01)MDLAHD=\x01LQH=K\x01G>\x01<9L9\x01OADD\x01"
        ":=\x01;GDD=;L=<\n\x01AF;DM<AF?\x01=D=;LJGFA;\x01E=<A;9D\x01J=;GJ<\x01\n"
        "<9L9\n\x01KL9>>\x01E=9KMJ=K\x019F<\x01AFL=JNA=OK\n\x01>A=D<\x01FGL=K\n\x019F<\x01"
        "LAE=\x01EGLAGF\x01LJ9;CAF?\x01\x07<=K;JA:=<\x01:=DGO\x08\x0c\x01 9L9\x01\n"
        ";GDD=;LAGF\x01OADD\x01:=?AF\x019L\x01HJ=\x0bAEHD=E=FL9LAGF\x01>GJ\x01=9;@\x01KAL=\x01"
        "9F<\x01;GFLAFM=\x01L@JGM?@\x01KMKL9AFE"
    )

    def test_detects_the_real_corrupt_document(self):
        assert pdfcheck.control_char_share(self.CIPHER) >= pdfcheck.CONTROL_CHAR_THRESHOLD
        assert pdfcheck.has_broken_encoding(self.CIPHER)
        assert pdfcheck.text_quality_label(self.CIPHER) == "broken_encoding"

    def test_clean_prose_is_not_flagged(self):
        clean = (
            "Trial length may affect the demonstration of both safety and efficacy. "
            "In terms of efficacy, sponsors should allow sufficient time to demonstrate "
            "a clinically meaningful effect on the primary endpoint. " * 6
        )
        assert pdfcheck.control_char_share(clean) == 0.0
        assert not pdfcheck.has_broken_encoding(clean)

    def test_newlines_and_tabs_are_not_control_chars(self):
        """PDF text is full of these; counting them would flag every document."""
        text = ("line one\nline two\r\ncolumn\tcolumn\n" * 40) + "x" * 200
        assert pdfcheck.control_char_share(text) == 0.0

    def test_fails_open_on_short_text(self):
        assert pdfcheck.control_char_share("\x03\x03\x03") == 0.0
        assert not pdfcheck.has_broken_encoding("\x03\x03\x03")

    def test_threshold_sits_between_measured_populations(self):
        """0.005 must exceed the worst legitimate document (0.00057) and fall below
        the mildest corrupt one (0.03622). If either stops holding, the constant is
        wrong, not the corpus."""
        assert 0.00057 < pdfcheck.CONTROL_CHAR_THRESHOLD < 0.03622


class TestSpaceCollapsed:
    """The gap `single_char_token_share`'s own docstring predicted: a text layer with
    no inter-word spaces. Fixture is verbatim from `ctgov-NCT02129699-Prot_000`, found
    because a *verified* citation came back exact and unreadable."""

    RUNON = (
        "19.3. Informedconsent\n"
        "Informedconsentforeachpatientwillbeobtainedpriortoinitiatinganytrialproceduresin\n"
        'accordance with the “Patient Information and InformedConsent" (see Appendix1). One\n'
        "signed and dated copy of the informed consent must be given to each patient and the\n"
        "originalcopymustberetainedintheinvestigator'strialrecords. Theinformedconsentform\n"
        "mustbeavailableinthecaseofdataaudits.Verificationofsignedinformedconsentandthe\n"
        "datesignedarerequiredforrandomisationtothistrial.\n"
        # Repeated to clear MIN_TOKENS_FOR_SPACING (200); one copy is ~33 tokens, so 8
        # copies give ~264. Repetition leaves the ratio unchanged, which is the point.
    ) * 8

    def test_detects_the_real_document(self):
        assert pdfcheck.runon_token_share(self.RUNON) >= pdfcheck.RUNON_TOKEN_THRESHOLD
        assert pdfcheck.is_space_collapsed(self.RUNON)
        assert pdfcheck.text_quality_label(self.RUNON) == "space_collapsed"

    def test_missed_by_the_phase_4_detector(self):
        """This is the whole reason the detector was added: the character-spacing
        metric scores it far below its gate."""
        assert pdfcheck.single_char_token_share(self.RUNON) < pdfcheck.CHARACTER_SPACING_THRESHOLD
        assert not pdfcheck.is_character_spaced(self.RUNON)

    def test_legitimate_dense_document_is_not_flagged(self):
        """A real FDA table page must stay under the threshold.

        This is an excerpt of `fda-78268`, which is the worst legitimate document in
        the corpus on this metric (0.00029 over its whole text) -- but the excerpt
        itself contains no run-on tokens and so scores 0.0. An earlier docstring
        claimed the fixture reproduced the corpus figure, which it does not; the
        assertion below is what the fixture actually establishes.
        """
        legit = (
            "rities reported on Form \nFDA 3480, and/or information \nsupporting that no "
            "migration to food is \nexpected \nForm FDA 3480 \nII.C.1 \n(Impurities) \nRef. 10\n"
            " \n 46\n\nContains Nonbinding Recommendations \nDraft-Not for Implementation \n"
            # x8, not x6. At x6 the fixture is 180 tokens, under
            # MIN_TOKENS_FOR_SPACING=200, so `runon_token_share` returned the fail-open 0.0
            # and this test passed without measuring anything -- the same reason
            # `test_fails_open_on_short_text` passes. Caught by the code-review gate.
        ) * 8
        # The guard assertion is the point: without it the metric fails open and the
        # test proves nothing.
        assert len(legit.split()) >= pdfcheck.MIN_TOKENS_FOR_SPACING
        assert pdfcheck.runon_token_share(legit) < pdfcheck.RUNON_TOKEN_THRESHOLD
        assert pdfcheck.text_quality_label(legit) == "ok"

    def test_long_non_alpha_tokens_do_not_count(self):
        """Dotted ToC leaders and rule characters are ordinary here; restricting to
        .isalpha() is what keeps them from flagging legitimate pages."""
        toc = ("Section " + "." * 40 + " 12\n") * 80
        assert len(toc.split()) >= pdfcheck.MIN_TOKENS_FOR_SPACING
        assert pdfcheck.runon_token_share(toc) == 0.0

    def test_fails_open_on_short_text(self):
        assert pdfcheck.runon_token_share("Informedconsentforeachpatientwill") == 0.0

    def test_threshold_sits_between_measured_populations(self):
        assert 0.00026 < pdfcheck.RUNON_TOKEN_THRESHOLD < 0.01759


class TestTextQualityLabel:
    def test_clean_text_is_ok(self):
        clean = "The sponsor should submit the completed application to the agency. " * 20
        assert pdfcheck.text_quality_label(clean) == "ok"

    def test_space_collapsed_is_not_quarantined(self):
        """Its text is degraded but readable, and Phase 5 confirmed citations into it
        verify at exact offsets. Discarding recoverable evidence would be worse."""
        assert "space_collapsed" not in pdfcheck.QUARANTINE_QUALITIES

    def test_unreadable_classes_are_quarantined(self):
        """Membership only. See `TestQuarantineIsActuallyWired` for the behaviour --
        this assertion passed for a whole phase while the pipeline ignored the set."""
        assert "broken_encoding" in pdfcheck.QUARANTINE_QUALITIES
        assert "character_spaced" in pdfcheck.QUARANTINE_QUALITIES

    def test_ok_is_never_quarantined(self):
        assert "ok" not in pdfcheck.QUARANTINE_QUALITIES


class TestQuarantineIsActuallyWired:
    """Behaviour, not declaration.

    `QUARANTINE_QUALITIES` listed `broken_encoding` for a whole phase while
    `cli._is_quarantined` compared against the string `"character_spaced"` alone, so
    two documents with Caesar-shifted glyph-code text layers stayed in the shipped
    index — 283 chunks, 2.06% of it — while the constant declared them unindexable.
    A frozenset-membership assertion cannot catch that.
    """

    def _q(self, record):
        from ragpipe.cli import _is_quarantined

        return _is_quarantined(record)

    def test_broken_encoding_verdict_quarantines(self):
        assert self._q({"text_quality": "broken_encoding"})

    def test_broken_encoding_ratio_quarantines_even_with_a_stale_verdict(self):
        """Extraction is cached by content hash, so a stored verdict can predate the
        current threshold. The live constant must stay authoritative."""
        assert self._q({"text_quality": "ok", "control_char_share": 0.13})

    def test_character_spaced_still_quarantines(self):
        assert self._q({"text_quality": "character_spaced"})
        assert self._q({"text_quality": "ok", "single_char_token_share": 0.73})

    def test_space_collapsed_is_indexed_not_quarantined(self):
        """Deliberate: its text is readable and its citations verify at exact offsets."""
        assert not self._q({"text_quality": "space_collapsed", "runon_token_share": 0.018})

    def test_clean_document_is_indexed(self):
        assert not self._q({"text_quality": "ok"})

    def test_missing_fields_default_to_indexable(self):
        assert not self._q({})

    def test_every_quarantine_quality_is_honoured(self):
        """Guards the coupling itself: adding a quality to the set must change
        behaviour, not just documentation."""
        for quality in pdfcheck.QUARANTINE_QUALITIES:
            assert self._q({"text_quality": quality}), quality


class TestExtractPersistsTheNewRatios:
    def test_extracted_doc_carries_both_new_shares(self):
        """The detectors were unreachable because nothing stored their output."""
        from ragpipe.extract import ExtractedDoc

        fields = ExtractedDoc.__dataclass_fields__
        assert "control_char_share" in fields
        assert "runon_token_share" in fields
