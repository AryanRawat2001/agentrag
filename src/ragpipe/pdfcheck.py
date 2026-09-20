"""PDF triage: is there a usable text layer, and what shape is the content?

This runs at ingest for one reason: a scanned PDF extracts to empty strings or
OCR noise, and if it enters the index silently it degrades retrieval in a way
that is very hard to diagnose three phases later. Better to detect it here,
quarantine it, and report the rate.

Classification is by the fraction of pages with almost no extractable text,
rather than a whole-document character count, so a mostly-text document with a
scanned section is not averaged into looking fine.

**The thresholds are coarse, and the report must say so.** At
MIXED_THRESHOLD = 0.20 a document needs a fifth of its pages blank to be labelled
`mixed`. In the current corpus 23 indexable documents contain at least one
image-only page but only one clears that bar — e.g. a 93-page protocol with 16
blank pages (17%) is still `digital_native`. `image_only_pages` is therefore the
honest per-document signal, and the verdict is a coarse routing decision layered
on top of it. Both are reported.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

# A page below this many extracted characters is treated as having no text
# layer. Digital-native pages in this corpus run to thousands; genuinely sparse
# pages (a section divider, a single figure) sit well under 100.
MIN_CHARS_PER_TEXT_PAGE = 100

IMAGE_ONLY_THRESHOLD = 0.80  # >= this fraction of blank pages -> image_only
MIXED_THRESHOLD = 0.20  # >= this fraction -> mixed

INDEXABLE_VERDICTS = frozenset({"digital_native", "mixed"})

# --- character-spacing detection ---------------------------------------------
#
# The checks above answer "is there text?". This one answers "is the text words?".
# They are not the same question, and conflating them let a corrupt document into
# the corpus for four phases.
#
# Some PDFs position every glyph individually with no space semantics. Extraction
# then yields `Pr ot o c ol B P 4 0 2 3 4 v er si o n 9` for "Protocol BP40234
# version 9" — a full text layer that passes every emptiness check while being
# unusable for retrieval. It is *not* repairable: intra-word and inter-word gaps are
# both a single space, so word boundaries are unrecoverable without dictionary-based
# re-segmentation.
#
# The signal is the share of whitespace-delimited tokens that are a single character.
# Measured over this corpus (155 documents):
#
#     median                      0.033
#     worst legitimate document   0.159  (an FDA index page of short entries + numbers)
#     the corrupt document        0.723
#
# 0.40 sits 2.5x above the worst legitimate case (0.40/0.159 = 2.52) and 0.32 below the
# corrupt one, so
# the threshold is not finely balanced. `MIN_TOKENS_FOR_SPACING` keeps the ratio off
# short fragments, where a couple of list markers would swamp it.
CHARACTER_SPACING_THRESHOLD = 0.40
MIN_TOKENS_FOR_SPACING = 200


def single_char_token_share(text: str) -> float:
    """Share of whitespace-delimited tokens that are exactly one character.

    Returns 0.0 for text too short to judge, so a caller cannot accidentally
    quarantine a stub on the strength of two list markers.

    **Known limitation, stated because it is the mirror image of what this catches.**
    The metric fails *open* on text with too few whitespace-delimited tokens for any
    reason — including a long document with no whitespace at all, which collapses to a
    single token and scores 0.0. A glyph-positioned PDF that emits no inter-glyph
    spaces would therefore pass the very check added to catch glyph-positioned PDFs.
    Not present in this corpus (the one corrupt document does emit spaces, which is
    why the ratio works), so this is left as a documented gap rather than a speculative
    second detector: `chars_per_token` below is the signal that would catch it.
    """
    tokens = text.split()
    if len(tokens) < MIN_TOKENS_FOR_SPACING:
        return 0.0
    return sum(1 for t in tokens if len(t) == 1) / len(tokens)


def is_character_spaced(text: str, threshold: float = CHARACTER_SPACING_THRESHOLD) -> bool:
    """True when the text layer is glyph-positioned noise rather than words.

    Such a document is quarantined rather than indexed, on the same reasoning as a
    scanned PDF: text that cannot be read cannot be retrieved, and letting it in
    silently degrades every metric downstream while looking like a plausible corpus.
    """
    return single_char_token_share(text) >= threshold


# Two more text-layer defect classes, found in Phase 5 by reading a *verified
# citation* rather than by a test. The quote came back exact and unusable:
# 'Informedconsentforeachpatientwillbeobtainedpriortoinitiatinganytrialprocedures'.
#
# Both are misses of `single_char_token_share`, and one of them is the exact gap its
# own docstring predicted above — a document whose text has no inter-word spaces
# collapses toward one long token and scores 0.0 on a metric that counts one-character
# tokens. The prediction was right; what was missing was an instance to measure.
#
# Measured over the 154 indexable documents, on each document's **whole extracted
# text** -- the basis these functions are actually called on:
#
#                          median    p95     worst legit    corrupt
#   control-char share     0.00000  0.00000    0.00053     0.03574, 0.13096
#   run-on token share     0.00000  0.00000    0.00029     0.01794
#
# An earlier version of this table reported 0.00057 / 0.03622 / 0.13019 / 0.00026 /
# 0.01759. Those came from concatenating each document's *chunks*, which double-counts
# the 15% inter-chunk overlap -- so the label said "per document, whole text" while the
# numbers described something else. The Phase 5 audit gate caught it by reproducing the
# old figures exactly under chunk-concatenation and getting different ones from the
# document text. Classification is unchanged (same 151/2/1, same documents); what was
# wrong was the calibration basis, which matters because `extract.py` calls these on
# whole document text.
#
# Both classes separate cleanly: the median and p95 are *exactly* zero, so any
# non-trivial value is already anomalous. Thresholds are set near the geometric
# midpoint between the worst legitimate document and the mildest corrupt one, which
# keeps the margin symmetric in ratio rather than in absolute terms — the right
# choice when the healthy population sits at zero and the scale is arbitrary:
#
#   control chars: 0.005 is 9.4x above 0.00053 and 7.2x below 0.03574
#   run-on tokens: 0.002 is 6.8x above 0.00029 and 9.0x below 0.01794
#
# (True geometric midpoints are 0.004355 and 0.002300; the shipped constants are
# 1-significant-figure roundings of those, which is why the ratios are not equal.)
CONTROL_CHAR_THRESHOLD = 0.005
RUNON_TOKEN_THRESHOLD = 0.002
MIN_CHARS_FOR_CONTROL = 500
RUNON_MIN_TOKEN_CHARS = 25

# Qualities that make a document unindexable. `space_collapsed` is deliberately NOT
# here: its text is degraded but *readable*, and Phase 5 confirmed empirically that
# citations into it verify exactly — the informed-consent quote above located at a
# real offset. It harms BM25 tokenization (a whole clause becomes one token) and
# dense embedding, so it is worth reporting, but quarantining readable text would
# discard recoverable evidence. `broken_encoding` and `character_spaced` are not
# readable by anything.
QUARANTINE_QUALITIES = frozenset({"character_spaced", "broken_encoding"})


def control_char_share(text: str) -> float:
    """Share of characters that are C0/C1 control codes other than newline, carriage
    return, and tab.

    Catches PDFs with no usable `ToUnicode` map, where extraction emits raw glyph
    codes instead of characters. In this corpus that looks like
    `(IIHFWLYHQHVV\\x03DQG\\x03,PSOHPHQWDWLRQ` — a Caesar-shifted text layer
    ("EFFECTIVENESS AND IMPLEMENTATION") padded with `\\x03` where spaces belong.

    The text is *not* repairable in general: the shift is a property of the embedded
    font's encoding, differs per document, and applies only to the subset of glyphs
    that font covers. Returns 0.0 for text too short to judge.
    """
    if len(text) < MIN_CHARS_FOR_CONTROL:
        return 0.0
    controls = sum(1 for ch in text if unicodedata.category(ch) == "Cc" and ch not in "\n\r\t")
    return controls / len(text)


def has_broken_encoding(text: str, threshold: float = CONTROL_CHAR_THRESHOLD) -> bool:
    """True when the text layer is raw glyph codes rather than characters."""
    return control_char_share(text) >= threshold


def runon_token_share(text: str) -> float:
    """Share of tokens that are implausibly long runs of letters.

    A PDF that positions words without emitting inter-word spaces extracts as
    `Informedconsentforeachpatientwillbeobtained...`. Restricting to `.isalpha()`
    matters: long non-alphabetic tokens are ordinary in this corpus (table rules,
    dotted table-of-contents leaders, hyphenated identifier strings), and counting
    them would flag legitimate documents. Returns 0.0 for text too short to judge.
    """
    tokens = text.split()
    if len(tokens) < MIN_TOKENS_FOR_SPACING:
        return 0.0
    runons = sum(1 for t in tokens if len(t) >= RUNON_MIN_TOKEN_CHARS and t.isalpha())
    return runons / len(tokens)


def is_space_collapsed(text: str, threshold: float = RUNON_TOKEN_THRESHOLD) -> bool:
    """True when the text layer has lost its inter-word spaces."""
    return runon_token_share(text) >= threshold


def text_quality_label(text: str) -> str:
    """Classify a text layer: `ok`, `character_spaced`, `broken_encoding`, or
    `space_collapsed`.

    Precedence runs worst-first, as a defensive ordering rather than a measured
    necessity. On this corpus the three classes do not overlap: the two
    `broken_encoding` documents score 0.066 and 0.069 on single-character token
    share, far below the 0.40 gate, so neither would be misfiled by the reverse
    order. The ordering is therefore untested by real data and exists so that a
    document tripping two detectors is reported under the more specific and more
    severe one instead of whichever check happened to run first.
    """
    if has_broken_encoding(text):
        return "broken_encoding"
    if is_character_spaced(text):
        return "character_spaced"
    if is_space_collapsed(text):
        return "space_collapsed"
    return "ok"


# Dash characters FDA typography actually uses in docket numbers. The Federal
# Register style sets them as en-dashes (U+2013), and PDF text extraction also
# breaks identifiers across lines, so a strict ASCII-hyphen pattern silently
# misses real occurrences. Verified: `FDA-1996-\nD-0012`, `FDA– 2022–D–0814`.
_DASH = r"[-‐‑‒–—]"

# FDA docket identifiers, e.g. FDA-2016-D-0734. These are the queries where
# lexical search beats dense retrieval outright, so counting them tells us
# whether the exact-identifier eval slice is viable on this corpus.
# `\s*` after each dash absorbs the line wraps introduced by extraction.
DOCKET_TOKEN_RE = re.compile(
    rf"\b[A-Z]{{2,4}}{_DASH}\s*\d{{4}}{_DASH}\s*[A-Z]{_DASH}\s*\d{{3,5}}\b"
)

# ClinicalTrials.gov registry IDs. Structurally unable to match DOCKET_TOKEN_RE
# (no separators at all), so without their own pattern every protocol scored zero
# identifier tokens and the combined metric was misleading.
REGISTRY_TOKEN_RE = re.compile(r"\bNCT\s*\d{8}\b")

# Proxy for tabular layout: a line holding three or more cells separated by runs
# of whitespace. Crude, but it ranks documents by table-heaviness well enough to
# choose which ones to hand-inspect, which is all Phase 0 needs. Not a table count.
TABLE_LINE_RE = re.compile(r"\S+(?:\s{2,}\S+){2,}")


class PdfSummary(dict):
    """Plain dict of inspection results; keys mirror FetchedDoc's PDF fields."""


def classify_pages(per_page_chars: list[int]) -> tuple[int, str]:
    """Classify a document from its per-page extracted-character counts.

    Split out from `inspect` so the threshold logic is testable without
    constructing PDFs — the thresholds are the part with actual behaviour, and
    they were previously only exercised by whatever happened to be in the corpus.

    Returns (blank_page_count, verdict).
    """
    if not per_page_chars:
        return 0, "unreadable"
    blank = sum(1 for c in per_page_chars if c < MIN_CHARS_PER_TEXT_PAGE)
    ratio = blank / len(per_page_chars)
    if ratio >= IMAGE_ONLY_THRESHOLD:
        return blank, "image_only"
    if ratio >= MIXED_THRESHOLD:
        return blank, "mixed"
    return blank, "digital_native"


def inspect(path: Path) -> PdfSummary:
    """Extract text page by page and summarise the document's shape.

    Deliberately catches broad exceptions: pypdf's documented base class does not
    cover everything a malformed PDF can raise (its own docstring warns of
    RecursionError, KeyError, struct.error, and DependencyError/DeprecationError
    are direct Exception subclasses). One unparseable document must degrade to a
    verdict, never abort the run.
    """
    try:
        reader = PdfReader(str(path))
        # No explicit decrypt() call: PdfReader already attempts the empty
        # password, and a wrong-password decrypt() returns a status rather than
        # raising, so an explicit attempt here would be dead code. A genuinely
        # locked file raises from page access below and is caught.
        pages = list(reader.pages)
    except Exception as exc:  # noqa: BLE001 - see docstring
        return _unreadable(f"{type(exc).__name__}: {exc}")

    if not pages:
        return _unreadable("zero pages")

    per_page_chars: list[int] = []
    table_lines = 0
    docket_tokens = 0
    registry_tokens = 0

    for page in pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - a single bad page shouldn't fail the doc
            text = ""
        stripped = text.strip()
        per_page_chars.append(len(stripped))
        if stripped:
            table_lines += sum(1 for line in stripped.splitlines() if TABLE_LINE_RE.search(line))
            docket_tokens += len(DOCKET_TOKEN_RE.findall(stripped))
            registry_tokens += len(REGISTRY_TOKEN_RE.findall(stripped))

    blank, verdict = classify_pages(per_page_chars)

    return PdfSummary(
        pages=len(pages),
        text_chars=sum(per_page_chars),
        image_only_pages=blank,
        text_layer=verdict,
        table_like_lines=table_lines,
        docket_like_tokens=docket_tokens,
        registry_like_tokens=registry_tokens,
    )


def _unreadable(reason: str) -> PdfSummary:
    return PdfSummary(
        pages=None,
        text_chars=None,
        image_only_pages=None,
        text_layer="unreadable",
        table_like_lines=None,
        docket_like_tokens=None,
        registry_like_tokens=None,
        unreadable_reason=reason,
    )


def is_indexable(text_layer: str | None) -> bool:
    """Whether a document with this verdict should enter the index.

    `mixed` passes: a protocol with a scanned appendix is still mostly useful
    text, and dropping it would bias the corpus toward tidy documents.
    """
    return text_layer in INDEXABLE_VERDICTS
