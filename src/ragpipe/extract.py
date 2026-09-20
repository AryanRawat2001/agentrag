"""PDF -> normalised text with page offsets, section structure, and identifiers.

Output is one JSON file per document under `data/extracted/`, plus a summary
index. Downstream phases read that instead of re-parsing PDFs, which matters
because pypdf extraction over this corpus takes minutes and Phase 1's chunking
sweep needs to run repeatedly.

## Structure detection

Two sources, because neither covers the corpus alone. Corpus-wide counts are in
`reports/extraction_report.md`; a substantial minority of PDFs carry no usable
outline at all, including protocols such as `ctgov-NCT03518125`, which relies
entirely on heuristic detection of its numbered headings.

  1. **Outline** (preferred). Authored by whoever made the PDF, so headings and
     nesting levels are real rather than inferred. Each bookmark is located at its
     actual position within its page — anchoring to the page top instead left 54%
     of outline sections empty and attributed each page's text to the last heading
     on it.
  2. **Heuristic** (fallback). Line patterns: decimal numbering (`4.2.1 Study
     Design`), roman numerals, single letters, and short ALL-CAPS lines.

Heuristic detection is genuinely imprecise and this module does not pretend
otherwise: the extraction report publishes per-document heading counts and flags
documents with none, so outliers can be hand-checked. Whether structure-aware
chunking actually beats fixed-size chunking on this corpus is an empirical
question that Phase 3's ablation answers — and if the detector is weak, that is
the number which will expose it.

## Incrementality

A document is re-extracted only when its manifest sha256 differs from the hash
recorded in its extraction output. The Phase 0 content pins therefore do double
duty as the change-detection key, exactly as the plan intended.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from ragpipe import identifiers as ident_mod
from ragpipe import pdfcheck
from ragpipe.models import SourceDoc

# --- heading heuristics ------------------------------------------------------
# Deliberately conservative: a false heading fragments a section and pollutes
# retrieval, which is harder to notice than a missed heading.

_MAX_HEADING_WORDS = 14
_MAX_HEADING_CHARS = 110

# A heading reappearing within this many pages is page furniture, not a section.
_RUNNING_HEADER_PAGE_WINDOW = 2

_NUMBERED = re.compile(r"^\s{0,8}(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(\S.*)$")
_ROMAN = re.compile(r"^\s{0,8}((?:IX|IV|VI{0,3}|I{1,3}|X{1,2}))\.\s+(\S.*)$")
_LETTERED = re.compile(r"^\s{0,8}([A-Z])\.\s+(\S.*)$")
_ALLCAPS = re.compile(r"^\s{0,8}([A-Z][A-Z0-9\s&,\-/():.']{4,70})\s*$")

# Table-of-contents lines ("4.2 Study Design ......... 17") produce a heading for
# every real heading in the document, doubling the section count and pointing
# every duplicate at the TOC page.
#
# Two patterns, deliberately separated. The dot-leader form is unambiguous and is
# what identifies a *page* as a ToC. The trailing-number form ("...text  17") also
# appears on ordinary body pages of line-numbered guidances and in data tables, so
# using it for page-level classification discarded real content: on fda-119789 it
# flagged 11 of 25 pages and dropped 13 headings — about half that document's
# structure. It is now used only to reject individual lines, which costs one
# heading corpus-wide.
_TOC_DOT_LEADER = re.compile(r"\.{3,}\s*\d+\s*$")
_TOC_TRAILING_NUM = re.compile(r"\s{2,}\d{1,3}\s*$")
_TOC_PAGE_MIN_HITS = 4

# A trailing period usually means a sentence, not a heading.
_SENTENCE_END = re.compile(r"[.;:,]\s*$")

# Lines opening with a legal citation are references, not headings.
_CITATION_START = re.compile(
    r"^\d{1,2}\s*(?:C\.?\s*F\.?\s*R\.?|U\.?\s*S\.?\s*C\.?|F\.?\s*R\.?)\b", re.I
)

# --- outline validation ------------------------------------------------------
# Some PDFs expose an accessibility *structure tree* through the same API as
# bookmarks. Every text run becomes an "outline entry", including single
# characters and empty strings, and they all resolve to page 1. Measured on this
# corpus: fda-71536 yielded 1,618 entries for 25 pages with a median heading
# length of 3 characters and 1,593 zero-length sections.
#
# So an outline's existence is not evidence that it is a table of contents. These
# bounds separate authored navigation from a serialised tag tree; failing any of
# them sends the document to heuristic detection instead.
_OUTLINE_MAX_ENTRIES_PER_PAGE = 8.0
_OUTLINE_MIN_SUBSTANTIVE_RATIO = 0.6
_OUTLINE_MAX_SINGLE_PAGE_RATIO = 0.5
_OUTLINE_MIN_HEADING_WORD_CHARS = 4


def outline_is_usable(outline: list[tuple[str, int, int]], n_pages: int) -> tuple[bool, str]:
    """Decide whether a PDF outline is a real table of contents.

    Returns (usable, reason). The reason is recorded in the extraction output so
    a rejected outline is visible rather than silently downgraded.
    """
    if len(outline) < 2:
        return False, "fewer than 2 entries"

    if n_pages and len(outline) / n_pages > _OUTLINE_MAX_ENTRIES_PER_PAGE:
        return False, (
            f"{len(outline) / n_pages:.1f} entries per page exceeds "
            f"{_OUTLINE_MAX_ENTRIES_PER_PAGE} — looks like a tag tree, not a ToC"
        )

    substantive = sum(
        1
        for title, _, _ in outline
        if len(re.sub(r"[^\w]", "", title)) >= _OUTLINE_MIN_HEADING_WORD_CHARS
    )
    ratio = substantive / len(outline)
    if ratio < _OUTLINE_MIN_SUBSTANTIVE_RATIO:
        return False, f"only {ratio:.0%} of entries have >=4 word characters"

    page_counts = Counter(page for _, _, page in outline)
    top_share = page_counts.most_common(1)[0][1] / len(outline)
    if top_share > _OUTLINE_MAX_SINGLE_PAGE_RATIO:
        return False, f"{top_share:.0%} of entries resolve to a single page"

    return True, "ok"


@dataclass(slots=True)
class PageSpan:
    page: int  # 1-based
    start: int  # char offset into the document text
    end: int
    chars: int


@dataclass(slots=True)
class Section:
    idx: int
    heading: str
    level: int
    page: int
    start: int  # char offset into the document text
    end: int


@dataclass(slots=True)
class ExtractedDoc:
    doc_id: str
    source: str
    sha256: str | None  # provenance: ties this output to a pinned input
    title: str
    text: str
    n_pages: int
    n_chars: int
    structure_source: str  # "outline" | "heuristic" | "none"
    outline_entries: int = 0  # raw bookmark count, before validation
    outline_note: str = ""  # why the outline was used or rejected
    # Text-layer *quality*, as opposed to presence. `pdfcheck` verifies a page has
    # extractable characters; this records whether those characters form words. A
    # glyph-positioned PDF yields a full text layer of unusable single characters,
    # which passed every emptiness check and sat in the corpus for four phases.
    single_char_token_share: float = 0.0
    # "ok" | "character_spaced" | "broken_encoding" | "space_collapsed".
    # The last two were added in Phase 5 and were initially computed nowhere — the
    # detectors existed and nothing called them, so `QUARANTINE_QUALITIES` described a
    # safeguard that did not run. Caught by the code-review gate.
    text_quality: str = "ok"
    control_char_share: float = 0.0
    runon_token_share: float = 0.0
    pages: list[PageSpan] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    identifier_counts: dict[str, int] = field(default_factory=dict)
    # Section level (comparable across classes) and subsection level (the honest
    # count of distinct citations). See identifiers.py on why both are kept.
    distinct_identifiers: dict[str, list[str]] = field(default_factory=dict)
    distinct_identifiers_detailed: dict[str, list[str]] = field(default_factory=dict)
    repaired_identifiers: int = 0
    identifiers: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Index row: everything except the text and the per-occurrence spans."""
        d = asdict(self)
        d.pop("text")
        d.pop("identifiers")
        d["n_sections"] = len(self.sections)
        d["pages"] = len(self.pages)
        return d


def _flatten_outline(reader: PdfReader) -> list[tuple[str, int, int]]:
    """Flatten PDF bookmarks into (title, level, page_index_0based).

    pypdf represents nesting as nested lists, and destination resolution can fail
    per-entry on malformed files, so each entry is resolved defensively.
    """
    out: list[tuple[str, int, int]] = []

    def walk(items: Any, level: int) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, level + 1)
                continue
            title = getattr(item, "title", None)
            if not title:
                continue
            try:
                page_no = reader.get_destination_page_number(item)
            except Exception:  # noqa: BLE001 - one bad bookmark shouldn't lose the rest
                continue
            if page_no is not None:
                out.append((str(title).strip(), level, int(page_no)))

    try:
        walk(reader.outline or [], 0)
    except Exception:  # noqa: BLE001 - malformed outline tree
        return []
    return out


def _looks_like_toc(page_text: str) -> bool:
    """Whether a page is a table of contents. Dot leaders only — see _TOC_*."""
    hits = sum(1 for line in page_text.splitlines() if _TOC_DOT_LEADER.search(line))
    return hits >= _TOC_PAGE_MIN_HITS


def _heading_from_line(line: str) -> tuple[str, int] | None:
    """Return (heading_text, level) if this line looks like a heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_CHARS:
        return None
    if _TOC_DOT_LEADER.search(line) or _TOC_TRAILING_NUM.search(line):
        return None

    # A legal citation at the start of a line matches the numbered-heading shape
    # ("21 CFR 820.30, Subpart C — Design Controls...") but is a reference, not a
    # heading. Rejecting these removes the most common false positive.
    if _CITATION_START.match(stripped):
        return None

    if m := _NUMBERED.match(line):
        number, rest = m.group(1), m.group(2).strip()
        if len(rest.split()) > _MAX_HEADING_WORDS or _SENTENCE_END.search(rest):
            return None
        # Headings start with a word, and with a capital. Requiring uppercase
        # filters numbered list items that continue prose.
        if not rest[:1].isalpha() or not rest[:1].isupper():
            return None
        return f"{number} {rest}", number.count(".") + 1

    if m := _ROMAN.match(line):
        rest = m.group(2).strip()
        if len(rest.split()) > _MAX_HEADING_WORDS or _SENTENCE_END.search(rest):
            return None
        return f"{m.group(1)}. {rest}", 1

    if m := _LETTERED.match(line):
        rest = m.group(2).strip()
        if len(rest.split()) > _MAX_HEADING_WORDS or _SENTENCE_END.search(rest):
            return None
        return f"{m.group(1)}. {rest}", 2

    if m := _ALLCAPS.match(line):
        candidate = m.group(1).strip()
        # Require at least two words and one vowel — filters page furniture,
        # figure labels, and stray capitalised fragments.
        if len(candidate.split()) < 2 or not re.search(r"[AEIOU]", candidate):
            return None
        return candidate, 1

    return None


def _locate_in_page(title: str, page_text: str, search_from: int) -> int | None:
    """Character offset of `title` within `page_text`, or None.

    Bookmark titles rarely match the extracted text byte for byte — line wrapping
    and collapsed whitespace differ — so this tries the whole title, then a
    prefix, then a whitespace-insensitive regex.
    """
    probe = title.strip()
    if not probe:
        return None

    for candidate in (probe, probe[:40], probe[:20]):
        if len(candidate) < 4:
            break
        pos = page_text.find(candidate, search_from)
        if pos >= 0:
            return pos

    head = probe[:30]
    if len(head) >= 6:
        loose = re.compile(r"\s*".join(re.escape(ch) for ch in head if not ch.isspace()))
        if m := loose.search(page_text, search_from):
            return m.start()
    return None


def _sections_from_outline(
    outline: list[tuple[str, int, int]],
    pages: list[PageSpan],
    page_texts: list[str],
    text_len: int,
) -> list[Section]:
    """Build sections from bookmarks, located at the heading's real position.

    Anchoring every bookmark to the top of its page — the obvious implementation —
    made 54% of outline sections zero-length, because `_close_sections` sets each
    end to the next start and bookmarks routinely share a page. Worse than empty:
    the whole page's text was attributed to the *last* heading on it, so a chunk
    labelled "C. Remote Regulatory Assessment" actually contained the text of
    "1. Objective". Locating the title inside the page fixes both.
    """
    # Bookmark-tree order is not guaranteed to be page order, and an out-of-order
    # entry would produce a negative-length span that silently slices to "".
    ordered = sorted(
        ((t, lv, pi) for t, lv, pi in outline if 0 <= pi < len(pages)),
        key=lambda e: e[2],
    )

    sections: list[Section] = []
    cursor_by_page: dict[int, int] = {}
    for title, level, page_idx in ordered:
        page_text = page_texts[page_idx]
        search_from = cursor_by_page.get(page_idx, 0)
        offset = _locate_in_page(title, page_text, search_from)
        if offset is None:
            # Unlocatable title: fall back to page start, but never before the
            # previous section on this page.
            offset = search_from
        cursor_by_page[page_idx] = offset + 1

        start = pages[page_idx].start + offset
        sections.append(
            Section(
                idx=len(sections),
                heading=title,
                level=level,
                page=page_idx + 1,
                start=start,
                end=start,
            )
        )

    # Monotonic starts are required for _close_sections to produce sane spans.
    sections.sort(key=lambda s: s.start)
    return _close_sections(sections, text_len)


def _sections_from_text(
    page_texts: list[str], pages: list[PageSpan], text_len: int
) -> list[Section]:
    sections: list[Section] = []
    # Running-header suppression, keyed on (heading, page). An earlier version
    # kept only the last three accepted headings with no page component, so a
    # page-footer date line survived as a "heading" 26 separate times in one
    # protocol ("11 May 2015" parsed as number 11 + text "May 2015"). Tracking the
    # last page a heading was seen on is what the comment always claimed.
    last_page_seen: dict[str, int] = {}
    for p_i, page_text in enumerate(page_texts):
        if _looks_like_toc(page_text):
            continue
        offset = pages[p_i].start
        cursor = 0
        for line in page_text.splitlines():
            line_start = page_text.find(line, cursor)
            if line_start < 0:
                line_start = cursor
            cursor = line_start + len(line)
            found = _heading_from_line(line)
            if not found:
                continue
            heading, level = found
            page_no = p_i + 1
            previous = last_page_seen.get(heading)
            if previous is not None and page_no - previous <= _RUNNING_HEADER_PAGE_WINDOW:
                last_page_seen[heading] = page_no
                continue
            last_page_seen[heading] = page_no
            sections.append(
                Section(
                    idx=len(sections),
                    heading=heading,
                    level=level,
                    page=p_i + 1,
                    start=offset + line_start,
                    end=offset + line_start,
                )
            )
    return _close_sections(sections, text_len)


def _close_sections(sections: list[Section], text_len: int) -> list[Section]:
    """Set each section's end to the next section's start."""
    for i, sec in enumerate(sections):
        sec.end = sections[i + 1].start if i + 1 < len(sections) else text_len
        sec.idx = i
    return sections


def extract_document(doc: SourceDoc, pdf_path: Path) -> ExtractedDoc:
    """Extract text, structure, and identifiers from one PDF."""
    reader = PdfReader(str(pdf_path))

    page_texts: list[str] = []
    for page in reader.pages:
        try:
            page_texts.append((page.extract_text() or "").strip())
        except Exception:  # noqa: BLE001 - a bad page yields nothing, not a failure
            page_texts.append("")

    # Single text blob with a page offset map, rather than text duplicated per
    # page: chunking slices by character offset, and citations need spans.
    parts: list[str] = []
    pages: list[PageSpan] = []
    cursor = 0
    for i, page_text in enumerate(page_texts):
        parts.append(page_text)
        pages.append(
            PageSpan(page=i + 1, start=cursor, end=cursor + len(page_text), chars=len(page_text))
        )
        cursor += len(page_text) + 2  # the "\n\n" join below
    text = "\n\n".join(parts)

    outline = _flatten_outline(reader)
    usable, outline_note = (
        outline_is_usable(outline, len(page_texts))
        if outline
        else (
            False,
            "no outline",
        )
    )
    sections: list[Section] = []
    structure_source = "none"
    if usable:
        sections = _sections_from_outline(outline, pages, page_texts, len(text))
        structure_source = "outline"
    if not sections:
        # A usable-looking outline that yields nothing in range must still fall
        # through to the heuristic detector rather than reporting "outline" with
        # zero sections.
        sections = _sections_from_text(page_texts, pages, len(text))
        structure_source = "heuristic" if sections else "none"
        if usable and not sections:
            outline_note = f"{outline_note}; outline yielded no in-range sections"

    found = ident_mod.extract(text)

    # Text-layer quality, recorded rather than acted on here: extraction's job is to
    # report what it got, and the quarantine decision belongs to the consumer.
    spacing = pdfcheck.single_char_token_share(text)

    return ExtractedDoc(
        doc_id=doc.doc_id,
        source=doc.source,
        sha256=doc.sha256,
        title=doc.title,
        text=text,
        n_pages=len(page_texts),
        n_chars=len(text),
        structure_source=structure_source,
        outline_entries=len(outline),
        outline_note=outline_note,
        single_char_token_share=round(spacing, 4),
        control_char_share=round(pdfcheck.control_char_share(text), 6),
        runon_token_share=round(pdfcheck.runon_token_share(text), 6),
        # `text_quality_label` rather than an inline character-spacing test, so all
        # four classes are actually assigned. Computed on the whole document text,
        # which is the basis the thresholds are calibrated against.
        text_quality=pdfcheck.text_quality_label(text),
        pages=pages,
        sections=sections,
        identifier_counts=ident_mod.density(found),
        distinct_identifiers={
            kind: sorted(ident_mod.distinct(found, kind))
            for kind in ident_mod.PATTERNS
            if any(i.kind == kind for i in found)
        },
        distinct_identifiers_detailed={
            kind: sorted(ident_mod.distinct(found, kind, detailed=True))
            for kind in ident_mod.PATTERNS
            if any(i.kind == kind for i in found)
        },
        repaired_identifiers=sum(1 for i in found if i.repaired),
        identifiers=[asdict(i) for i in found],
        # Carried forward so retrieval can filter on metadata without re-reading
        # the manifest: status drives the version slice, center the ACL demo.
        metadata={
            "doc_type": doc.doc_type,
            "status": doc.status,
            "center": doc.center,
            "offices": doc.offices,
            "docket": doc.docket,
            "issue_date": doc.issue_date,
            "product_areas": doc.product_areas,
            "topics": doc.topics,
            "landing_url": doc.landing_url,
            "url": doc.url,
            **{k: v for k, v in doc.extra.items() if k != "sample_reason"},
        },
    )


def output_path(root: Path, doc: SourceDoc) -> Path:
    return root / doc.source / f"{doc.doc_id}.json"


def needs_extraction(path: Path, pinned_sha256: str | None) -> bool:
    """True when there is no output, or the output came from different content.

    Reuses the Phase 0 content pins as the change-detection key.
    """
    if not path.exists():
        return True
    if pinned_sha256 is None:
        return True
    try:
        with path.open(encoding="utf-8") as fh:
            recorded = json.load(fh).get("sha256")
    except (OSError, ValueError):
        # ValueError covers both JSONDecodeError and UnicodeDecodeError. The
        # latter matters: output is written with ensure_ascii=False and the corpus
        # is full of en-dashes, so an interrupted write can truncate mid-character.
        # UnicodeDecodeError is not a JSONDecodeError, so catching only that
        # aborted the entire run instead of re-extracting one document.
        return True
    return recorded != pinned_sha256


def write_document(extracted: ExtractedDoc, path: Path) -> None:
    """Write atomically, so an interruption cannot leave a half-written output
    that a later run would either trust or crash on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(extracted), ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
