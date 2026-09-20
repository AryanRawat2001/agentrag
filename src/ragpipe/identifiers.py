"""Exact identifiers found in document text.

These exist for one purpose: the exact-identifier eval slice, which is the
cleanest demonstration that lexical retrieval beats dense retrieval. An embedding
of "21 CFR 314.50" carries almost no signal about *which* regulation it is —
neighbouring section numbers embed nearly identically — while BM25 matches it
exactly. If hybrid search is worth building, this is the slice that proves it.

**Which identifier class carries the slice was measured, not assumed.** Corpus-wide
counts are in `reports/extraction_report.md`; CFR citations win on both coverage
and distinct values, and Phase 0's assumption that dockets would carry it was
wrong — dockets appear in metadata but almost never in body text.

## Two granularities, deliberately

`canonical` is **section level** (`21 CFR 117.136`) and `canonical_detailed`
carries the subsection (`21 CFR 117.136(a)(2)(i)`). Both are recorded because they
answer different questions and a single field cannot:

  * Section level is the right *retrieval* unit — the chunk discussing a
    regulation is the retrieval target regardless of which subsection a query
    cites — and it is the only level at which classes are comparable.
  * Subsection level is the honest count of *distinct citations*, since
    `117.136(a)(2)` and `117.136(a)(2)(i)` genuinely differ.

An earlier version captured subsections for U.S.C. but not CFR, so the two
classes' distinct counts were measured at different resolutions while sitting in
one column. That is why both are now explicit.

## Line-number contamination

Line-numbered FDA draft guidances interleave the marginal line number with the
citation during PDF extraction, so `21 CFR 812.2(c)` printed on line 71 extracts
as `21 CFR 71 \\n812.2(c)`. A naive pattern captures `21 CFR 71` and loses the
real citation. Two principled repairs, not symptom matching:

  1. CFR has exactly 50 titles and the U.S. Code has 54. A higher title is
     impossible — this alone caught `73 CFR 1271.3`.
  2. When the captured number ends a line and a section-shaped number starts the
     next, the captured one is the line number and the real section follows.

Offsets are recorded alongside each match because Phase 5's citation verifier
needs to assert that a cited span exists in the source text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# PDF extraction breaks identifiers across lines, and FDA's Federal Register
# typography uses en-dashes, so patterns tolerate Unicode dashes and internal
# whitespace.
_DASH = r"[-‐‑‒–—]"
_WS = r"[ \t\r\n ]*"

# `§{0,2}` and `[Pp]arts?` are load-bearing. An earlier `§?` plus singular `Part`
# meant `21 CFR §§ 211.42(d)` and `21 CFR parts 210 and 211` matched *nothing at
# all* — because the optional group failed and took the following digits with it —
# so two documents reported zero CFR citations while containing several.
_SECT = r"§{0,2}"
_PART = rf"(?:[Pp]arts?{_WS})?"

# Optional trailing subsection chain, e.g. "(a)(2)(i)".
_SUBSEC = r"(?:\([A-Za-z0-9]{1,4}\))*"

# There are 50 CFR titles and 54 U.S. Code titles.
MAX_TITLE = {"cfr": 50, "usc": 54}

# A captured section number followed by a line break and a section-shaped number
# means the captured value was a marginal line number.
_LINE_NUMBER_TAIL = re.compile(r"\A\s*\n\s*(\d+\.\d+)")


@dataclass(frozen=True, slots=True)
class Identifier:
    """One identifier occurrence, with its span in the document text."""

    kind: str
    raw: str  # exactly as it appears, including any line break
    canonical: str  # section level — comparable across classes, the retrieval unit
    canonical_detailed: str  # includes subsection, when present
    start: int
    end: int
    repaired: bool = False  # a line-number artifact was corrected


PATTERNS: dict[str, re.Pattern[str]] = {
    "cfr": re.compile(
        rf"\b\d{{1,2}}{_WS}C\.?{_WS}F\.?{_WS}R\.?{_WS}{_PART}{_SECT}{_WS}"
        rf"\d+(?:\.\d+)?{_SUBSEC}",
        re.I,
    ),
    "usc": re.compile(
        rf"\b\d{{1,2}}{_WS}U\.?{_WS}S\.?{_WS}C\.?{_WS}{_SECT}{_WS}\d+[a-z]?{_SUBSEC}",
        re.I,
    ),
    "fed_register": re.compile(rf"\b\d{{2,3}}{_WS}F\.?{_WS}R\.?{_WS}\d{{3,6}}\b"),
    "docket": re.compile(
        rf"\b[A-Z]{{2,4}}{_DASH}{_WS}\d{{4}}{_DASH}{_WS}[A-Z]{_DASH}{_WS}\d{{3,5}}\b"
    ),
    "registry": re.compile(rf"\bNCT{_WS}\d{{8}}\b"),
    "ich": re.compile(rf"\bICH{_WS}[EQSM]\d{{1,2}}[A-Z]?(?:{_WS}\(R\d\))?", re.I),
}

# The class the exact-identifier eval slice is built on. Named here so the eval
# harness and the report agree rather than each hard-coding a choice.
PRIMARY_KIND = "cfr"

# Structural parsers used for canonicalisation. Rebuilding from parsed parts is
# what makes "21CFR 312.32", "21 C.F.R. 312.32", "21 CFR Part 312.32" and
# "21 CFR\n312.32" collapse to one value; collapsing whitespace alone does not.
_CFR_PARTS = re.compile(
    rf"(\d{{1,2}}){_WS}C\.?{_WS}F\.?{_WS}R\.?{_WS}{_PART}{_SECT}{_WS}"
    rf"(\d+(?:\.\d+)?)({_SUBSEC})",
    re.I,
)
_USC_PARTS = re.compile(
    rf"(\d{{1,2}}){_WS}U\.?{_WS}S\.?{_WS}C\.?{_WS}{_SECT}{_WS}(\d+[a-z]?)({_SUBSEC})", re.I
)
_FR_PARTS = re.compile(rf"(\d{{2,3}}){_WS}F\.?{_WS}R\.?{_WS}(\d{{3,6}})")
_REGISTRY_PARTS = re.compile(rf"NCT{_WS}(\d{{8}})", re.I)
_ICH_PARTS = re.compile(rf"ICH{_WS}([EQSM]\d{{1,2}}[A-Z]?)(?:{_WS}\((R\d)\))?", re.I)


def _flatten(raw: str) -> str:
    return re.sub(r"\s+", " ", re.sub(_DASH, "-", raw)).strip()


def _canonical_titled(kind: str, raw: str) -> tuple[str, str, int, str] | None:
    """Parse a title/section/subsection identifier. Returns
    (section_level, detailed, title, section) or None if unparseable."""
    pattern = _CFR_PARTS if kind == "cfr" else _USC_PARTS
    m = pattern.search(_flatten(raw))
    if not m:
        return None
    title, section, subsec = int(m.group(1)), m.group(2), m.group(3) or ""
    label = "CFR" if kind == "cfr" else "USC"
    # Subsection case is preserved: (C) and (c) are different subdivisions.
    base = f"{title} {label} {section.lower() if kind == 'usc' else section}"
    return base, base + subsec, title, section


def canonical(kind: str, raw: str) -> str:
    """Section-level canonical form. See the module docstring on granularity."""
    result = canonical_pair(kind, raw)
    return result[0] if result else _flatten(raw).upper()


def canonical_pair(kind: str, raw: str) -> tuple[str, str] | None:
    """Return (section_level, detailed) canonical forms, or None if unparseable."""
    if kind in MAX_TITLE:
        parsed = _canonical_titled(kind, raw)
        if not parsed:
            return None
        base, detailed, title, _ = parsed
        if title > MAX_TITLE[kind] or title < 1:
            return None  # impossible title — a contaminated match
        return base, detailed

    flat = _flatten(raw)
    if kind == "fed_register" and (m := _FR_PARTS.search(flat)):
        v = f"{m.group(1)} FR {m.group(2)}"
        return v, v
    if kind == "registry" and (m := _REGISTRY_PARTS.search(flat)):
        v = f"NCT{m.group(1)}"
        return v, v
    if kind == "ich" and (m := _ICH_PARTS.search(flat)):
        code = m.group(1).upper()
        v = f"ICH {code}({m.group(2).upper()})" if m.group(2) else f"ICH {code}"
        return v, v
    if kind == "docket":
        v = re.sub(r"\s*-\s*", "-", flat).upper()
        return v, v
    return flat.upper(), flat.upper()


def extract(text: str, kinds: list[str] | None = None) -> list[Identifier]:
    """Find every identifier occurrence in `text`, sorted by position.

    Matches whose title is impossible are dropped, and line-number contamination
    is repaired using the text following the match.
    """
    wanted = kinds or list(PATTERNS)
    out: list[Identifier] = []

    for kind in wanted:
        pattern = PATTERNS.get(kind)
        if pattern is None:
            continue
        for m in pattern.finditer(text):
            raw = m.group(0)
            repaired = False

            if kind in MAX_TITLE:
                tail = _LINE_NUMBER_TAIL.match(text[m.end() : m.end() + 24])
                parsed = _canonical_titled(kind, raw)
                # Repair only when the captured section has no decimal point and
                # the next line starts with one that does: that asymmetry is the
                # signature of a marginal line number, not of a real citation.
                if tail and parsed and "." not in parsed[3]:
                    label = "CFR" if kind == "cfr" else "USC"
                    title = parsed[2]
                    if 1 <= title <= MAX_TITLE[kind]:
                        base = f"{title} {label} {tail.group(1)}"
                        out.append(Identifier(kind, raw, base, base, m.start(), m.end(), True))
                        continue

            pair = canonical_pair(kind, raw)
            if pair is None:
                continue  # impossible title, or unparseable
            out.append(Identifier(kind, raw, pair[0], pair[1], m.start(), m.end(), repaired))

    out.sort(key=lambda i: (i.start, i.kind))
    return out


def density(found: list[Identifier]) -> dict[str, int]:
    """Occurrence count per identifier class."""
    counts = dict.fromkeys(PATTERNS, 0)
    for ident in found:
        counts[ident.kind] += 1
    return counts


def distinct(found: list[Identifier], kind: str | None = None, detailed: bool = False) -> set[str]:
    """Distinct canonical identifiers, optionally filtered to one class.

    The eval slice needs *distinct* identifiers — thousands of occurrences of a
    handful of citations would not make thousands of usable questions. Pass
    detailed=True for subsection-level granularity.
    """
    return {
        (i.canonical_detailed if detailed else i.canonical)
        for i in found
        if kind is None or i.kind == kind
    }
