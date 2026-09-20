"""Phase 1 extraction checkpoint report.

Answers the questions Phase 2 and 3 depend on:

  structure coverage   -> can structure-aware chunking even be attempted, and on
                          how much of the corpus?
  section counts       -> is heading detection plausible, or obviously broken?
  identifier density   -> which identifier class can carry the exact-identifier
                          eval slice, measured on the full corpus rather than a
                          30-document probe
  distinct identifiers -> how many *questions* the slice can actually support
                          (457 occurrences of five citations is five questions)

Population discipline carried over from Phase 0: every figure states the set it
covers. That report's only real defect was computing numbers over one population
and printing them beside prose implying another.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragpipe import identifiers as ident_mod

# Heading detection is heuristic. These bounds do not gate anything — they flag
# documents for hand inspection, because a detector producing one heading per
# paragraph is as broken as one producing none, and both look fine in an average.
SECTIONS_PER_1K_CHARS_HIGH = 2.0
MIN_CHARS_PER_PAGE_OK = 100


def _dist(values: list[float], population: str) -> dict[str, Any]:
    if not values:
        return {"n": 0, "population": population}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "population": population,
        "min": round(ordered[0], 2),
        "median": round(statistics.median(ordered), 2),
        "p90": round(ordered[min(n - 1, max(0, math.ceil(0.9 * n) - 1))], 2),
        "max": round(ordered[-1], 2),
        "total": round(sum(ordered), 2),
    }


def _table(rows: list[tuple[Any, ...]], headers: tuple[str, ...]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        *["| " + " | ".join(str(c) for c in row) + " |" for row in rows],
    ]


def build(summaries: list[dict[str, Any]], failures: list[dict[str, str]]) -> dict[str, Any]:
    """Compute the payload from extraction index rows (no document text needed)."""
    n = len(summaries)

    by_structure = Counter(s["structure_source"] for s in summaries)
    sections = [s["n_sections"] for s in summaries]
    chars = [s["n_chars"] for s in summaries]

    # Identifier aggregation. `distinct` is the number that matters for the eval
    # slice: occurrences tell you how often a citation is mentioned, distinct
    # values tell you how many questions you can write.
    occurrences: Counter[str] = Counter()
    docs_with: Counter[str] = Counter()
    distinct_global: dict[str, set[str]] = {k: set() for k in ident_mod.PATTERNS}
    distinct_detailed: dict[str, set[str]] = {k: set() for k in ident_mod.PATTERNS}
    # Per-source split. A corpus-wide coverage percentage hides that CFR citations
    # are overwhelmingly an FDA phenomenon and registry IDs exclusively a protocol
    # one, which materially changes what the eval slice actually covers.
    by_source_occ: dict[str, Counter[str]] = {}
    by_source_docs: dict[str, Counter[str]] = {}
    source_totals: Counter[str] = Counter()
    repaired = 0

    for s in summaries:
        src = s["source"]
        source_totals[src] += 1
        by_source_occ.setdefault(src, Counter())
        by_source_docs.setdefault(src, Counter())
        repaired += s.get("repaired_identifiers") or 0
        for kind, count in (s.get("identifier_counts") or {}).items():
            occurrences[kind] += count
            by_source_occ[src][kind] += count
            if count:
                docs_with[kind] += 1
                by_source_docs[src][kind] += 1
        for kind, values in (s.get("distinct_identifiers") or {}).items():
            distinct_global[kind].update(values)
        for kind, values in (s.get("distinct_identifiers_detailed") or {}).items():
            distinct_detailed[kind].update(values)

    # Heading-detection outliers, both directions.
    no_sections = [s["doc_id"] for s in summaries if s["n_sections"] == 0]
    dense = sorted(
        (
            {
                "doc_id": s["doc_id"],
                "n_sections": s["n_sections"],
                "n_chars": s["n_chars"],
                "per_1k": round(1000 * s["n_sections"] / max(s["n_chars"], 1), 2),
                "structure_source": s["structure_source"],
            }
            for s in summaries
            if s["n_chars"] > 0
            and 1000 * s["n_sections"] / s["n_chars"] > SECTIONS_PER_1K_CHARS_HIGH
        ),
        key=lambda d: -d["per_1k"],
    )

    thin = [
        {"doc_id": s["doc_id"], "chars_per_page": round(s["n_chars"] / max(s["pages"], 1))}
        for s in summaries
        if s["pages"] and s["n_chars"] / s["pages"] < MIN_CHARS_PER_PAGE_OK
    ]

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "counts": {
            "extracted": n,
            "failed": len(failures),
            "with_any_section": sum(1 for s in summaries if s["n_sections"] > 0),
        },
        "by_source": dict(Counter(s["source"] for s in summaries)),
        "structure_source": dict(by_structure),
        "sections": _dist([float(v) for v in sections], "extracted documents"),
        "chars": _dist([float(v) for v in chars], "extracted documents"),
        "chars_per_page": _dist(
            [s["n_chars"] / s["pages"] for s in summaries if s["pages"]], "extracted documents"
        ),
        "corpus_chars_per_page": (
            sum(s["n_chars"] for s in summaries) / max(sum(s["pages"] for s in summaries), 1)
        ),
        "identifiers": {
            kind: {
                "docs_containing": docs_with[kind],
                "occurrences": occurrences[kind],
                "distinct_section_level": len(distinct_global[kind]),
                "distinct_with_subsection": len(distinct_detailed[kind]),
                "by_source": {
                    src: {
                        "docs": by_source_docs[src][kind],
                        "of_docs": source_totals[src],
                        "occurrences": by_source_occ[src][kind],
                    }
                    for src in sorted(source_totals)
                },
            }
            for kind in ident_mod.PATTERNS
        },
        "repaired_identifier_occurrences": repaired,
        "primary_kind": ident_mod.PRIMARY_KIND,
        # Full lists are retained and truncated only at display time. Storing
        # `dense[:10]` here and then printing `len(...)` in the prose made the
        # reported count silently cap at 10 — the same defect already fixed once
        # for `docs_with_no_sections`.
        "flags": {
            "docs_with_no_sections": no_sections,
            "docs_with_suspiciously_dense_sections": dense,
            "docs_with_thin_text_per_page": thin,
        },
        "failures": failures,
    }


def render_markdown(s: dict[str, Any]) -> str:
    c = s["counts"]
    n = c["extracted"]
    primary = s["primary_kind"]
    ids = s["identifiers"]

    lines = [
        "# Extraction report — Phase 1",
        "",
        f"_Generated {s['generated_at']}_",
        "",
        "Normalised text, page offsets, section structure, and exact identifiers for",
        "every indexable document. Rebuild with `make extract`; a document is",
        "re-extracted only when its manifest sha256 changes, so the Phase 0 content",
        "pins double as the change-detection key.",
        "",
        "## Coverage",
        "",
        *_table(
            [
                ("Documents extracted", n),
                ("Extraction failed", c["failed"]),
                (
                    "With at least one section",
                    f"{c['with_any_section']} ({100 * c['with_any_section'] / n:.1f}%)"
                    if n
                    else "n/a",
                ),
            ],
            ("Metric", "Value"),
        ),
        "",
        "By source: " + ", ".join(f"`{k}` {v}" for k, v in s["by_source"].items()),
        "",
        "## Structure source",
        "",
        "Two detectors, because neither covers the corpus alone. `outline` means the",
        "PDF ships bookmarks — authored headings with real nesting. `heuristic` means",
        "they were inferred from line patterns, which is genuinely imprecise.",
        "",
        *_table(list(s["structure_source"].items()), ("Source", "Documents")),
        "",
        *_table(
            [
                ("Sections per document", _fmt(s["sections"])),
                ("Characters per document", _fmt(s["chars"])),
                # No total for this row: summing 155 ratios yields a number with no
                # referent, and printing it under a shared "(total)" header beside
                # two real corpus totals invited reading it as one.
                ("Characters per page (per-document rate)", _fmt(s["chars_per_page"], total=False)),
            ],
            ("Metric", "min / median / p90 / max (total)"),
        ),
        "",
        f"Corpus-wide characters per page is **{s['corpus_chars_per_page']:,.2f}** "
        f"(total characters / total pages). The row above is the distribution of "
        f"*per-document* rates, whose median sits below the corpus-wide figure "
        f"because short documents are denser.",
        "",
        "Character totals here run about 0.075% above the Phase 0 corpus report for "
        "the same documents. The difference is exactly two characters per page "
        "boundary — the `\\n\\n` separators added when pages are concatenated into "
        "one addressable blob. Phase 0 counted pages independently.",
        "",
        "## Exact-identifier density",
        "",
        "This table decides which identifier class the exact-identifier eval slice is",
        "built on — the slice that demonstrates lexical retrieval beating dense",
        "retrieval. Occurrences tell you how often a citation is mentioned; distinct",
        "values tell you how many questions can actually be written.",
        "",
        "**`distinct` is reported at two granularities**, because one number cannot serve",
        "both purposes. Section level is the retrieval unit and the only level at which",
        "classes are comparable; subsection level is the honest count of distinct",
        "citations. An earlier version captured subsections for `usc` but not `cfr`, so",
        "the two rows were measured at different resolutions inside a single column.",
        "",
        *_table(
            [
                (
                    f"`{kind}`" + (" **(primary)**" if kind == primary else ""),
                    v["docs_containing"],
                    f"{100 * v['docs_containing'] / n:.0f}%" if n else "n/a",
                    f"{v['occurrences']:,}",
                    f"{v['distinct_section_level']:,}",
                    f"{v['distinct_with_subsection']:,}",
                )
                for kind, v in sorted(ids.items(), key=lambda kv: -kv[1]["docs_containing"])
            ],
            (
                "Identifier",
                "Docs",
                "Coverage",
                "Occurrences",
                "Distinct (section)",
                "Distinct (+subsec)",
            ),
        ),
        "",
        "### Source split",
        "",
        "The corpus-wide coverage column hides an almost total source split, which matters",
        "because this table's job is choosing the slice. These classes are not spread",
        "across the corpus — each is a property of a document type.",
        "",
        *_table(
            [
                (
                    f"`{kind}`",
                    *[
                        f"{sv['docs']}/{sv['of_docs']} docs, {sv['occurrences']:,} occ."
                        for sv in v["by_source"].values()
                    ],
                )
                for kind, v in sorted(ids.items(), key=lambda kv: -kv[1]["occurrences"])
            ],
            ("Identifier", *[f"`{src}`" for src in next(iter(ids.values()))["by_source"]]),
        ),
        "",
        f"So the exact-identifier slice built on `{primary}` is effectively FDA-only. That",
        "is acceptable — it still exercises the retrieval behaviour the slice exists to",
        "test — but it is a property to state rather than hide behind a corpus-wide",
        "denominator.",
        "",
        "Phase 0 assumed `docket` would carry this slice, because 93 documents list one in",
        "their metadata. Body text says otherwise, and the slice is built on",
        f"`{primary}` instead.",
        "",
        f"**{s['repaired_identifier_occurrences']} occurrences were repaired** from",
        "line-number contamination in line-numbered draft guidances, where PDF extraction",
        "interleaves the marginal line number with the citation. Matches whose title",
        "exceeds the real number of CFR (50) or U.S. Code (54) titles are dropped outright.",
        "",
        "## Heading-detection quality",
        "",
        "Heuristic detection is imprecise in both directions, and an average hides both.",
        "These flags exist for hand inspection, and they gate nothing — Phase 3's",
        "ablation is what actually decides whether structure-aware chunking beats",
        "fixed-size chunking on this corpus.",
        "",
    ]

    flags = s["flags"]
    no_sec = flags["docs_with_no_sections"]
    lines += [
        f"**{len(no_sec)} documents produced no sections.** Structure-aware chunking must",
        "fall back to fixed-size for these, which is itself a comparison worth reporting.",
    ]
    if no_sec:
        lines += ["", "```", *[f"  {d}" for d in no_sec[:20]]]
        if len(no_sec) > 20:
            lines.append(f"  ... and {len(no_sec) - 20} more")
        lines.append("```")

    dense = flags["docs_with_suspiciously_dense_sections"]
    lines += [
        "",
        f"**{len(dense)} documents exceed {SECTIONS_PER_1K_CHARS_HIGH} sections per 1,000",
        "characters**, which usually means the detector is firing on body text rather than",
        "headings.",
    ]
    if dense:
        # len(dense) above is the true count; only the display is capped.
        lines += [
            "",
            *_table(
                [
                    (
                        d["doc_id"],
                        d["structure_source"],
                        d["n_sections"],
                        f"{d['n_chars']:,}",
                        f"{d['per_1k']:.2f}",
                    )
                    for d in dense[:10]
                ],
                ("Document", "Source", "Sections", "Chars", "Per 1k chars"),
            ),
        ]
        if len(dense) > 10:
            lines += ["", f"... and {len(dense) - 10} more."]

    thin = flags["docs_with_thin_text_per_page"]
    if thin:
        lines += [
            "",
            f"**{len(thin)} documents yield under {MIN_CHARS_PER_PAGE_OK} characters per",
            "page** — partial scans that passed Phase 0's document-level bar.",
            "",
            *_table(
                [(d["doc_id"], d["chars_per_page"]) for d in thin[:10]],
                ("Document", "Chars per page"),
            ),
        ]
        if len(thin) > 10:
            lines += ["", f"... and {len(thin) - 10} more."]

    if s["failures"]:
        lines += ["", "## Extraction failures", ""]
        lines += [f"- `{f['doc_id']}` — {f['error']}" for f in s["failures"]]

    return "\n".join(lines) + "\n"


def _fmt(d: dict[str, Any], total: bool = True) -> str:
    if not d.get("n"):
        return "no data"
    core = f"{d['min']:,} / {d['median']:,} / {d['p90']:,} / {d['max']:,}"
    return f"{core} (total {d['total']:,})" if total else f"{core} (no meaningful total)"


def write(payload: dict[str, Any], report_path: Path, json_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
