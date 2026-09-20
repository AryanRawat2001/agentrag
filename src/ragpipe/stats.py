"""Phase 0 checkpoint report.

Everything here answers a question a later phase depends on:

  chunk estimates      -> is the corpus small enough to re-embed for free during
                          the chunking sweep? (if not, the sweep won't happen)
  scanned rate         -> how much of the corpus needs quarantining
  table-heaviness      -> which documents to hand-inspect for the table slice
  identifier counts    -> is the exact-identifier eval slice viable
  draft/final pairs    -> is the version-currency slice viable
  ACL partition sizes  -> is permission-filtered retrieval demonstrable
  data-quality counts  -> what to disclose rather than quietly paper over

**Population discipline.** An independent audit of the first version of this
report found that every number was arithmetically correct while several were
computed over one document population and presented beside prose implying
another — bytes over all 160 downloaded documents next to characters over the 158
indexable ones, inflating the apparent corpus footprint by 5%. Nothing crashed
and nothing looked odd. That is exactly how a retrieval ablation table ends up
quietly wrong, so every figure below states its population explicitly and the
`POP_*` constants are used rather than ad-hoc filtering.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragpipe import pdfcheck
from ragpipe.models import FetchedDoc, SourceDoc

# Rough chars-per-token for English prose. Only used for order-of-magnitude
# chunk-count estimates, never for anything billed.
CHARS_PER_TOKEN = 4
CHUNK_OVERLAP = 0.15

POP_MANIFEST = "all 160 in manifest"
POP_DOWNLOADED = "all downloaded"
POP_INDEXABLE = "indexable only"


def _pct(n: int, total: int) -> str:
    return f"{(100 * n / total):.1f}%" if total else "n/a"


def _decade(iso_date: str | None) -> str:
    """ISO date -> decade label, e.g. '2010s'. Undated documents keep their own bucket."""
    if not iso_date or len(iso_date) < 4 or not iso_date[:4].isdigit():
        return "undated"
    return f"{iso_date[:3]}0s"


def _dist(values: list[int], population: str) -> dict[str, Any]:
    """Distribution summary. p90 is nearest-rank, which for small n is the only
    definition that cannot fall below the median (the naive
    `int(0.9 * (n - 1))` index returns the *minimum* at n=2)."""
    if not values:
        return {"n": 0, "population": population}
    ordered = sorted(values)
    n = len(ordered)
    p90_index = min(n - 1, max(0, math.ceil(0.9 * n) - 1))
    return {
        "n": n,
        "population": population,
        "min": ordered[0],
        "median": int(statistics.median(ordered)),
        "p90": ordered[p90_index],
        "max": ordered[-1],
        "total": sum(ordered),
    }


def _table(rows: list[tuple[Any, ...]], headers: tuple[str, ...]) -> list[str]:
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def build(manifest: list[SourceDoc], fetched: dict[str, FetchedDoc]) -> dict[str, Any]:
    """Compute the full stats payload from the manifest and fetch results."""
    ok = [d for d in manifest if fetched.get(d.doc_id) and fetched[d.doc_id].ok]
    failed = [d for d in manifest if d.doc_id in fetched and not fetched[d.doc_id].ok]
    missing = [d for d in manifest if d.doc_id not in fetched]

    def f(doc: SourceDoc) -> FetchedDoc:
        return fetched[doc.doc_id]

    indexable = [d for d in ok if pdfcheck.is_indexable(f(d).text_layer)]
    total_chars = sum(f(d).text_chars or 0 for d in indexable)
    est_tokens = total_chars // CHARS_PER_TOKEN

    stats: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "counts": {
            "in_manifest": len(manifest),
            "downloaded_ok": len(ok),
            "download_failed": len(failed),
            "not_yet_fetched": len(missing),
            "indexable": len(indexable),
            "pinned_sha256": sum(1 for d in manifest if d.sha256),
        },
        "by_source": dict(Counter(d.source for d in manifest)),
        "text_layer": dict(Counter(f(d).text_layer or "unknown" for d in ok)),
        # All three distributions cover the SAME population, so the totals are
        # directly comparable and the section's prose applies to every row.
        "pages": _dist([f(d).pages or 0 for d in indexable], POP_INDEXABLE),
        "chars": _dist([f(d).text_chars or 0 for d in indexable], POP_INDEXABLE),
        "bytes": _dist([f(d).bytes or 0 for d in indexable], POP_INDEXABLE),
        # Reported separately and labelled, because disk footprint is genuinely a
        # property of everything downloaded, quarantined documents included.
        "downloaded_footprint_bytes": sum(f(d).bytes or 0 for d in ok),
        "estimated_tokens": est_tokens,
        "estimated_chunks": {
            f"{size}_tok": int(est_tokens / (size * (1 - CHUNK_OVERLAP))) if est_tokens else 0
            for size in (256, 512, 1024)
        },
    }

    # --- text-layer detail ---------------------------------------------------
    # The verdict alone understates partial scanning: at MIXED_THRESHOLD = 0.20 a
    # document needs a fifth of its pages blank to be labelled `mixed`, so
    # documents with a scanned appendix mostly read as `digital_native`.
    blank_page_docs = [d for d in indexable if (f(d).image_only_pages or 0) > 0]
    stats["partial_scans"] = {
        "indexable_docs_with_blank_pages": len(blank_page_docs),
        "labelled_mixed": sum(1 for d in blank_page_docs if f(d).text_layer == "mixed"),
        "labelled_digital_native": sum(
            1 for d in blank_page_docs if f(d).text_layer == "digital_native"
        ),
        "blank_pages_total": sum(f(d).image_only_pages or 0 for d in indexable),
    }

    # --- eval-slice viability ------------------------------------------------
    pairs: dict[str, list[str]] = defaultdict(list)
    for d in indexable:
        if key := d.extra.get("pair_key"):
            pairs[key].append(f"{d.status}:{d.doc_id}")

    fda_idx = [d for d in indexable if d.source == "fda_guidance"]
    ctg_idx = [d for d in indexable if d.source == "ctgov_protocol"]

    # Identifier classes are counted separately. Bundling them hid that the FDA
    # docket pattern structurally cannot match an NCT ID, so every protocol
    # scored zero and the combined figure implied coverage that did not exist.
    stats["slices"] = {
        "draft_final_pairs_complete": sum(1 for v in pairs.values() if len(v) >= 2),
        "fda_docs_with_docket_metadata": sum(1 for d in fda_idx if d.docket),
        "fda_docs_with_docket_in_text": sum(
            1 for d in fda_idx if (f(d).docket_like_tokens or 0) > 0
        ),
        "fda_docket_tokens_in_text": sum(f(d).docket_like_tokens or 0 for d in fda_idx),
        "protocols_with_registry_metadata": sum(1 for d in ctg_idx if d.docket),
        "protocols_with_registry_id_in_text": sum(
            1 for d in ctg_idx if (f(d).registry_like_tokens or 0) > 0
        ),
        "registry_tokens_in_text": sum(f(d).registry_like_tokens or 0 for d in indexable),
        "docs_with_table_like_lines": sum(1 for d in indexable if (f(d).table_like_lines or 0) > 0),
        "table_like_lines_total": sum(f(d).table_like_lines or 0 for d in indexable),
        "undated_docs": sum(1 for d in indexable if not d.issue_date),
    }

    stats["acl_partition"] = dict(
        Counter(
            d.center or d.extra.get("sponsor_class") or "unknown" for d in indexable
        ).most_common()
    )
    stats["status"] = dict(Counter(d.status or "n/a" for d in manifest))
    stats["era"] = dict(Counter(_decade(d.issue_date) for d in manifest).most_common())
    stats["sample_reason"] = dict(
        Counter(d.extra.get("sample_reason") or "unknown" for d in manifest)
    )

    # --- data quality --------------------------------------------------------
    # Each check enforces exactly what its label claims. The previous version
    # counted multi-office documents across the whole manifest with no source
    # filter, and counted only the `date_invalid` flag while claiming to cover
    # missing dates — both correct by coincidence on this data, both liable to
    # drift silently on the next.
    fda_all = [d for d in manifest if d.source == "fda_guidance"]
    negligible = [
        d
        for d in ok
        if (f(d).pages or 0) > 0
        and (f(d).text_chars or 0) / (f(d).pages or 1) < pdfcheck.MIN_CHARS_PER_TEXT_PAGE
    ]
    stats["data_quality"] = {
        "fda_docs_with_unusable_issue_date": sum(
            1 for d in fda_all if d.extra.get("date_invalid") or not d.issue_date
        ),
        "fda_docs_with_multiple_offices": sum(1 for d in fda_all if len(d.offices) > 1),
        "fda_docs_without_docket": sum(1 for d in fda_all if not d.docket),
        "pdfs_that_failed_to_open": sum(1 for d in ok if f(d).text_layer == "unreadable"),
        "pdfs_opened_but_negligible_text": len(negligible),
        "docs_missing_sha256_pin": sum(1 for d in manifest if not d.sha256),
    }

    stats["most_table_heavy"] = [
        {
            "doc_id": d.doc_id,
            "table_like_lines": f(d).table_like_lines or 0,
            "pages": f(d).pages,
            "title": d.title[:80],
        }
        for d in sorted(indexable, key=lambda x: -(f(x).table_like_lines or 0))[:10]
    ]
    stats["failures"] = [{"doc_id": d.doc_id, "url": d.url, "error": f(d).error} for d in failed]
    stats["quarantined"] = [
        {
            "doc_id": d.doc_id,
            "text_layer": f(d).text_layer,
            "pages": f(d).pages,
            "text_chars": f(d).text_chars,
            "title": d.title[:70],
        }
        for d in ok
        if not pdfcheck.is_indexable(f(d).text_layer)
    ]
    return stats


def _fmt_dist(d: dict[str, Any]) -> str:
    if not d.get("n"):
        return "no data"
    return f"{d['min']:,} / {d['median']:,} / {d['p90']:,} / {d['max']:,} (total {d['total']:,})"


def render_markdown(stats: dict[str, Any]) -> str:
    c = stats["counts"]
    total = c["in_manifest"]
    sl = stats["slices"]
    ps = stats["partial_scans"]
    n_idx = c["indexable"]

    lines = [
        "# Corpus report — Phase 0",
        "",
        f"_Generated {stats['generated_at']}_",
        "",
        "Corpus: public FDA guidance documents + ClinicalTrials.gov protocols.",
        "Reproduce with `make corpus`. `corpus/manifest.jsonl` records every document's",
        "URL and, once fetched, its sha256; subsequent fetches verify against that pin,",
        "so upstream content changes fail loudly instead of silently altering the corpus.",
        "",
        "## Acquisition",
        "",
        *_table(
            [
                ("In manifest", total),
                ("Downloaded OK", f"{c['downloaded_ok']} ({_pct(c['downloaded_ok'], total)})"),
                ("Download failed", c["download_failed"]),
                ("Not yet fetched", c["not_yet_fetched"]),
                ("**Indexable**", f"**{n_idx}** ({_pct(n_idx, total)})"),
                ("sha256 pinned", f"{c['pinned_sha256']} / {total}"),
            ],
            ("Stage", "Documents"),
        ),
        "",
        "By source: " + ", ".join(f"`{k}` {v}" for k, v in stats["by_source"].items()),
        "",
        "## Text layer",
        "",
        "Classified per page rather than per document, so a mostly-text file with a",
        "scanned section is not averaged into looking fine.",
        "",
        *_table(list(stats["text_layer"].items()), ("Verdict", "Documents")),
        "",
        "The verdict is a coarse routing decision, and on its own it understates",
        f"partial scanning: **{ps['indexable_docs_with_blank_pages']} indexable documents",
        f"contain at least one image-only page** ({ps['blank_pages_total']} pages in total),",
        f"but only {ps['labelled_mixed']} clears the 20% threshold for `mixed` —",
        f"{ps['labelled_digital_native']} are labelled `digital_native`. Per-document",
        "`image_only_pages` in `corpus_stats.json` is the honest signal; the thresholds",
        "are due a revisit once Phase 1 has real extracted text.",
        "",
        "## Size",
        "",
        f"All three rows cover the same population — **{POP_INDEXABLE}** ({n_idx} documents) —",
        "so the totals are directly comparable.",
        "",
        *_table(
            [
                ("Pages", _fmt_dist(stats["pages"])),
                ("Extracted characters", _fmt_dist(stats["chars"])),
                ("Bytes on disk", _fmt_dist(stats["bytes"])),
            ],
            ("Metric", "min / median / p90 / max (total)"),
        ),
        "",
        f"Disk footprint of **everything downloaded**, quarantined documents included: "
        f"{stats['downloaded_footprint_bytes']:,} bytes.",
        "",
        f"Estimated tokens across indexable documents: **{stats['estimated_tokens']:,}**",
        "",
        *_table(
            [
                (f"{k.replace('_tok', '')}-token chunks", f"{v:,}")
                for k, v in stats["estimated_chunks"].items()
            ],
            ("Chunk size", "Estimated chunks"),
        ),
        "",
        "Estimates treat the corpus as one continuous token stream, so they ignore",
        "per-document boundaries (~1 partial chunk per document) and are exactly 4:2:1",
        "by construction.",
        "",
        "## Eval-slice viability",
        "",
        "Each row is a precondition for one planned eval slice. Units differ by row and",
        "are stated explicitly — an occurrence count is not a document count.",
        "",
        *_table(
            [
                (
                    "Version currency",
                    "documents",
                    2 * sl["draft_final_pairs_complete"],
                    f"{sl['draft_final_pairs_complete']} complete draft/final pairs",
                ),
                (
                    "Exact identifier (FDA)",
                    "documents",
                    sl["fda_docs_with_docket_metadata"],
                    "carry a docket ID in metadata",
                ),
                (
                    "Exact identifier (FDA)",
                    "documents",
                    sl["fda_docs_with_docket_in_text"],
                    f"docket ID appears in body text "
                    f"({sl['fda_docket_tokens_in_text']} occurrences)",
                ),
                (
                    "Exact identifier (protocols)",
                    "documents",
                    sl["protocols_with_registry_id_in_text"],
                    f"NCT ID in body text, of "
                    f"{sl['protocols_with_registry_metadata']} with one in metadata",
                ),
                (
                    "Table lookup",
                    "documents",
                    sl["docs_with_table_like_lines"],
                    f"contain >=1 table-like line ({sl['table_like_lines_total']} lines total)",
                ),
                (
                    "Temporal",
                    "documents",
                    sl["undated_docs"],
                    "undated, i.e. temporal edge cases",
                ),
            ],
            ("Slice", "Unit", "Count", "Meaning"),
        ),
        "",
        "The identifier rows are the load-bearing ones for the hybrid-versus-dense",
        "claim, and they are the weakest: most documents carry an identifier only in",
        "metadata. Either identifiers get indexed from metadata into searchable text, or",
        "the slice is rebuilt on identifiers that are abundant in body text (CFR",
        "citations, ICH codes). Resolved in Phase 1.",
        "",
        "## ACL partition",
        "",
        f"Issuing center (FDA) or sponsor class (protocols), over **{POP_INDEXABLE}**",
        "— the documents that will actually be retrievable. This is the document-level",
        "permission filter for the access-control demo: a real partition, not synthetic.",
        "",
        *_table(list(stats["acl_partition"].items()), ("Partition", "Documents")),
        "",
        "## Sampling",
        "",
        *_table(list(stats["sample_reason"].items()), ("Why selected", "Documents")),
        "",
        "## Data quality",
        "",
        "Known defects in the source metadata, recorded rather than silently patched.",
        "",
        *_table(
            [(k.replace("_", " "), v) for k, v in stats["data_quality"].items()],
            ("Issue", "Documents"),
        ),
        "",
        "`pdfs that failed to open` counts files pypdf could not parse at all. It is",
        "distinct from `pdfs opened but negligible text`, which counts files that parsed",
        "cleanly and yielded almost nothing — the practical failure, and the one that",
        "would poison an index silently.",
    ]

    if stats["most_table_heavy"]:
        lines += [
            "",
            "## Most table-heavy documents",
            "",
            "Hand-inspection shortlist for the table-lookup slice. The detector is a",
            "whitespace-column proxy for ranking, not a table count.",
            "",
            "| Table-like lines | Pages | Document |",
            "|---|---|---|",
        ]
        lines += [
            f"| {d['table_like_lines']:,} | {d['pages']} | {d['doc_id']} — {d['title']} |"
            for d in stats["most_table_heavy"]
        ]

    if stats["quarantined"]:
        lines += [
            "",
            "## Quarantined (not indexable)",
            "",
            "| Verdict | Pages | Extracted chars | Document |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {d['text_layer']} | {d['pages']} | {d['text_chars']} "
            f"| {d['doc_id']} — {d['title']} |"
            for d in stats["quarantined"]
        ]

    if stats["failures"]:
        lines += ["", "## Download failures", ""]
        lines += [f"- `{d['doc_id']}` — {d['error']}" for d in stats["failures"]]

    return "\n".join(lines) + "\n"


def write(stats: dict[str, Any], report_path: Path, json_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(stats), encoding="utf-8")
    json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
