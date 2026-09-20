"""Phase 1b chunking checkpoint report.

Answers what Phase 2 and 3 need before the ablation can mean anything:

  chunk counts / sizes  -> does each strategy produce a comparable index, or is one
                           strategy quietly producing half as many chunks?
  section alignment     -> is `structural` actually structural, or mostly falling
                           back to fixed windows?
  duplicate clusters    -> how much of the corpus is repeated boilerplate, and
                           whether it repeats across documents or within one
  identifier coverage   -> whether every citation is reachable in some chunk (the
                           real ceiling on the exact-identifier eval slice) and how
                           dense citations are in the haystack

Population discipline as in earlier phases: each figure names the set it covers.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragpipe import chunking
from ragpipe import identifiers as ident_mod


def _dist(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "min": round(ordered[0], 1),
        "median": round(statistics.median(ordered), 1),
        "p90": round(ordered[min(n - 1, max(0, math.ceil(0.9 * n) - 1))], 1),
        "max": round(ordered[-1], 1),
        "total": round(sum(ordered), 1),
    }


def _table(rows: list[tuple[Any, ...]], headers: tuple[str, ...]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        *["| " + " | ".join(str(c) for c in row) + " |" for row in rows],
    ]


def _share(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%" if denominator else "n/a"


def _fmt(d: dict[str, Any]) -> str:
    if not d.get("n"):
        return "no data"
    return f"{d['min']:,} / {d['median']:,} / {d['p90']:,} / {d['max']:,}"


def build_strategy_stats(
    strategy: str,
    chunks: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    duplicate_of: dict[str, str],
    section_spans: dict[str, set[int]] | None = None,
) -> dict[str, Any]:
    """Per-strategy statistics from its chunk list and duplicate clustering."""
    sizes = [float(c["n_chars"]) for c in chunks]
    # `embed_text` is what gets embedded and indexed; `text` is the raw source
    # slice. Summing `n_chars` and calling it an embedding cost understated
    # `structural` by ~3%, because 95.8% of its chunks carry a prepended heading and
    # none of `fixed`'s do — an error that hit only the strategy the column exists
    # to compare.
    embed_sizes = [float(len(c.get("embed_text") or c["text"])) for c in chunks]
    per_doc = Counter(c["doc_id"] for c in chunks)
    primary = ident_mod.PRIMARY_KIND

    # Index volume counts overlapped text twice; corpus volume is the union of
    # spans per document. Both are real numbers about different things, and
    # reporting only the first made the two strategies look like they indexed
    # different amounts of content when they cover exactly the same text.
    covered_by_doc: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for c in chunks:
        covered_by_doc[c["doc_id"]].add((c["start"], c["end"]))
    unique_chars = 0
    for spans in covered_by_doc.values():
        merged_end = -1
        for start, end in sorted(spans):
            if start > merged_end:
                unique_chars += end - start
                merged_end = end
            elif end > merged_end:
                unique_chars += end - merged_end
                merged_end = end

    # Section alignment, measured against the *real* section offsets from the
    # extracted documents.
    #
    # Two wrong versions preceded this. Counting `section_idx is not None` counted
    # chunks that merely *carry* a label — sub-split windows inherit their parent's
    # idx. Deriving the section envelope from the labelled chunks themselves was
    # worse: self-referential, reducing to "was this label group emitted as one
    # chunk" and reporting 42.3% where agreement with actual section offsets is
    # 28.9%. The boundary set now comes from the documents, not from the output
    # being measured.
    spans = section_spans or {}
    labelled = exact = one_boundary = interior = 0
    for c in chunks:
        if c["section_idx"] is None:
            continue
        labelled += 1
        doc_bounds = spans.get(c["doc_id"], set())
        starts_on = c["start"] in doc_bounds
        ends_on = c["end"] in doc_bounds
        if starts_on and ends_on:
            exact += 1
        elif starts_on or ends_on:
            one_boundary += 1
        else:
            interior += 1

    fallback_docs = {c["doc_id"] for c in chunks if c["structural_fallback"]}

    with_primary = sum(1 for c in chunks if any(i["kind"] == primary for i in c["identifiers"]))
    with_any_id = sum(1 for c in chunks if c["identifiers"])
    # The actual ceiling the prose describes: distinct citations reachable in at
    # least one chunk. Chunk prevalence is a property of the haystack, not a ceiling,
    # and its denominator differs per strategy so the two rows were not comparable.
    reachable = {
        i["canonical"]
        for c in chunks
        for i in c["identifiers"]
        if i["kind"] == primary and i.get("canonical")
    }

    # A cluster spanning several documents is repeated boilerplate — the thing
    # worth collapsing. A cluster inside one document is usually an artifact of
    # window overlap and says nothing about the corpus.
    cross_doc = [c for c in clusters if c["n_documents"] > 1]
    within_doc = [c for c in clusters if c["n_documents"] == 1]

    return {
        "strategy": strategy,
        "n_chunks": len(chunks),
        "n_documents": len(per_doc),
        "sizes_chars": _dist(sizes),
        "index_chars": int(sum(embed_sizes)),
        "index_tokens": int(sum(embed_sizes) / chunking.CHARS_PER_TOKEN),
        "source_text_tokens": int(sum(sizes) / chunking.CHARS_PER_TOKEN),
        "heading_overhead_tokens": int((sum(embed_sizes) - sum(sizes)) / chunking.CHARS_PER_TOKEN),
        "unique_corpus_chars": unique_chars,
        "unique_corpus_tokens": int(unique_chars / chunking.CHARS_PER_TOKEN),
        "overlap_inflation": (sum(sizes) / unique_chars - 1.0) if unique_chars else 0.0,
        "chunks_per_document": _dist([float(v) for v in per_doc.values()]),
        "section_labelled": labelled,
        "section_exact_boundaries": exact,
        "section_one_boundary": one_boundary,
        "section_interior_window": interior,
        "structural_fallback_documents": len(fallback_docs),
        "chunks_with_primary_identifier": with_primary,
        "chunks_with_any_identifier": with_any_id,
        "distinct_primary_reachable": len(reachable),
        "duplicates": {
            "chunks_marked_duplicate": len(duplicate_of),
            "clusters": len(clusters),
            "cross_document_clusters": len(cross_doc),
            "within_document_clusters": len(within_doc),
            "largest_clusters": [
                {
                    "canonical_id": c["canonical_id"],
                    "size": c["size"],
                    "n_documents": c["n_documents"],
                    "similarity_floor": c["similarity_floor"],
                    "preview": c["preview"],
                }
                for c in sorted(clusters, key=lambda x: -x["size"])[:8]
            ],
        },
    }


def build(per_strategy: list[dict[str, Any]], deferred: dict[str, str]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "strategies": per_strategy,
        "deferred_strategies": deferred,
        "primary_identifier": ident_mod.PRIMARY_KIND,
    }


def render_markdown(s: dict[str, Any]) -> str:
    strategies = s["strategies"]
    primary = s["primary_identifier"]

    lines = [
        "# Chunking report — Phase 1b",
        "",
        f"_Generated {s['generated_at']}_",
        "",
        "Every strategy produces chunks in the same shape with the same metadata, so",
        "Phase 2's retrieval harness can score them against each other. Nothing here",
        "says which strategy is better — that is what the ablation is for. This report",
        "only establishes that the comparison will be fair.",
        "",
        "Chunk sizes are character counts, with 4 characters per token as the documented",
        "approximation. A real tokenizer arrives with the embedding model in Phase 3 and",
        "sizes get re-baselined then; all strategies share the same approximation, so the",
        "comparison between them is unaffected.",
        "",
        "## Strategies compared",
        "",
        *_table(
            [
                (
                    f"`{st['strategy']}`",
                    f"{st['n_chunks']:,}",
                    f"{st['index_tokens']:,}",
                    f"{st['unique_corpus_tokens']:,}",
                    f"+{100 * st['overlap_inflation']:.1f}%",
                    f"{st['heading_overhead_tokens']:,}",
                    _fmt(st["sizes_chars"]),
                    _fmt(st["chunks_per_document"]),
                )
                for st in strategies
            ],
            (
                "Strategy",
                "Chunks",
                "Index tokens",
                "Corpus tokens",
                "Overlap",
                "Heading tokens",
                "Chars min/med/p90/max",
                "Chunks per doc min/med/p90/max",
            ),
        ),
        "",
        "**Index tokens** is what would actually be embedded: `embed_text`, which for",
        "`structural` includes the prepended section heading. It counts overlapped text",
        "twice, so it is the embedding-cost figure. **Corpus tokens** is the union of source",
        "spans — the actual content, byte-identical between the two strategies.",
        "",
        "The two differ for two separate reasons, and they pull in opposite directions:",
        "`fixed` re-counts more text through overlap, while `structural` adds heading",
        "characters that are not in the source at all. An earlier version summed the raw",
        "source slices and called that the embedding bill, which understated `structural`",
        "by its entire heading overhead — an error affecting only the strategy the column",
        "exists to compare against.",
        "",
    ]

    if s["deferred_strategies"]:
        lines += [
            "**Not implemented:** "
            + "; ".join(f"`{k}` — {v}" for k, v in s["deferred_strategies"].items())
            + ". Registered rather than silently omitted, so the strategy table cannot be",
            "mistaken for a complete comparison.",
            "",
        ]

    def dup_share(st: dict[str, Any]) -> str:
        marked = st["duplicates"]["chunks_marked_duplicate"]
        return f"{100 * marked / max(st['n_chunks'], 1):.1f}%"

    lines += [
        "## Structure alignment",
        "",
        "How much of `structural` is *genuinely* section-aligned. This matters because a",
        "strategy that mostly falls back to fixed windows would score like `fixed` in the",
        "ablation while appearing to test something else.",
        "",
        "Counting chunks that merely carry a section label overstates it badly: an",
        "over-long section is sub-split into overlapping fixed windows, and every window",
        "inherits its parent's section index. Boundary agreement is the honest measure, so",
        "it is broken out rather than aggregated.",
        "",
        "**Definition, stated because two are plausible:** a cut counts as aligned when it",
        "falls on *any* real section boundary in that document, taken from the extracted",
        "sections rather than from the chunk output. The stricter reading — a chunk exactly",
        "spanning its own labelled section — is lower, because short sections are merged",
        "and the merged unit carries the first section's index while ending at a later",
        "section's boundary. Both cuts are still real structural boundaries, which is what",
        "this column measures.",
        "",
        *_table(
            [
                (
                    f"`{st['strategy']}`",
                    f"{st['section_labelled']:,}",
                    f"{st['section_exact_boundaries']:,} "
                    f"({_share(st['section_exact_boundaries'], st['section_labelled'])})",
                    f"{st['section_one_boundary']:,}",
                    f"{st['section_interior_window']:,}",
                    st["structural_fallback_documents"],
                )
                for st in strategies
            ],
            (
                "Strategy",
                "Carries a section label",
                "Both cuts on a section boundary",
                "One cut",
                "Neither cut",
                "Fallback docs",
            ),
        ),
        "",
        "Percentages are over the *labelled* chunks, which is what the four columns",
        "partition — not over all chunks. The `fixed` row is zero by construction rather",
        "than by measurement: it never assigns a section label.",
        "",
        "The *interior window* column is the number to watch: those chunks share no",
        "boundary with any section and are fixed windows wearing a section label.",
        "",
        "## Duplicate clustering",
        "",
        "Near-duplicates are detected by MinHash over word 5-grams and **clustered, not",
        "dropped**. Regulatory boilerplate is legitimately near-identical across",
        'documents, so deleting it would destroy the ability to answer "which documents',
        'impose this requirement?". One member is canonical; the rest record',
        "`duplicate_of` and keep their own source location.",
        "",
        "The cross-document versus within-document split is reported because the two",
        "mean different things — but both are real. Window overlap is not what creates",
        "them: 15% shared text tops out far below the 0.85 Jaccard threshold. Exactly one",
        "cluster in this corpus contains a pair of adjacent windows sharing their overlap",
        "region, and deleting that shared region leaves them at 0.880 — so repetition, not",
        "overlap, is what clustered them. Within-document clusters are genuine repetition:",
        "the same consent or diary language recurring at different points in one document,",
        "which is exactly the duplication a retrieval ablation should care about.",
        "",
        *_table(
            [
                (
                    f"`{st['strategy']}`",
                    f"{st['duplicates']['chunks_marked_duplicate']:,}",
                    dup_share(st),
                    st["duplicates"]["clusters"],
                    st["duplicates"]["cross_document_clusters"],
                    st["duplicates"]["within_document_clusters"],
                )
                for st in strategies
            ],
            (
                "Strategy",
                "Chunks marked dup",
                "Share",
                "Clusters",
                "Cross-doc",
                "Within-doc",
            ),
        ),
        "",
        "## Exact-identifier slice ceiling",
        "",
        "The ceiling on the exact-identifier slice is how many distinct citations are",
        "reachable in at least one chunk — a citation present in no chunk cannot be",
        "retrieved however good the retriever is. That is the first column.",
        "",
        "Chunk prevalence is reported alongside it, but it is *not* a ceiling: it",
        "describes how dense citations are in the haystack, and its denominator differs",
        "per strategy, so the two prevalence figures are not comparable with each other.",
        "",
        *_table(
            [
                (
                    f"`{st['strategy']}`",
                    f"**{st['distinct_primary_reachable']:,}**",
                    f"{st['chunks_with_primary_identifier']:,}",
                    f"{100 * st['chunks_with_primary_identifier'] / max(st['n_chunks'], 1):.1f}%",
                    f"{st['chunks_with_any_identifier']:,}",
                )
                for st in strategies
            ],
            (
                "Strategy",
                f"Distinct `{primary}` reachable (ceiling)",
                f"Chunks with `{primary}`",
                "Prevalence",
                "Chunks with any identifier",
            ),
        ),
    ]

    for st in strategies:
        biggest = st["duplicates"]["largest_clusters"]
        if not biggest:
            continue
        lines += [
            "",
            f"## Largest duplicate clusters — `{st['strategy']}`",
            "",
            "The floor is exact Jaccard over the *whole* cluster, so it can sit below the",
            "0.85 duplicate threshold on a cluster of three or more. That is single-linkage",
            "transitivity, not a threshold violation: one real cluster here has pairs at",
            "0.891 and 0.876 — both clearing the bar — while the two outer members sit at",
            "0.781 and were never compared directly. Reporting the true floor rather than",
            "only the similarity to the canonical member is what makes this visible.",
            "",
            "| Size | Docs | Exact Jaccard floor | Preview |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {c['size']} | {c['n_documents']} | {c['similarity_floor']} | {c['preview']} |"
            for c in biggest
        ]

    return "\n".join(lines) + "\n"


def write(payload: dict[str, Any], report_path: Path, json_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def cluster_metadata(
    clusters: list[Any], chunk_docs: dict[str, str], chunk_text: dict[str, str]
) -> list[dict[str, Any]]:
    """Enrich raw clusters with document spread and a text preview for the report."""
    out: list[dict[str, Any]] = []
    for c in clusters:
        docs = {chunk_docs[m] for m in c.member_ids if m in chunk_docs}
        preview = chunk_text.get(c.canonical_id, "")[:90].replace("\n", " ").replace("|", "\\|")
        out.append(
            {
                "canonical_id": c.canonical_id,
                "member_ids": c.member_ids,
                "size": c.size,
                "similarity_floor": c.similarity_floor,
                "n_documents": len(docs),
                "preview": preview + ("..." if len(preview) == 90 else ""),
            }
        )
    return out


def group_by_document(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        grouped[c["doc_id"]].append(c)
    return grouped
