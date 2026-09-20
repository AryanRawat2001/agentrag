# Extraction report — Phase 1

_Generated 2026-08-18T10:32:45+00:00_

Normalised text, page offsets, section structure, and exact identifiers for
every indexable document. Rebuild with `make extract`; a document is
re-extracted only when its manifest sha256 changes, so the Phase 0 content
pins double as the change-detection key.

## Coverage

| Metric | Value |
|---|---|
| Documents extracted | 155 |
| Extraction failed | 0 |
| With at least one section | 147 (94.8%) |

By source: `ctgov_protocol` 40, `fda_guidance` 115

## Structure source

Two detectors, because neither covers the corpus alone. `outline` means the
PDF ships bookmarks — authored headings with real nesting. `heuristic` means
they were inferred from line patterns, which is genuinely imprecise.

| Source | Documents |
|---|---|
| outline | 88 |
| heuristic | 59 |
| none | 8 |

| Metric | min / median / p90 / max (total) |
|---|---|
| Sections per document | 0.0 / 16.0 / 133.0 / 963.0 (total 7,306.0) |
| Characters per document | 694.0 / 40,816.0 / 187,891.0 / 801,075.0 (total 11,986,402.0) |
| Characters per page (per-document rate) | 564.48 / 2,340.91 / 3,043.03 / 6,951.38 (no meaningful total) |

Corpus-wide characters per page is **2,584.95** (total characters / total pages). The row above is the distribution of *per-document* rates, whose median sits below the corpus-wide figure because short documents are denser.

Character totals here run about 0.075% above the Phase 0 corpus report for the same documents. The difference is exactly two characters per page boundary — the `\n\n` separators added when pages are concatenated into one addressable blob. Phase 0 counted pages independently.

## Exact-identifier density

This table decides which identifier class the exact-identifier eval slice is
built on — the slice that demonstrates lexical retrieval beating dense
retrieval. Occurrences tell you how often a citation is mentioned; distinct
values tell you how many questions can actually be written.

**`distinct` is reported at two granularities**, because one number cannot serve
both purposes. Section level is the retrieval unit and the only level at which
classes are comparable; subsection level is the honest count of distinct
citations. An earlier version captured subsections for `usc` but not `cfr`, so
the two rows were measured at different resolutions inside a single column.

| Identifier | Docs | Coverage | Occurrences | Distinct (section) | Distinct (+subsec) |
|---|---|---|---|---|---|
| `cfr` **(primary)** | 109 | 70% | 2,587 | 521 | 1,098 |
| `usc` | 75 | 48% | 330 | 78 | 174 |
| `fed_register` | 45 | 29% | 143 | 87 | 87 |
| `docket` | 26 | 17% | 41 | 29 | 29 |
| `registry` | 20 | 13% | 56 | 40 | 40 |
| `ich` | 20 | 13% | 55 | 16 | 16 |

### Source split

The corpus-wide coverage column hides an almost total source split, which matters
because this table's job is choosing the slice. These classes are not spread
across the corpus — each is a property of a document type.

| Identifier | `ctgov_protocol` | `fda_guidance` |
|---|---|---|
| `cfr` | 18/40 docs, 56 occ. | 91/115 docs, 2,531 occ. |
| `usc` | 2/40 docs, 4 occ. | 73/115 docs, 326 occ. |
| `fed_register` | 0/40 docs, 0 occ. | 45/115 docs, 143 occ. |
| `registry` | 20/40 docs, 56 occ. | 0/115 docs, 0 occ. |
| `ich` | 8/40 docs, 19 occ. | 12/115 docs, 36 occ. |
| `docket` | 0/40 docs, 0 occ. | 26/115 docs, 41 occ. |

So the exact-identifier slice built on `cfr` is effectively FDA-only. That
is acceptable — it still exercises the retrieval behaviour the slice exists to
test — but it is a property to state rather than hide behind a corpus-wide
denominator.

Phase 0 assumed `docket` would carry this slice, because 93 documents list one in
their metadata. Body text says otherwise, and the slice is built on
`cfr` instead.

**63 occurrences were repaired** from
line-number contamination in line-numbered draft guidances, where PDF extraction
interleaves the marginal line number with the citation. Matches whose title
exceeds the real number of CFR (50) or U.S. Code (54) titles are dropped outright.

## Heading-detection quality

Heuristic detection is imprecise in both directions, and an average hides both.
These flags exist for hand inspection, and they gate nothing — Phase 3's
ablation is what actually decides whether structure-aware chunking beats
fixed-size chunking on this corpus.

**8 documents produced no sections.** Structure-aware chunking must
fall back to fixed-size for these, which is itself a comparison worth reporting.

```
  ctgov-NCT02545114-Prot_000
  ctgov-NCT02783898-Prot_001
  ctgov-NCT03388437-Prot_SAP_000
  ctgov-NCT03825393-Prot_SAP_000
  ctgov-NCT04486066-Prot_SAP_001
  ctgov-NCT04660552-Prot_003
  fda-71783
  fda-88642
```

**7 documents exceed 2.0 sections per 1,000
characters**, which usually means the detector is firing on body text rather than
headings.

| Document | Source | Sections | Chars | Per 1k chars |
|---|---|---|---|---|
| fda-70934 | heuristic | 81 | 26,290 | 3.08 |
| fda-175603 | heuristic | 387 | 143,195 | 2.70 |
| fda-72096 | heuristic | 5 | 1,872 | 2.67 |
| fda-75894 | heuristic | 963 | 368,423 | 2.61 |
| fda-75909 | outline | 228 | 94,921 | 2.40 |
| fda-72557 | heuristic | 63 | 27,043 | 2.33 |
| fda-94048 | outline | 14 | 6,326 | 2.21 |
