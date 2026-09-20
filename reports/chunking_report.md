# Chunking report — Phase 1b

_Generated 2026-08-18T10:42:33+00:00_

Every strategy produces chunks in the same shape with the same metadata, so
Phase 2's retrieval harness can score them against each other. Nothing here
says which strategy is better — that is what the ablation is for. This report
only establishes that the comparison will be fair.

Chunk sizes are character counts, with 4 characters per token as the documented
approximation. A real tokenizer arrives with the embedding model in Phase 3 and
sizes get re-baselined then; all strategies share the same approximation, so the
comparison between them is unaffected.

## Strategies compared

| Strategy | Chunks | Index tokens | Corpus tokens | Overlap | Heading tokens | Chars min/med/p90/max | Chunks per doc min/med/p90/max |
|---|---|---|---|---|---|---|---|
| `fixed` | 11,527 | 3,172,599 | 2,737,505 | +15.9% | 0 | 201.0 / 1,091.0 / 1,216.0 / 1,274.0 | 1.0 / 41.0 / 176.0 / 717.0 |
| `semantic` | 11,468 | 2,737,505 | 2,737,505 | +0.0% | 0 | 200.0 / 1,056.0 / 1,196.0 / 1,461.0 | 2.0 / 40.5 / 179.0 / 614.0 |
| `structural` | 13,423 | 3,193,185 | 2,737,505 | +11.4% | 142,833 | 200.0 / 1,040.0 / 1,191.0 / 1,314.0 | 1.0 / 46.0 / 234.0 / 765.0 |

**Index tokens** is what would actually be embedded: `embed_text`, which for
`structural` includes the prepended section heading. It counts overlapped text
twice, so it is the embedding-cost figure. **Corpus tokens** is the union of source
spans — the actual content, byte-identical between the two strategies.

The two differ for two separate reasons, and they pull in opposite directions:
`fixed` re-counts more text through overlap, while `structural` adds heading
characters that are not in the source at all. An earlier version summed the raw
source slices and called that the embedding bill, which understated `structural`
by its entire heading overhead — an error affecting only the strategy the column
exists to compare against.

## Structure alignment

How much of `structural` is *genuinely* section-aligned. This matters because a
strategy that mostly falls back to fixed windows would score like `fixed` in the
ablation while appearing to test something else.

Counting chunks that merely carry a section label overstates it badly: an
over-long section is sub-split into overlapping fixed windows, and every window
inherits its parent's section index. Boundary agreement is the honest measure, so
it is broken out rather than aggregated.

**Definition, stated because two are plausible:** a cut counts as aligned when it
falls on *any* real section boundary in that document, taken from the extracted
sections rather than from the chunk output. The stricter reading — a chunk exactly
spanning its own labelled section — is lower, because short sections are merged
and the merged unit carries the first section's index while ending at a later
section's boundary. Both cuts are still real structural boundaries, which is what
this column measures.

| Strategy | Carries a section label | Both cuts on a section boundary | One cut | Neither cut | Fallback docs |
|---|---|---|---|---|---|
| `fixed` | 0 | 0 (n/a) | 0 | 0 | 0 |
| `semantic` | 0 | 0 (n/a) | 0 | 0 | 0 |
| `structural` | 12,720 | 3,013 (23.7%) | 4,213 | 5,494 | 8 |

Percentages are over the *labelled* chunks, which is what the four columns
partition — not over all chunks. The `fixed` row is zero by construction rather
than by measurement: it never assigns a section label.

The *interior window* column is the number to watch: those chunks share no
boundary with any section and are fixed windows wearing a section label.

## Duplicate clustering

Near-duplicates are detected by MinHash over word 5-grams and **clustered, not
dropped**. Regulatory boilerplate is legitimately near-identical across
documents, so deleting it would destroy the ability to answer "which documents
impose this requirement?". One member is canonical; the rest record
`duplicate_of` and keep their own source location.

The cross-document versus within-document split is reported because the two
mean different things — but both are real. Window overlap is not what creates
them: 15% shared text tops out far below the 0.85 Jaccard threshold. Exactly one
cluster in this corpus contains a pair of adjacent windows sharing their overlap
region, and deleting that shared region leaves them at 0.880 — so repetition, not
overlap, is what clustered them. Within-document clusters are genuine repetition:
the same consent or diary language recurring at different points in one document,
which is exactly the duplication a retrieval ablation should care about.

| Strategy | Chunks marked dup | Share | Clusters | Cross-doc | Within-doc |
|---|---|---|---|---|---|
| `fixed` | 19 | 0.2% | 19 | 4 | 15 |
| `semantic` | 76 | 0.7% | 67 | 24 | 43 |
| `structural` | 52 | 0.4% | 51 | 23 | 28 |

## Exact-identifier slice ceiling

The ceiling on the exact-identifier slice is how many distinct citations are
reachable in at least one chunk — a citation present in no chunk cannot be
retrieved however good the retriever is. That is the first column.

Chunk prevalence is reported alongside it, but it is *not* a ceiling: it
describes how dense citations are in the haystack, and its denominator differs
per strategy, so the two prevalence figures are not comparable with each other.

| Strategy | Distinct `cfr` reachable (ceiling) | Chunks with `cfr` | Prevalence | Chunks with any identifier |
|---|---|---|---|---|
| `fixed` | **520** | 1,370 | 11.9% | 1,709 |
| `semantic` | **520** | 1,222 | 10.7% | 1,526 |
| `structural` | **520** | 1,358 | 10.1% | 1,701 |

## Largest duplicate clusters — `fixed`

The floor is exact Jaccard over the *whole* cluster, so it can sit below the
0.85 duplicate threshold on a cluster of three or more. That is single-linkage
transitivity, not a threshold violation: one real cluster here has pairs at
0.891 and 0.876 — both clearing the bar — while the two outer members sit at
0.781 and were never compared directly. Reporting the true floor rather than
only the similarity to the canonical member is what makes this visible.

| Size | Docs | Exact Jaccard floor | Preview |
|---|---|---|---|
| 2 | 1 | 0.865 |  (Section 7.1.2)    Refer to Protocol Section 7.0 for  complete list of observations and  ... |
| 2 | 1 | 0.924 |  low number of  white blood cells can  make it easier to get  infections  o a low number o... |
| 2 | 1 | 0.942 | if the subject or his/her guardian requests the release of  information in writing, the Ce... |
| 2 | 1 | 0.989 | NVITED TO TAKE PART IN THIS STUDY?    This is a clinical trial, a type of research study. ... |
| 2 | 1 | 0.886 | 32  Page 153  and surgery are used to kill and remove as much tumor as possible.  Blood st... |
| 2 | 1 | 0.941 | ing neuroblastoma than treatments we have used in the past. Unfortunately, there  is no gu... |
| 2 | 1 | 1.0 |  government, which will help us protect the privac y of our research subjects. Information... |
| 2 | 1 | 0.853 | ocument which are in bold type in their entirety.) Editorial  changes to these sections ma... |

## Largest duplicate clusters — `semantic`

The floor is exact Jaccard over the *whole* cluster, so it can sit below the
0.85 duplicate threshold on a cluster of three or more. That is single-linkage
transitivity, not a threshold violation: one real cluster here has pairs at
0.891 and 0.876 — both clearing the bar — while the two outer members sit at
0.781 and were never compared directly. Reporting the true floor rather than
only the similarity to the canonical member is what makes this visible.

| Size | Docs | Exact Jaccard floor | Preview |
|---|---|---|---|
| 5 | 3 | 0.667 | Important Medical Events (IME) that may not result in death, be life threatening, or requi... |
| 3 | 1 | 0.878 | Pediatr Blood  Cancer 46(7):719-22, 2006  ANBL0532  Page 131    SAMPLE INFORMED CONSENT / ... |
| 3 | 1 | 1.0 | Your doctor will still take care of you.    We will tell you about new information that ma... |
| 3 | 1 | 0.9 | WHOM DO I CALL IF I HAVE QUESTIONS OR PROBLEMS?    For questions about the study or a rese... |
| 3 | 1 | 1.0 | This web site will not include information that can identify you. At most, the web site  w... |
| 3 | 1 | 0.884 | Each dose of panobinostat should be taken with a 4 oz / 120 ml glass of water .  Drug must... |
| 3 | 3 | 1.0 | FDA’s guidance documents, including this guidance, do not establish legally enforceable  r... |
| 2 | 1 | 1.0 | Patients < 12 months of age: 0.017  mg/kg/dose once daily x 3 doses  Patients > 12 months ... |

## Largest duplicate clusters — `structural`

The floor is exact Jaccard over the *whole* cluster, so it can sit below the
0.85 duplicate threshold on a cluster of three or more. That is single-linkage
transitivity, not a threshold violation: one real cluster here has pairs at
0.891 and 0.876 — both clearing the bar — while the two outer members sit at
0.781 and were never compared directly. Reporting the true floor rather than
only the similarity to the canonical member is what makes this visible.

| Size | Docs | Exact Jaccard floor | Preview |
|---|---|---|---|
| 3 | 3 | 0.789 | Renal Cell Carcinoma: Developing Drugs and Biologics for  Adjuvant Treatment   Guidance fo... |
| 2 | 1 | 0.865 |  (Section 7.1.2)    Refer to Protocol Section 7.0 for  complete list of observations and  ... |
| 2 | 1 | 0.924 |  low number of  white blood cells can  make it easier to get  infections  o a low number o... |
| 2 | 1 | 0.942 | if the subject or his/her guardian requests the release of  information in writing, the Ce... |
| 2 | 1 | 1.0 | ocument which are in bold type in their entirety.) Editorial  changes to these sections ma... |
| 2 | 1 | 1.0 | this one.    You are being asked to take part in this study because you have high risk neu... |
| 2 | 1 | 0.917 | uring therapy, please ask your doctor. If you  become pregnant during the research study, ... |
| 2 | 1 | 0.941 | ing neuroblastoma than treatments we have used in the past. Unfortunately, there  is no gu... |
