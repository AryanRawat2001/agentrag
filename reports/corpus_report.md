# Corpus report — Phase 0

_Generated 2026-08-13T11:05:56+00:00_

Corpus: public FDA guidance documents + ClinicalTrials.gov protocols.
Reproduce with `make corpus`. `corpus/manifest.jsonl` records every document's
URL and, once fetched, its sha256; subsequent fetches verify against that pin,
so upstream content changes fail loudly instead of silently altering the corpus.

## Acquisition

| Stage | Documents |
|---|---|
| In manifest | 160 |
| Downloaded OK | 160 (100.0%) |
| Download failed | 0 |
| Not yet fetched | 0 |
| **Indexable** | **155** (96.9%) |
| sha256 pinned | 160 / 160 |

By source: `ctgov_protocol` 40, `fda_guidance` 120

## Text layer

Classified per page rather than per document, so a mostly-text file with a
scanned section is not averaged into looking fine.

| Verdict | Documents |
|---|---|
| digital_native | 154 |
| image_only | 5 |
| mixed | 1 |

The verdict is a coarse routing decision, and on its own it understates
partial scanning: **18 indexable documents
contain at least one image-only page** (40 pages in total),
but only 1 clears the 20% threshold for `mixed` —
17 are labelled `digital_native`. Per-document
`image_only_pages` in `corpus_stats.json` is the honest signal; the thresholds
are due a revisit once Phase 1 has real extracted text.

## Size

All three rows cover the same population — **indexable only** (155 documents) —
so the totals are directly comparable.

| Metric | min / median / p90 / max (total) |
|---|---|
| Pages | 1 / 17 / 79 / 222 (total 4,637) |
| Extracted characters | 694 / 40,784 / 187,735 / 800,633 (total 11,977,438) |
| Bytes on disk | 9,567 / 308,430 / 1,171,442 / 33,023,673 (total 138,457,036) |

Disk footprint of **everything downloaded**, quarantined documents included: 139,827,547 bytes.

Estimated tokens across indexable documents: **2,994,359**

| Chunk size | Estimated chunks |
|---|---|
| 256-token chunks | 13,760 |
| 512-token chunks | 6,880 |
| 1024-token chunks | 3,440 |

Estimates treat the corpus as one continuous token stream, so they ignore
per-document boundaries (~1 partial chunk per document) and are exactly 4:2:1
by construction.

## Eval-slice viability

Each row is a precondition for one planned eval slice. Units differ by row and
are stated explicitly — an occurrence count is not a document count.

| Slice | Unit | Count | Meaning |
|---|---|---|---|
| Version currency | documents | 16 | 8 complete draft/final pairs |
| Exact identifier (FDA) | documents | 93 | carry a docket ID in metadata |
| Exact identifier (FDA) | documents | 26 | docket ID appears in body text (41 occurrences) |
| Exact identifier (protocols) | documents | 20 | NCT ID in body text, of 40 with one in metadata |
| Table lookup | documents | 64 | contain >=1 table-like line (528 lines total) |
| Temporal | documents | 3 | undated, i.e. temporal edge cases |

The identifier rows are the load-bearing ones for the hybrid-versus-dense
claim, and they are the weakest: most documents carry an identifier only in
metadata. Either identifiers get indexed from metadata into searchable text, or
the slice is rebuilt on identifiers that are abundant in body text (CFR
citations, ICH codes). Resolved in Phase 1.

## ACL partition

Issuing center (FDA) or sponsor class (protocols), over **indexable only**
— the documents that will actually be retrievable. This is the document-level
permission filter for the access-control demo: a real partition, not synthetic.

| Partition | Documents |
|---|---|
| Center for Drug Evaluation and Research | 22 |
| Office of Inspections and Investigations | 15 |
| Center for Biologics Evaluation and Research | 15 |
| Center for Veterinary Medicine | 13 |
| Human Foods Program | 12 |
| Center for Devices and Radiological Health | 11 |
| Office of the Commissioner | 10 |
| Center for Tobacco Products | 10 |
| INDUSTRY | 7 |
| Office of Regulatory Affairs | 7 |
| NETWORK | 6 |
| NIH | 6 |
| OTHER | 6 |
| OTHER_GOV | 6 |
| FED | 5 |
| INDIV | 4 |

## Sampling

| Why selected | Documents |
|---|---|
| stratified_sponsor_class | 38 |
| largest_available_document | 2 |
| stratified_center_status_era | 104 |
| draft_final_pair | 16 |

## Data quality

Known defects in the source metadata, recorded rather than silently patched.

| Issue | Documents |
|---|---|
| fda docs with unusable issue date | 4 |
| fda docs with multiple offices | 50 |
| fda docs without docket | 24 |
| pdfs that failed to open | 0 |
| pdfs opened but negligible text | 5 |
| docs missing sha256 pin | 0 |

`pdfs that failed to open` counts files pypdf could not parse at all. It is
distinct from `pdfs opened but negligible text`, which counts files that parsed
cleanly and yielded almost nothing — the practical failure, and the one that
would poison an index silently.

## Most table-heavy documents

Hand-inspection shortlist for the table-lookup slice. The detector is a
whitespace-column proxy for ranking, not a table count.

| Table-like lines | Pages | Document |
|---|---|---|
| 102 | 114 | ctgov-NCT03801915-Prot_SAP_000 — Perioperative MVT-5873, a Fully Human Monoclonal Antibody Against a CA 19-9 Epit |
| 93 | 222 | ctgov-NCT03386721-Prot_SAP_000 — Basket Study to Evaluate the Therapeutic Activity of Simlukafusp Alfa as a Combi |
| 45 | 197 | ctgov-NCT00567567-Prot_SAP_000 — Comparing Two Different Myeloablation Therapies in Treating Young Patients Who A |
| 37 | 58 | ctgov-NCT02114684-Prot_000 — Improving Retreatment Success (IMPRESS) |
| 21 | 21 | ctgov-NCT04469686-Prot_SAP_000 — Phase 3 Study to Evaluate the Safety and Efficacy of Hydrocortisone Acetate Supp |
| 20 | 56 | ctgov-NCT04125745-Prot_SAP_000 — Oral CXA-10 in Pulmonary Arterial Hypertension |
| 18 | 67 | ctgov-NCT01749540-Prot_000 — Study to Evaluate the Safety and Efficacy of Luspatercept (ACE-536) in Participa |
| 18 | 93 | ctgov-NCT02669758-Prot_000 — A Long-Term Safety and Tolerability Study of ALKS 3831 in Adults With Schizophre |
| 16 | 221 | ctgov-NCT01511614-Prot_SAP_000 — Nicotine Withdrawal Symptoms and Smoking Relapse |
| 13 | 31 | ctgov-NCT04202133-Prot_000 — Neurocognitive Benefits of a Weight Management Program |

## Quarantined (not indexable)

| Verdict | Pages | Extracted chars | Document |
|---|---|---|---|
| image_only | 8 | 709 | fda-70844 — Industry-Supported Scientific and Educational Activites |
| image_only | 2 | 0 | fda-71910 — FDA Animal Products Database Data Entry Form |
| image_only | 4 | 0 | fda-72423 — General Guidelines for OTC Combination Products |
| image_only | 8 | 709 | fda-75334 — Industry Supported Scientific and Educational Activities:  Guidance fo |
| image_only | 1 | 0 | fda-75456 — Sample WHO Certificate for Quality of a Pharmaceutical Product |
