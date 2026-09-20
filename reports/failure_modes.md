# Failure analysis — five ways this pipeline is wrong

Every figure here is read from a committed artifact and recomputed on each build; a mutation test asserts that no number in this prose survives replacing the inputs. That is not ceremony. Writing this section is what revealed that the corpus line-number exposure quoted in three live places had been computed before the post-quarantine regeneration — the conclusion held, the arithmetic did not, and nothing caught it because the figure had never lived in a machine-readable file.

## 1. Dense retrieval collapses on the query type this corpus is for

| chunking | BM25 recall@10 | dense recall@10 | naive RRF | sparse / dense |
|---|---:|---:|---:|---:|
| `fixed` | 0.9553 | 0.0128 | 0.1628 | **75×** |
| `semantic` | 0.9717 | 0.0172 | 0.2386 | **56×** |
| `structural` | 0.9508 | 0.0222 | 0.2019 | **43×** |

On exact identifiers — `21 CFR 314.50`, `NCT02942264`, docket numbers — a sentence embedding is close to useless: **0.9553 against 0.0128**, a factor of 75. The embedding maps an identifier to a region of space shared by every other identifier, because the token is rare and carries no distributional meaning. Keyword matching has no such problem: the string is either there or it is not.

**And naive fusion makes it worse than sparse alone** (0.1628 against 0.9553). Reciprocal rank fusion rewards agreement between retrievers, so a retriever with no signal on this slice does not abstain — it votes, and outvotes a correct top hit. The fix that works is min-max fusion at a dense weight of 0.1, which matches BM25 on identifiers while gaining on section lookup.

**Consequence for the shipped system:** the served retriever is BM25 alone. That is not a simplification for the demo; it is what the table supports.

## 2. Reference answers manufacture obligations that the source does not impose

**8 of the golden set's answers restated a recommendation as a requirement**, found by reading every one of 25 judged pairs against its source.

| document | pair | what the source says vs what the answer says |
|---|---|---|
| `NCT02942264` | `016` | source: "the study treatment SHOULD be discontinued immediately, and the pregnancy reported to the Sponsor no later than 24 hours". Answer: "MUST be discontinued" and "MUST be reported". A… |
| `NCT04125745` | `021` | source: "CXA-10 capsules SHOULD be stored refrigerated between 2 and 8C". Answer: "MUST be refrigerated". A storage recommendation restated as a requirement. |
| `NCT04125745` | `023` | source: "For safety reasons, we MAY discontinue the study if the first two subjects enrolled experience any unexpected fatal or life-threatening events". Answer: "The study WILL be… |
| `fda-75426` | `085` | source: "The question presented for a vote SHOULD have minimal qualifiers, not be leading, and should avoid the use of double or triple negatives." Answer: "MUST contain minimal qualifiers,… |
| `fda-75426` | `087` | source: "the names of the committee members and their respective votes SHOULD be read aloud". Answer: "MUST be read aloud and entered into the public record." |
| `fda-75426` | `089` | source: "the Chair SHOULD first check with the DFO or other senior FDA officials". Answer: "The Chair MUST verify with the DFO or senior FDA officials." |
| `fda-78268` | `094` | source frames the whole list as advice: "What special RECOMMENDATIONS apply ... The following RECOMMENDATIONS apply to amendments, updates, and supplements", and the item itself is "Begin… |
| `fda-78268` | `097` | source: "You SHOULD include a Table of Contents: - With any submission regarding a GRAS notice...". Answer: "A Table of Contents IS REQUIRED for any submission". The question is leading in… |

FDA guidance states the convention explicitly: **must** is a requirement, **should** is a recommendation, **may** is an option. An answer that says *must* about a source that says *should* has invented a legal obligation — and it is lexically **indistinguishable** from a correct one, which is why five machine checks passed all of these. The last of those checks is answer/evidence word coverage, and a bag of words cannot see the difference between `may` and `shall` any more than between `60 days` and `30 days`.

3 of them come from a single document (`fda-75426`), which carries FDA's own `Contains Nonbinding Recommendations` marking.

**Consequence:** a reference answer with a manufactured requirement scores a *correct* model answer as wrong, so this failure mode corrupts the evaluation rather than the product. It is why the set is now curated and why the verdicts are committed.

## 3. The over-refusal rate was mostly a labelling failure, not a model failure

| slice | refused |
|---|---:|
| `exact_identifier` | 3/4 |
| `section_lookup` | 1/4 |
| `title_lookup` | 0/4 |
| `unanswerable` | 4/4 |

Measured naively, the model refused **4 of 12** answerable queries (33.3%), which reads as a broken refusal gate. It is not.

The `exact_identifier` ground truth marks a chunk relevant when the identifier **occurs** in it. That is correct for retrieval and wrong as an answerability label. `ident-0000`'s gold span is `21 CFR 1.980(k)` inside a footnote list: the document *cites* the regulation and never states its requirements, so a model asked "what are the requirements of 21 CFR 1.980?" and refusing is **right**, and scoring it as over-refusal measures the dataset.

Excluding the mention-labelled slice: **1 of 8** (12.5%). Unanswerable queries were refused 4/4.

The refusals that remain genuinely questionable:

- `section-0003` (section_lookup) — “3.7.4 Risk mitigation strategy”

**Consequence:** golden answers cannot be derived from the retrieval eval set, which is why Phase 6 drafted a separate one. A metric that looked like a model defect was a property of the ruler.

## 4. Answers reach past the span they cite

Tier 1 located **22 of 22** claimed citations — no fabricated quotes. Tier 2 then asked whether the located text *supports* the answer, and **7 of 8** answers were fully supported (0.875), with 1 partial.

A quote being real is not the same as a quote being sufficient. The recurring shape is an answer that synthesises across a chunk — combining a sentence with the one after it — while citing only the span that carries the headline phrase. Tier 1 cannot see this by construction: it asks "is this quote in the document?", and the answer is yes.

The judge that measures this is itself calibrated against **constructed negatives** — a real answer paired with another document's evidence, which cannot be supported under any reading. It rejects those at 100.0% and accepts true positives at 87.5%. The asymmetry is deliberate: a low negative rate is unambiguous evidence of a broken judge, while a low positive rate is ambiguous between a bad judge and bad reference data — and here it was the reference data, which is finding 2.

**Consequence:** read the headline as a range, not a point. At n=8 answers, one verdict either way moves it by 0.125

## 5. The verifier's own tiers can certify a change in meaning

The most uncomfortable finding, because it is a defect in the feature this project is *about*. A citation tier that discounted line-adjacent integers — added to handle FDA draft guidances, whose extracted text interleaves legislative line numbers mid-sentence — scored a model's **“40 CFR”** as a verified citation of a document's **“21 CFR”**.

Exposure, recomputed on the corpus currently on disk: **7,141 of 13,423 chunks (53.2%) lose at least one digit** to that normalisation, 86,231 digit characters in total. So a fabricated quote can land in that bucket by coincidence, across half the corpus.

Verification buckets from the shipped generation run:

| bucket | n |
|---|---:|
| `exact` | 18 |
| `normalized` | 4 |
| `line_number_ambiguous` | 0 |
| `too_short` | 0 |
| `unverified` | 0 |
| `wrong_chunk` | 0 |

The tier was demoted to a non-verified diagnostic, and it is reported as its own colour in the UI so it can never read as a pass. It survived its author's review because it was tested on the case it was built for and never on the case it would break. **When widening a check to fix false negatives, the test that matters is the one for the false positives it introduces.**

## Observed once, and therefore not a rate

Two failure modes have been seen in live use but are **absent from every artifact**, and the honest thing is to label them anecdotes rather than quietly promote them. The generation sample is n=16 because the free tier allows 20 requests per day, and that ceiling — not a sampling decision — is why these are unbounded.

- **`wrong_chunk`**: the very first request served through the container returned four citations, three `exact` and one whose quote was a real sentence from the retrieved context attributed to the **wrong chunk** — a true quote with a false address. The shipped eval shows `wrong_chunk: 0`, so this project has no measured rate for it.
- **`line_number_ambiguous` in production**: a live query about a device change control plan returned a quote that verified only under line-number normalisation. The shipped eval shows `line_number_ambiguous: 0`.

Both are visible on the dashboard when they occur, which is the point of showing the model's quote diffed against the document rather than a boolean.
