# Generation evaluation — grounded answers with verified citations

Generator `gemini-3.6-flash`, retriever `bm25`, top-5 context. 16 of 16 queries scored.

Queries are the first N of each slice from `corpus/evalset.jsonl` — a deterministic stratified prefix, not a random sample. Stratified because refusal rate is the headline and an unstratified draw would give the `unanswerable` slice a random size.

## Citation verification

Tier 1 locates each claimed quote in the source it cites (plan deviation #14). Nothing the model reports about position is trusted.

| Outcome | Citations | Share | Meaning |
|---|---:|---:|---|
| `exact` | 18 | 81.8% | byte-identical to the cited chunk |
| `normalized` | 4 | 18.2% | matches modulo whitespace runs (PDF artifacts) |
| `line_number_ambiguous` | 0 | 0.0% | **not verified** — failed the strict check, but would match if interleaved line numbers were discounted; an extraction-artifact hint |
| `wrong_chunk` | 0 | 0.0% | real text, wrong provenance — a plumbing defect |
| `unverified` | 0 | 0.0% | not in the document — fabricated or paraphrased |
| `too_short` | 0 | 0.0% | under 24 chars — not evidence |
| **total claimed** | **22** | | |

Mean citation precision (verified / claimed, averaged over answers that claimed at least one citation): **1.000**. Refusals claim nothing and are excluded — scoring their undefined precision as zero would make a model that refuses correctly look less faithful.

That is a **macro** average over answers. The **micro** figure — verified citations over all claimed citations, which is what the bucket table above sums to — is **1.000** (22/22). Both are correct and they answer different questions: the macro figure weights every answer equally, the micro figure weights every citation equally. An answer with one citation moves the macro average as much as an answer with nine.

## Refusal

| Population | Refusal rate | Correct behaviour |
|---|---:|---|
| `unanswerable` slice | 100.0% | refusing — higher is better |
| answerable slices (all) | 33.3% | answering — but see the caveat |
| answerable, excl. mention-labelled | 12.5% | answering — lower is better |

**The two answerable rows differ, and only the second means what the name suggests.** The `exact_identifier` slice marks a chunk relevant when an identifier *occurs* in it, which is the right label for retrieval and the wrong one for generation. Its queries ask what a regulation requires, while the corpus mostly just cites the regulation, so a refusal there is often correct rather than an over-refusal. Establishing real answerability is Phase 6's work.

## Per slice

| Slice | n | Refusal rate | Mean cite precision | Claimed | Verified | Label caveat |
|---|---:|---:|---:|---:|---:|---|
| `exact_identifier` | 4 | 75.0% | 1.000 | 1 | 1 | mention-labelled |
| `section_lookup` | 4 | 25.0% | 1.000 | 5 | 5 |  |
| `title_lookup` | 4 | 0.0% | 1.000 | 16 | 16 |  |
| `unanswerable` | 4 | 100.0% | n/a | 0 | 0 |  |

## Refusal threshold calibration

Pairwise separability of top retrieval score, answerable vs unanswerable: **1.0000** AUC over 48 pairs — the share of (answerable, unanswerable) pairs the retriever orders correctly, ties counting 0.5.

| Answerable pool | AUC | Pairs |
|---|---:|---:|
| all answerable slices | 1.0000 | 48 |
| `exact_identifier` only | 1.0000 | 16 |

Both pools are shown because the choice is not cosmetic. Phase 3 measured separability over `exact_identifier` against `unanswerable` only, and `reports/retrieval_eval.md` records why: pooling all answerable queries let section headings and document titles — long, high-IDF strings that score highly for unrelated reasons — dominate the pool and *reverse* which chunking strategy looked better. The pooled figure is the headline here because it describes the whole eval set, but the restricted row is the one comparable in population to Phase 3 (though still not in statistic — that was accuracy).

A perfect AUC says a separating threshold exists, not which to ship. What each candidate would do to this sample:

| Threshold | Unanswerable gated | Answerable wrongly gated |
|---:|---:|---:|
| 5.3523 | 0/4 | 0/12 |
| 6.5605 | 3/4 | 0/12 |
| 7.7687 | 4/4 | 1/12 |
| 8.9769 | 4/4 | 1/12 |
| 10.1851 | 4/4 | 2/12 |
| 11.3933 | 4/4 | 4/12 |
| 12.6015 | 4/4 | 5/12 |
| 13.8098 | 4/4 | 5/12 |
| 15.0180 | 4/4 | 5/12 |
| 16.2262 | 4/4 | 7/12 |
| 17.4344 | 4/4 | 8/12 |
| 18.6426 | 4/4 | 9/12 |
| 19.8508 | 4/4 | 11/12 |

The score gate is **off by default**, because the mechanism is retriever-specific. Phase 3 measured best-threshold separability accuracy at 0.983 for BM25 alone on `structural`, against 0.667 for min-max fusion and 0.942 for rank fusion plus reranking — so the threshold must be validated against whatever Phase 7 actually serves rather than inherited from an earlier table. Note that accuracy is a *different statistic* from the AUC above and the two are not comparable.

## Faithfulness — tier 1 and tier 2 composed

Tier 1 asks whether the quote is really in the document it cites. Tier 2 asks whether that quote supports the answer. **Neither implies the other**, and Phase 6 measured the gap directly: roughly a quarter of golden pairs had quotes that were real and did not back the claim.

| Measure | Value | Reading |
|---|---:|---|
| Citations claimed | 22 | what the model asserted |
| Located (tier 1) | 22 | the quote exists where cited |

Tier-2 verdicts, counted over **answers**:

| Verdict | Answers |
|---|---:|
| supported | 7 |
| partial | 1 |
| unsupported | 0 |
| unjudged | 0 |

**7 of 8 answers fully supported = 0.875.** The question put to the judge is whether *all* of an answer's located quotes together support it — one verdict per answer. This is the faithfulness figure.

Citation-level ratios are **deliberately absent** under this unit. `n_supported` counts answers while `n_located` counts citations, so dividing them mixes units: an earlier version of this report did exactly that and printed 7/22 = 0.318 as "the share of citations that survive reading", which it is not. Re-run with `--judge-unit citation` for a genuine per-citation breakdown.

Judge `gemini-3.1-flash-lite`, calibrated before use: it rejected 100.0% of constructed negatives (a real answer paired with another document's evidence, which cannot be supported) and accepted 87.5% of presumed-supported golden pairs. The negative rate is the one that decides usability: a judge answering "supported" to everything scores 1.000 on positives and 0.000 there.

## Cost and latency

| Measure | Value |
|---|---:|
| Prompt tokens | 29,811 |
| Output tokens | 4,215 |
| Thinking tokens | 0 |
| Latency p50 | 12.534s |
| Latency p95 | 27.105s |

The two latency rows use different conventions, which is worth stating rather than leaving to be discovered: `p50` is the interpolated median, `p95` is nearest-rank over 16 observations — at this sample size that makes p95 the single slowest query. Neither is a computation error; mixing them silently would be.

Thinking tokens are reported separately because they are billed and invisible in the response. They are zero here by configuration: Gemini 3.x thinks by default, `thinkingBudget: 0` is rejected with a 400, and `thinkingLevel: "minimal"` is the only setting that reaches zero — the default spent 105 thinking tokens to emit the single token `OK`.

No dollar figure: the free tier bills nothing and confirmed paid per-token rates for this model were not available, so a cost column would be a guess. Phase 4's audit found four such guessed literals in a report generator.
