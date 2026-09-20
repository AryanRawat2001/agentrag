# Serving path — per-stage latency

Plan deviation #8 asks for per-stage p50/p95 and cost per query. This is the re-runnable half: `make bench` regenerates it, and the figures below are the artifact rather than a line in a chat log. `/stats` on a running service reports the same stages over real served traffic, including end-to-end totals.

| stage | n | p50 | p95 | min | max | source |
|---|---:|---:|---:|---:|---:|---|
| `retrieve` | 540 | **0.179 ms** | 0.280 ms | 0.089 ms | 0.562 ms | measured |
| `verify` | 24 | **0.006 ms** | 0.211 ms | 0.002 ms | 0.216 ms | measured |
| `generate` | 16 | **12.53 s** | 27.11 s | 2.16 s | 27.11 s | stored |

Notes per stage, because the sample sizes are not comparable:

- **`retrieve`** — BM25 over 13,423 passages, k=5, 3 passes
- **`verify`** — 8 real answers, 22 citations replayed, 3 passes
- **`generate`** — from reports/generation_eval.json (gemini-3.6-flash); not re-measured, because the free tier is 20 requests per day

## The shape is the finding

**Generation costs roughly 5 orders of magnitude more than retrieval** — 12.53 s against 0.179 ms at p50. Stated as a power of ten rather than a multiplier on purpose: the numerator is the median of a distribution that spans more than a factor of ten (below), so a precise-looking multiple would be false precision.

Every local stage in this pipeline is free by comparison, which is why the stages are timed separately. A single end-to-end number would move only when the third-party API moved, and would hide the fact that **nothing this project controls is the bottleneck** — no amount of retrieval or verifier optimisation would change a served request's latency measurably.

**"The verifier is free" is a measurement, not a claim.** Verification runs in 0.006 ms at p50 over 24 replays — the citation check that is this project's entire point costs less than the search that feeds it. It is timed rather than assumed because a verification step that was expensive would be a reason not to ship one, and that deserved a number.

## The generation figure is a wide distribution, not a number

Generation ranged **2.16 s to 27.11 s** across 16 calls — a 12.6x spread on the same model, with p50 12.53 s and p95 27.11 s. Output size explains little of it: the correlation between tokens produced and time taken is weak, so this is variance in somebody else's infrastructure rather than variance in the work.

Practical consequence: **a single generation latency figure from this project should not be trusted to two significant figures.** An earlier live request through the container measured ~6.3 s, which looks like a contradiction of the p50 above and is simply one draw from this distribution. Quote the range, or quote p50 *with* p95 — and treat any single observation as an anecdote.

## Why the verify row has a wide p95

Not noise, and not cold start. Verification cost is driven by the **verdict it reaches**, because the two verified methods do different amounts of work: `exact` is a substring search, while `normalized` has to build a whitespace-collapsed copy of the chunk plus an index map back to the original character offsets.

**Different estimator from the stage table above, deliberately.** Those rows are raw single-shot timings; these are each answer's *median over many repeats*, because an operation costing 0.005 ms is otherwise measuring scheduler noise. That is why the slowest figure here can sit *below* the stage row's max — the two describe different populations, and comparing them across tables would be an error.

| answers containing | answers | median over repeats | slowest answer |
|---|---:|---:|---:|
| `exact` | 6 | 0.005 ms | 0.013 ms |
| `normalized` | 2 | 0.134 ms | 0.200 ms |

An answer whose citations all matched exactly verifies in 0.005 ms; one containing a normalised match takes 0.134 ms — **27x** more. The direction matters more than the size: `normalized` is the *common* case in text extracted from PDFs, where line breaks land mid-sentence, so real traffic sits at the expensive end of this range rather than the cheap end this sample is weighted toward. Even there it is 5 orders of magnitude below generation.

## What these numbers are not

**Not a capacity forecast.** Single process, single client, no concurrency, and the local stages are measured in-process rather than over HTTP — so they exclude the transport that `ann_recall.md` measures separately at ~0.8 ms. They are good for comparing stages against each other and for the ratio above.

**The generation row was not measured in this run.** It is read from the stored Phase 5 outcomes, because the free tier is 20 requests per day and spending that budget on a latency row would cost a day of evaluation for a figure already on disk. `ragpipe bench --live-generate N` measures it fresh. The `source` column carries this so a stored figure is never mistaken for evidence about the current build.

**Cost per query is reported in tokens, not dollars**, in `reports/generation_eval.json` and on `/stats`. The free tier bills nothing and no confirmed paid per-token rate for these models was obtained; a currency figure here would be invented, and an earlier phase of this project was caught doing exactly that four times.
