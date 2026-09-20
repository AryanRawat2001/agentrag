# Serving path — what approximate search costs on this corpus

Qdrant 1.19.0 over HTTP, HNSW `m=16` `ef_construct=100`, 13,423 vectors x 384 dims, 60 queries at k=10, over **5 independent index builds**. 13,423 vectors confirmed in the index before measuring.

**Recall here is *index* recall against exact search**, not retrieval recall against ground truth. Phase 3 deliberately shipped exact cosine search so that this number could be isolated — an approximate dense row in the ablation table would otherwise conflate how good the embedding model is with how much the index gave up to be fast.

Recall and agreement are the **median across builds**, with the observed range beside them. `p50` is the median of the per-build p50s; **`p95` is the worst build's**, not a median, and neither latency column carries a range. Every per-build value is in `ann_recall.json` under `*_by_build`, so each aggregate here is re-derivable rather than merely asserted. See the reproducibility section: a single build's number is a sample.

| `hnsw_ef` | index recall@k | range | top-k identical | range | p50 ms | p95 ms |
|---:|---:|:--|---:|:--|---:|---:|
| **default (unset)** | 0.9883 | 0.9833–0.9883 | 0.9000 | 0.8500–0.9000 | 3.39 | 6.65 |
| 4 | 0.8517 | 0.8217–0.8683 | 0.3167 | 0.2833–0.3667 | 2.82 | 7.49 |
| 8 | 0.8517 | 0.8217–0.8683 | 0.3167 | 0.2833–0.3667 | 2.70 | 5.54 |
| 16 | 0.9167 | 0.9050–0.9183 | 0.5333 | 0.4833–0.5667 | 2.98 | 5.58 |
| 32 | 0.9583 | 0.9550–0.9667 | 0.7333 | 0.7167–0.7667 | 2.85 | 5.12 |
| 64 | 0.9850 | 0.9800–0.9867 | 0.9000 | 0.8333–0.9000 | 2.98 | 6.63 |
| 128 | 0.9900 | 0.9900–0.9950 | 0.9167 | 0.9000–0.9500 | 3.11 | 4.99 |
| 256 | 0.9983 | 0.9967–1.0000 | 0.9833 | 0.9667–1.0000 | 3.48 | 9.33 |
| **exact (numpy, in-process)** | 1.0000 | — | 1.0000 | — | **0.46** | 0.69 |

## The headline is a negative result

**At Qdrant's default search settings, approximation costs close to nothing on recall — median 0.9883 (0.9833–0.9883) — but it reorders the top-k more often than that suggests: only 90.0% (0.8500–0.9000) of queries come back with the identical top-10.**

An earlier single-build version of this report gave both figures as exactly 1.0000 and concluded that approximation costs nothing. Repeated rebuilds over the identical query sample do not reproduce either number, and 1.0000 sits *outside* the range above rather than at the top of it. Build randomisation (below) accounts for part of the spread but not for a value the rebuilds never reach, and the script that produced the original figure was ad-hoc and is not recoverable — which is the reason this measurement is now a command. The unexplained residual is recorded rather than explained away; the claim above is the one repeated builds support.

Below the default the curve degrades as expected, and the `top-k identical` column degrades much faster than recall — at `ef=4` recall is still 0.8517 while only 31.7% of queries return the identical top-10. That second column is the one a RAG system should care about, because rank order is what feeds fusion and reranking, and both are order-sensitive.

**And the vector database is slower than the thing it replaces.** Exact search is 0.46 ms p50 as an in-process numpy matrix product; Qdrant at its default is ~3.39 ms. But the gap is *not* HNSW being slow — a bare HTTP round trip to the same container, doing no search at all, measures **0.78 ms p50**. So the decomposition is roughly 0.78 ms of transport and ~2.61 ms of search, against 0.46 ms of doing it locally.

The conclusion is about scale, not about Qdrant: **13,423 vectors x 384 dims is a trivial matrix product**, so there is no approximation worth making and no index worth consulting over a network. This is the same shape of finding as Phase 3's "hybrid always wins is false" — the received best practice is measured and, on this corpus, it does not pay. It also vindicates the Phase 3 decision to keep the eval harness on exact search: had the ablation table used ANN, every dense and hybrid row would have carried an invisible index error for nothing.

**Read that decomposition as a bound, not a split.** The transport baseline is a `GET /` returning Qdrant's version JSON, measured on an idle server before any collection exists. The timed query is a `POST` carrying a 384-float vector and parsing 10 scored points against a live index. Request and response serialisation that genuinely belongs to transport is therefore counted in the "search" remainder — so the transport figure is a **floor** and the search figure a **ceiling**. The conclusion does not depend on the split: either way the round trip costs more than the matrix product it replaces.

## What would change the answer

Qdrant earns its place at a size this corpus does not reach. The crossover is where the matrix product stops being free — around a million vectors at 384 dims, well past 13,423. Not extrapolated here, because a number invented for a corpus that does not exist is worth less than saying so.

## Two ways this measurement silently measured nothing

Both are worth recording because both produced a full table of perfect numbers, and each looked like a successful run.

1. **Qdrant local mode brute-forces every query.** The first sweep, against `location=":memory:"`, reported index recall **1.0000 at every `ef` from 4 to 256** — it was comparing exact search against exact search. Qdrant warns about this; the warning was nearly scrolled past. `sweep_recall` now refuses to run against a local client.
2. **A server will not index a small collection.** `indexing_threshold` defaults to 20,000 KB, and this corpus is about 20 MB across several segments, so a default collection would also have answered every query exactly. The store now forces indexing *and* `assert_indexed` blocks until Qdrant confirms the vector count, because the setting working is a claim and the count is evidence.

One residual caveat, stated rather than resolved: the collection's `full_scan_threshold` is 10000 KB, so Qdrant may still full-scan individual segments below that size, and segment layout is decided per build. `hnsw_ef` demonstrably changes recall, which proves HNSW is being consulted, and full recall is *not* reached at the default — it takes `ef=256` to reach 0.9983. A larger corpus, above the threshold in every segment, would remove this as a variable.

## Reproducibility: the variance is in the build, not the search

Worth stating precisely, because the first version of this report got the cause wrong and drew the wrong boundary around it. It said search order varies run to run and that therefore only the *low end* of the curve should be read loosely. Both halves were wrong.

A controlled run separated the two candidates: sweep twice over one built index, then rebuild and sweep again. **Two sweeps over the same index are bit-identical** at every `ef` — search is deterministic. Rebuilding the same 13,423 vectors is not, and the spread is not confined to low `ef`: the widest recall band in the table above is 0.8217–0.8683 at `ef=4`. HNSW graph construction is randomised; the search over it is not.

So the variance is not noise to be averaged away at the low end — it reaches the top of the curve, and it is what produced the original "costs nothing" headline. Recall and agreement above are medians over 5 independent builds with the range shown, which is the smallest honest way to report a number that depends on a random graph. (An earlier version of this sentence claimed *every* column was a median with a range; two of them are not, and the caveat above now says which.)

The determinism check is re-run every time this report is generated rather than remembered: this run confirmed two sweeps over one build agree exactly.

Latency figures are a floor, not a forecast: single client, no concurrency, container on the same host. They are good for comparing settings against each other and for the transport-versus-search decomposition above, and not for capacity planning.
