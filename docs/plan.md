# Plan of record

The build plan, the decisions behind it, and the process that gates each phase.
This is the committed source of truth; `docs/progress.md` is the running log of
what has actually shipped.

## Goal

A production-grade RAG system over public regulatory documentation: dense **and**
sparse retrieval, rank fusion, cross-encoder reranking, and answers whose
citations are **verified against source text** rather than merely emitted.

The project is a job-search artifact. Success is not "it runs" — it is "the
retrieval ablation table and the failure analysis survive interview scrutiny."
That framing decides every tradeoff below: anything that does not end up as a
number, a table, or a documented failure mode is a candidate for cutting.

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Corpus | Public FDA guidance + ClinicalTrials.gov protocols | Acronym- and identifier-dense regulatory text is where hybrid beats pure dense; metadata gives free version, ACL, and temporal slices |
| Generation | **Behind a provider interface. Gemini 2.5 Flash first (free tier); Claude when access exists** | Revised in Phase 5. The original reason for Claude was native citations returning character-level offsets, which make tier-1 verification a deterministic span check. That reason still holds and Claude remains the preferred generator — but organisation policy blocks first-party access and the available Bedrock role permits only models AWS has retired, so the project cannot depend on it. See deviation #14: verification moves from *trusting* returned offsets to *independently locating* quoted spans, which works with any model and is arguably the stronger check |
| Embeddings | BGE-M3 (dense + learned sparse), local, behind a provider interface. **Validated on `bge-small-en-v1.5` first** | Free re-embedding is what makes the chunking and embedding-model sweeps actually happen; a hosted API is benchmarked as one ablation row. Sequencing revised in Phase 3: BGE-M3 is 568M parameters, so validating the pipeline on a 33M model first turns a blocking hour-long run into a minutes-long one and makes model size an extra ablation row. Same destination, cheaper path |
| Reranker | Local cross-encoder | Highest-leverage single component in production RAG; absent from the source guide's stack table |
| Vector store | Qdrant, **for the serving path only** | Named dense **and** sparse vectors with server-side fusion — hybrid lives in one store instead of a hand-joined second index. **Measured in Phase 7 and it does not pay at this scale:** at default settings approximation costs about a point of index recall (0.9883, median of five builds) but reorders the top-k for roughly one query in ten, and the round trip is ~7.4x slower than an in-process matrix product (3.39 ms vs 0.46 ms p50), of which ~0.78 ms is transport. Kept because it is the serving path and the containerisation is a deliverable, not because it is faster. The eval harness stays on exact search — see `reports/ann_recall.md` |
| Sparse | `bm25s` + BGE-M3 learned sparse | `bm25s` is far faster than `rank_bm25`; having both enables a lexical-vs-learned-sparse ablation almost nobody runs |
| Chunk size | **1,024 characters** (~256 tokens), 15% overlap | Revised in Phase 4 from 2,048. **A trade, not a domination** — across all denominator-free metrics 2,048 actually wins 22 comparisons to 1,024's 11, and `mrr@10` prefers 2,048 on 8 of 12 slice×retriever pairs. 1,024 is chosen on two grounds that hold: truncation 10.4% → 0.6% (a correctness defect, not a preference) and +0.042 refusal separability, which Phase 5 depends on. The cost is rank-1 precision. Revisit if Phase 5 needs `hit@1` more than the refusal threshold |
| Pace | Quality over speed, with per-phase checkpoints | No deadline, but checkpoints are the guardrail against drift |

## Deviations from the source build guide

The source guide (`docs/source-guide.md`, local only) is a reasonable spine. It
does include a reranker and an eval framework — its accompanying tech-stack table
omits both. Where we depart from it:

| # | Guide says | We do | Why |
|---|---|---|---|
| 1 | All evaluation in Phase 4, after generation | **Split evaluation in two.** Retrieval metrics (Recall@k, MRR, nDCG) before the retriever; generation metrics after | The guide's ordering commits to chunk strategy, `k`, fusion weights, and rerank cutoff with no ruler, then builds the ruler and asks for a chunking comparison — an implicit rework loop. Retrieval metrics need only `question → relevant chunk IDs`: no LLM, no cost, seconds to run, so they can gate every change |
| 2 | "Implement RRF … make the weighting configurable (0.7 dense / 0.3 sparse)" | Implement **vanilla RRF, weighted RRF, and min-max weighted score fusion** as separate ablation rows | These are different methods. Vanilla RRF has no per-list weights — scale-free and tuning-free is the entire point; its only knob is the rank constant. What the guide describes is weighted score fusion. Conflating them is a visible tell |
| 3 | Bracketed `[1]` citations plus a structured response envelope | **Native citations; assemble the JSON envelope in application code** | Claude's `citations` is incompatible with `output_config.format` (returns 400). Choosing native citations buys character-level spans, which makes verification two-tier: a deterministic span-existence check, then an LLM judge only for whether the span supports the claim |
| 4 | Skip near-duplicate chunks (cosine > 0.95) | **Cluster near-duplicates, keep one canonical chunk carrying references to every source location** | Regulatory boilerplate is legitimately near-identical across documents. Silently dropping it destroys the ability to answer "which documents impose this requirement?" and damages provenance |
| 5 | 50 golden Q&A pairs, hand-written | **150–200 pairs, LLM-drafted and human-curated** | Across five slices, 50 pairs is 10 per slice; one flipped example swings a slice by 10 points. Too noisy to make decisions on |
| 6 | Cross-encoder "or LLM-as-judge" for reranking | **Cross-encoder as default**; LLM rerank at most one comparison row | Milliseconds and free versus seconds and metered. Not interchangeable |
| 7 | (absent) | **Contextual retrieval** — LLM-written context blurb prepended per chunk before embedding | Measurably reduces retrieval-failure rate; the clearest signal of having read past the quickstart. Prompt caching + Batch API keep it cheap |
| 8 | (absent) | **Per-stage p50/p95 latency and cost per query** | "Production-grade" claims need these; another table, nearly free |
| 9 | 14-day schedule | Phase checkpoints, no dates | The schedule is *why* the guide compresses eval into two days and 50 questions |
| 10 | Hybrid retrieval as the destination | **Hybrid is measured per slice, and rejected where it loses.** Fusion weights are reported as a sweep, not fixed | Added in Phase 3 from a measurement, not a preference. Fusing dense into BM25 dropped `exact_identifier` recall@10 from 0.992 to 0.171 — RRF rewards agreement between lists, so a retriever with no signal on a query type contributes noise that outranks a correct top hit. "Hybrid always wins" is the claim the project set out to test, and on one slice it is false |
| 11 | (absent) | **Report context-window truncation as a retrieval metric** | Text past the window is never embedded and therefore never retrievable, but it fails as mediocre numbers rather than as an error. At 512 tokens this corpus lost 16.8% of tokens before normalization |
| 12 | "Experiment with chunk sizes (512/1024/2048 tokens)" as one task among many | **Sweep chunk size, report the denominator trap explicitly, and state the result as a trade-off** | Phase 4 found `recall@k` rises with chunk size for mechanical reasons — larger chunks mean fewer chunks per section, shrinking recall's denominator — while `hit@k` is flat. Reading recall alone concludes "bigger is better". But the correction has its own trap: reading only the metrics that favour the new answer concludes "smaller dominates", which the audit refuted 22 times. The defensible output of a sweep is a stated trade with a named reason for the pick, not a winner |
| 13 | One API call per chunk for contextual retrieval | **Group ~40 chunks per call behind a cached document prefix** | The per-chunk recipe's 8x saving depends on cache hits that batch parallelism undermines — a cache entry is only readable once the first response begins, so N concurrent requests sharing a prefix can all miss. Real cost sits in an uncontrolled $33–$268 band; grouping is $6 and makes caching a bonus rather than load-bearing |

| 14 | (absent — the guide uses bracketed `[1]` markers) | **Quote-and-verify citations, provider-independent** | Deviation #3 chose Claude's native citations for character-level offsets. With no Claude access available, tier 1 changes from "check the model's offsets" to "locate the model's quoted span in the source by exact search". The two-tier verifier is otherwise unchanged, and chunk `text` is already a byte-exact source slice, so the span is locatable. Two consequences worth stating: the check no longer trusts anything the model reports about position, which is *stronger*; and a model not trained to emit verbatim quotes will paraphrase more, so the verification **failure rate** rises — which becomes its own measurable row, "citation-verification failure rate by generator", rather than a hidden cost |

| 15 | 50 hand-written pairs, one at a time | **Draft ~10 golden Q&A pairs per API call, grounded in one document's chunks** | Same reasoning as deviation #13, applied to a different budget. The free-tier quota is **20 requests per day per model** and is counted per *request*, not per token, against a 1M-token context. One pair per call makes a 200-pair set a multi-day job; ten pairs per call makes it ~20 requests, i.e. one session. An earlier estimate in this project treated the two as equivalent and concluded Phase 6 would take a week — wrong for the same reason the contextual-retrieval cost estimate was wrong before grouping. The cost is that batched drafting tends toward repetitive questions, so curation samples across batches rather than reading the first N |

| 16 | Ship a vector database because that is what RAG stacks do | **Measure what the vector store costs before crediting it** | Phase 3 deliberately kept the eval harness on exact cosine search so that ANN recall loss could be isolated later rather than folded invisibly into every dense and hybrid row. Phase 7 collected it: on this corpus approximation costs **about a point of recall** at default settings while **reordering the top-k for roughly one query in ten**, and the network round trip makes the store **~7.4x slower** than the matrix product it replaces (~0.78 ms of that is transport, read as a floor rather than a partition). The decision stands for the serving path — containerisation is a deliverable and the store earns its place at a scale this corpus does not reach — but it stands as a *measured* trade rather than an assumed win. Same shape as deviation #10: the received best practice, tested, and on this corpus false |

| 17 | Measure ANN recall once and report the number | **Report a randomised measurement as a band over independent rebuilds, and re-check the determinism assumption every run** | HNSW graph construction is randomised, so index recall at a given `ef` is a property of the build, not of the index configuration. Phase 7a built once and reported recall and top-k agreement of exactly **1.0000** at Qdrant's default, concluding approximation costs nothing; making the same measurement reproducible as `ragpipe ann` reproduced neither figure. A controlled run then showed two sweeps over one built index are *bit-identical* — search is deterministic, the variance is construction — which also falsified the report's own caveat, that segment search order varies and only low `ef` should be read loosely. The measurement now defaults to five builds, reports a median with the observed range, **persists every per-build value** so each aggregate is re-derivable rather than asserted, and re-runs the within-build determinism check on every invocation — comparing the ranked chunk ids themselves, since the first version of that check compared only mean recall and an exact-match count, both invariant under reordering, while the report claimed "bit-identical". Sibling of #16: that deviation says measure the received best practice, this one says measure it enough times to know whether you measured it |

| 18 | Judge a slice by whichever metric was picked for it first | **Check that a slice's primary metric can actually move before concluding anything from it** | `title_lookup` shipped with `hit@10`, which is saturated on this corpus: the queries are verbatim document titles, so 47 of 48 configurations scored exactly 1.0000. Three documents then carried the conclusion "the slice is saturated and no longer discriminates — it should be hardened or dropped". Measured instead: `hit@1` spans 0.8500–0.9667 over the same runs (8 distinct values, 3.0 binomial standard errors at n=60) and `mrr@10` spans 0.9218–0.9833. The **slice** discriminates; the **metric** was saturated. Finding the right document *somewhere* in the top 10 given its exact title is trivial — ranking it *first* is not. Acting on the original note would have deleted a working measurement, and its alternative remedy (shorten or paraphrase the queries) would have changed the ground truth to fix a reporting choice. `hit@1` over `mrr@10` because the two order all 48 configurations near-identically (ρ 0.948, same top five), so interpretability costs nothing. A generic test now fails any slice whose primary metric takes fewer than two distinct values |

| 19 | Curate a reference set by reading a sample of it | **Build detectors for the defect classes your validation is known to be blind to, then read every pair they flag** | The golden set passed five machine checks, the last of which is answer/evidence lexical *coverage* — a bag of words. Phase 6 measured exactly what that misses: `60 days` against `30 days`, `may` against `shall`, and `shall not submit` against `shall submit` all score coverage **1.000**. Numbers were already handled; obligation level and polarity were not, and in regulatory text they are the whole meaning. So Phase 8c added modal-strength, polarity, and attribution detectors, flagged **24 of 159** pairs, and read every one against its source. **8 were rejected** — all restating a recommendation as a requirement (`should` → `must`), including a pregnancy-discontinuation instruction and a drug-storage temperature. A random sample of 24 would have found roughly one of them. The detectors are triage, never verdicts: every rejection is a human judgement recorded with its reasoning in `corpus/curation_verdicts.json`, the one artifact here that cannot be regenerated, only redone |

| 20 | Write the failure analysis as prose | **Generate it from the artifacts, and fail the build if any figure is not derived from one** | Phase 8 asks for five named failure modes, and the natural way to produce them is to write down what you remember measuring. Doing that instead as a renderer over committed payloads — with a mutation test that fails if any number in the prose survives replacing the inputs with sentinels — immediately exposed **three statistics quoted from the pre-quarantine corpus**: the line-number exposure in `citations.py` and `dashboard.py` ("7,229 of 13,706 chunks (52.7%)", now 7,141 of 13,423), the entire contextual-retrieval cost table in `contextual.py` (~1.5% high throughout), and a heading-overlap figure in `retrieval.py` that is **labelled superseded rather than replaced**, because the original measurement's definition is not recorded and inventing one would produce a number that looks like a correction while measuring something else. Every conclusion survived; none of the arithmetic did; nothing would have caught any of it, because all three lived only in prose. Failure modes observed live but absent from every artifact (`wrong_chunk`, production `line_number_ambiguous`) are labelled anecdotes with the cause of the small sample named, rather than promoted to rates |

## Phases

Each phase ends with a named artifact. A phase is not done until its artifact
exists and the verification gate below has passed.

| # | Phase | Checkpoint artifact | Status |
|---|---|---|---|
| 0 | Corpus acquisition + characterisation | `reports/corpus_report.md` | **shipped** — both gates passed |
| 1 | Loader, chunking strategies, content-hash incremental indexing, duplicate clustering | `reports/extraction_report.md`, `reports/chunking_report.md` | **shipped** — both gates passed |
| 2 | **Retrieval eval harness** + golden retrieval set | `reports/retrieval_eval.md`, BM25 baseline | **shipped** — both gates passed |
| 3 | Dense index, fusion variants, cross-encoder rerank, semantic chunking | **Ablation table v1** | **shipped** — both gates passed |
| 4 | Contextual retrieval + chunk-size sweep | Ablation table v2, `reports/chunk_size_sweep.md` | **shipped** — both gates passed. Contextual retrieval built + tested but not run (blocked on API access) |
| 5 | Grounded generation, **quote-and-verify** citations, refusal path | `reports/generation_eval.md` — citation verification buckets, refusal rate, separability | **shipped** — both gates passed (23 findings, all reproduced and fixed). Generator switched to Gemini free tier (deviation #14). Tier 2 (LLM judge) deferred |
| 6 | Generation eval + golden answer set + **tier 2 of the verifier** | `reports/generation_eval.md` — tier 1 22/22 located, tier 2 **7/8 answers fully supported (0.875)**, refusal 4/4 on unanswerables | **shipped** — both gates passed (28 findings, all reproduced and fixed). n=16 queries (free-tier daily cap); tier-2 figure is a range 0.75–0.875 across re-runs. Golden set machine-validated and, in Phase 8c, **curated**: 151 accepted after 8 pairs were rejected by human reading for misstating obligation level (deviation #19) |
| 7 | FastAPI service, query dashboard, Docker, **Qdrant serving path** | `reports/ann_recall.md` (done), running demo, per-stage p50/p95, cost per query | **both gates passed.** All deliverables exist and have been exercised end to end through the container: `ragpipe serve` + dashboard at `/`, `ragpipe ann`, `docker compose`. Per-stage timing on `/stats`; cost in tokens, null in dollars by design. Making the ANN measurement reproducible cost its headline — see deviation #17. **Sparse vectors in Qdrant were unimplemented at the time**, so "hybrid lives in one store" was not yet true — closed in Phase 9 (deviation #21). 22 findings across the two gates (8 numerical, 14 code review), all reproduced before acting, all fixed; **zero arithmetic errors** across 227 recomputed figures |
| 8 | Failure analysis, README, demo video, case study | README opening with the table; five named failure modes | **shipped, except the recording.** 8a: `title_lookup`'s primary metric moved `hit@10` → `hit@1` — the slice was never saturated, the metric was (deviation #18). 8b: `ragpipe bench` → `reports/serving_bench.{md,json}`, closing the per-stage latency artifact gap for deviation #8 without spending quota. 8c: golden set **curated** — 24 flagged, all read, 8 rejected for misstated obligation level (deviation #19), verdicts committed. 8d: **five named failure modes** → `reports/failure_modes.{md,json}` + README section, every figure artifact-derived (deviation #20) — which exposed three corpus statistics quoted from before the post-quarantine regeneration. Case study written. `bin/build-demo` renders the video from the live service — headless-Chrome frames, subtitles generated from the same strings, and a probe that refuses to build when the day's quota is spent, when a per-minute rate limit is mistaken for one, or when not one quote could be located; 26 tests, and its narration figures are read from `/health` and `failure_modes.json` rather than typed (two were wrong as literals). **Both gates passed in two rounds** — 36 findings, then 27 more against the fixed code, because five of the round-one fixes had introduced new defects and no round-one correction had propagated completely. `metrics.py` went 29% → 100% coverage; `separability` had none at all. **The recording itself is deferred to a fresh free-tier quota** — a calendar constraint, not an incomplete deliverable |
| 9 | **Distribution** — publish to GitHub, add to the personal site (`~/Aryan_Website`, Vercel auto-deploy), and rework the resume (`~/Resume Builder`) | Public repo, live project page, resume bullets leading with measured results | |

Phase 9 exists because the artifact only pays off once it is visible in the three
places a recruiter looks. It starts after Phase 8, not before — publishing a
half-built system is worse than publishing nothing. Both the GitHub push and the
website push are outward-facing (the latter is a production deploy), so each needs
explicit confirmation, and the repo must be audited first to ensure
`docs/source-guide.md` and `data/` are excluded.

## Planned eval slices

Each requires a corpus property. Phase 0 confirmed which are viable — see
`reports/corpus_report.md`.

| Slice | Tests | Corpus precondition |
|---|---|---|
| Straightforward lookup | Baseline retrieval | Any |
| Exact identifier | Where BM25 beats dense outright — **currently 0.955 vs 0.013 recall@10 on `fixed` (74.8x) and 0.951 vs 0.022 on `structural` (42.8x)**. Phase 3 first measured this as 0.992 vs 0.032 (31x) at the 2,048-char target and on the pre-quarantine golden set; the gap has widened at every re-measurement, and the ratio is sensitive to both, so it should always be quoted with its chunking | Docket / registry IDs, CFR citations, ICH codes present **in body text** |
| Version currency | "Which revision is authoritative?" | Draft *and* Final of the same guidance |
| Multi-hop | Combining two documents | Cross-referencing documents |
| Unanswerable | Refusal instead of fabrication | Questions deliberately outside the corpus |
| Table lookup | Extraction of tabular content | Schedule-of-assessment tables |
| Temporal | Superseded requirements | Issue dates spanning decades |
| ACL-filtered | Permission-scoped retrieval | Real document-level partition (issuing center) |

**Four of these shipped**, under the names the harness uses: `exact_identifier`,
`section_lookup` (the "straightforward lookup" row), `title_lookup`, and
`unanswerable`. The rest remain viable-but-unbuilt and are candidates for Phase 8 if
the failure analysis wants them.

**`title_lookup` was not saturated — its primary metric was. Resolved in Phase 8.**
This entry used to read "the slice is saturated and no longer discriminates... a slice
nothing can lose on measures nothing, so it either needs hardening or removal from the
headline table". Checking it against the stored report rather than re-reading the note:
`hit@10` takes **2** distinct values over 48 configurations (47 of them exactly 1.0000),
while `hit@1` spans **0.8500–0.9667** — 8 distinct values, and 3.0 binomial standard
errors at n=60. The queries are verbatim document titles, so retrieving the right
document *somewhere* in the top 10 is trivial; ranking it *first* is not.

The primary metric moved to `hit@1` and the slice was kept. `mrr@10` orders all 48
configurations near-identically (Spearman ρ 0.948, identical top five), so `hit@1` was
chosen for interpretability at no cost in ordering. No retrieval was re-run: every
metric was already stored for every run, so this is a re-summary and
`reports/retrieval_eval.json` records that in `primary_metric_note`.

**The lesson is the one this project keeps relearning:** the note was a *conclusion about
the data* ("nothing can lose here") written from a single summary statistic, and acting on
it would have deleted a working measurement. Hardening the queries — the other option on
the table — would have been worse still: it would have changed the ground truth to fix a
reporting choice.

## Process

### Stepwise build

One phase at a time. No phase starts before the previous phase's artifact exists
and its verification gate has passed. Within a phase, work is tracked as tasks so
partial progress is visible rather than implied.

### Verification gate

Every phase is reviewed by independent subagents before being marked shipped.
Two standing mandates, run in parallel:

1. **Code review** — read the phase's source, exercise its pure functions with
   edge-case inputs, and check that stated rationales in comments are actually
   true of the implementation. A comment claiming a safeguard that does not work
   is a reportable defect.
2. **Numerical audit** — recompute every figure in the phase's report
   independently, from raw data, *without* importing the code that produced it.
   Circular verification is worthless.

Phases 2, 3, and 5 get a stricter gate than the rest: a silent bug in the eval
harness, the fusion layer, or the citation verifier corrupts every number
downstream of it, and the failure mode is a plausible-looking table rather than
a crash.

Subagent findings are **not** taken at face value. Each is reproduced before
being acted on; false positives are recorded as such in `docs/progress.md`.

### Documentation

- `docs/plan.md` (this file) — decisions and phase definitions. Updated when a
  decision changes, with the reason.
- `docs/progress.md` — append-only log. One entry per checkpoint: what shipped,
  real numbers, what verification found, what was deferred.
- `reports/` — generated artifacts, committed. These are the portfolio evidence.
- Corrections are logged, not overwritten. A wrong claim that was caught and
  fixed is more credible than a history with no mistakes in it.
