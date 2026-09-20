# Hybrid-search RAG with verified citations — case study

**What it is.** A retrieval-augmented question-answering system over public FDA guidance
documents and ClinicalTrials.gov protocols — 160 fetched and sha256-pinned,
155 with a usable text layer, **152 indexed** after three were
quarantined for unusable extracted text — two with corrupted glyph encoding and one whose
characters came out space-separated. Built to answer one question honestly:
*can you trust what the model just told you?* Every quote in an answer is located in the
source document before the user sees it, and a quote that cannot be found is reported as
unfound rather than dropped.

**Why it exists.** I wanted to know whether the standard RAG advice holds up when you
measure it. It mostly does not, and the interesting part of the project is the measuring.

---

## The headline: "hybrid retrieval beats dense" is the wrong framing

The received wisdom is that combining keyword and embedding search beats either alone. On
this corpus, over 240 probe queries with character-level ground truth and 192
retriever/slice configurations:

| slice | BM25 | dense | naive RRF | min-max @ 0.1 dense |
|---|---:|---:|---:|---:|
| exact identifier, `recall@10` | **0.9508** | 0.0222 | 0.2019 | 0.9475 |
| section lookup, `recall@10` | 0.8282 | 0.7385 | 0.7972 | **0.8365** |

Three findings, in order of how much they surprised me:

**1. Dense retrieval is close to useless on exact identifiers** — **43×**
behind keyword search on the shipped `structural` chunking, and up to **75×**
on `fixed`. (The table above is `structural`, so its own two numbers give the first figure;
the larger one is the worst case across chunkings, and quoting it beside a `structural`
table without saying so would be sleight of hand.) Asked for `21 CFR 314.50` or `NCT02942264`,
a sentence embedding maps the identifier into a region of space shared by every other
identifier, because the token is rare and carries no distributional meaning. On a
regulatory corpus, that is not an edge case; it is what people search for.

**2. Naive fusion is *worse than sparse alone* on that slice** —
0.2019 against
0.9508. Reciprocal rank fusion rewards
agreement between retrievers, so a retriever with no signal does not abstain: it votes,
and it outvotes a correct top hit. This is the finding I would have missed entirely by
reading benchmarks instead of running one.

**3. The fusion config that works is a near-wash, and that is the honest way to say it.**
Min-max normalisation at a dense weight of 0.1 costs **-0.0033**
on identifiers and gains **+0.0083**
on section lookup. That is a trade, not a win, and the shipped service uses **BM25 alone** —
because BM25 also has the best refusal separability of any first stage measured, and a
score gate calibrates most cleanly against it.

---

## What I actually built

**Eval first.** The golden retrieval set came before any retriever. Ground truth is
**character spans, not chunk ids**, so the same 240 queries stay valid across re-chunking
and every strategy comparison is honest rather than circular.

**Quote-and-verify citations.** The system does not trust a model's claim about where its
quote came from. It takes the quoted text and **searches for it** in the cited document,
reporting one of six outcomes — `exact`, `normalized`, `line_number_ambiguous`,
`wrong_chunk`, `unverified`, `too_short`. Provider-independent, and strictly stronger than
trusting returned offsets.

**A two-tier verifier.** Tier 1 asks "is this quote real?" Tier 2 asks "does it support
the claim?", and the judge is calibrated against **constructed negatives** — a real answer
paired with another document's evidence, which cannot be supported under any reading. A
judge that answers "supported" to everything scores perfectly on positives; only the
negatives catch it.

**A served API and dashboard.** FastAPI, containerised, with the model's quote
**word-diffed against the document** so a reader can see a one-token disagreement — `40 CFR`
against `21 CFR`, `may` against `shall` — rather than reading a boolean.

Measured per stage, from `reports/serving_bench.md`:

| stage | n | p50 |
|---|---:|---:|
| retrieve (BM25 over 13,423 passages) | 540 | < 0.2 ms |
| verify (locate every quote) | 24 | 0.006 ms |
| generate (third-party API) | 16 | 12.5 s |

The two local rows are bounds, not point figures: they are wall-clock timings whose samples
the payload does not retain, so they move a few percent per run and a three-decimal reading
is false precision. `verify` is quoted exactly because 0.006 ms survives a re-run at that
precision; `generate` is exact because it is read from 16 stored per-call latencies.

Generation is roughly **five orders of magnitude** more expensive than anything this
project controls — see `reports/serving_bench.md`, which computes the exponent rather than
asserting it. "The verifier is free" is a measurement, not a claim.

---

## The part I would put on a whiteboard

Every phase passed **two independent verification gates** before being called done: a code
review that reads the source and exercises the functions, and a numerical audit that
recomputes every reported figure from raw data *without importing the code that produced
it*. Findings were never taken at face value — each one was reproduced before being acted
on, and false positives were logged.

Across those gates, the record is:

- **Zero arithmetic errors in any generated artifact**, across roughly a thousand
  recomputed figures. One *hand-written* statistic was wrong — a Spearman correlation
  published as 0.917 when it is 0.948, which turned out to be a table cell transcribed as
  a correlation. The audit gate caught it. The distinction is the point: figures a command
  produces have held up; figures a human typed into prose have not, twice.
- **Every real defect was a claim, a check that ran in the wrong place, or a check that
  never ran at all.**

That pattern held so consistently it became the design principle. A few examples:

- A **citation tier I added verified a change in meaning**: it scored a model's "40 CFR" as
  a verified citation of a document's "21 CFR", because **53.2%** of
  chunks lose digits to line-number stripping. It survived my own review because I tested
  it on the case it was built for and never on the case it would break.
- Three golden-set validation checks had **zero call sites** for an entire phase while the
  documentation described them as protection.
- A figure of `0.136` turned out to measure the *judging unit*, not the model — per-citation
  judging returned "partial" for 18 of 22 citations, which is the wrong question when an
  answer cites several. Answer-level judging gave 0.875.
- A report divided answers by citations and printed `7/22 = 0.318` as "the share of
  citations that survive reading".
- The ANN measurement reported **recall 1.0000 at every setting** — twice, for two different
  reasons (Qdrant local mode brute-forces; a server will not index a small collection). Both
  produced a full table of perfect numbers and looked like successful runs.
- Making that measurement reproducible **cost it its headline**: one index build reported
  1.0000, five builds report 0.9883. HNSW construction is
  randomised, so a single build's recall is a sample.

---

## What it cost, and what it says no to

**The vector database does not pay at this scale.** Qdrant over HTTP is **~7.4×** slower
than the in-process matrix product it replaces (3.39 ms against
0.46 ms p50), and at 13,423 × 384 there is no approximation worth
making. It stays in the stack because containerising the serving path is a deliverable —
but as a *measured* trade, not an assumed win.

**Cost is reported in tokens, not dollars.** The free tier bills nothing and no confirmed
paid per-token rate was obtainable, so a currency figure would be invented. An earlier phase
of this project was caught doing exactly that four times.

**The generation sample is n=16** because the free tier allows 20 requests per
day. That ceiling is stated everywhere a generation figure appears, and two real failure
modes — a `wrong_chunk` citation and a production near-miss — are labelled **anecdotes**
rather than rates, because they were seen live and appear in no artifact.

---

## Where it fails

Five failure modes, each with a figure, in `reports/failure_modes.md`. The one I find most
instructive: **8 of the golden set's reference answers had
manufactured a legal obligation** — restating a source's "should" as "must", including a
pregnancy-discontinuation instruction and a drug-storage temperature. All of them passed
five machine validation checks, because the last of those is answer/evidence word coverage,
and a bag of words cannot tell `may` from `shall` any more than `60 days` from `30 days`.

Finding them needed detectors aimed at that specific blind spot, then reading every pair
they flagged against its source. 151 pairs survive, and the human verdicts are
committed — because a judgement about whether an answer misstates a regulation is the one
artifact here that cannot be regenerated, only redone.

---

## What I would do differently

1. **Write the artifact before the prose, every time.** Three corpus statistics were quoted
   in live source files from before a corpus regeneration. Every conclusion survived; none of
   the arithmetic did. Nothing caught it because the figures had never lived in a
   machine-readable file. The reports now derive every number from a payload, with mutation
   tests that fail if a figure survives replacing the inputs.
2. **Choose the primary metric by checking it can move.** A slice was written off as
   "saturated and no longer discriminating" on the basis of one metric at one depth. The
   slice was fine; `hit@10` was saturated because the queries are verbatim titles. `hit@1`
   spans 0.85–0.97 over the same runs.
3. **Budget for the quota, not the tokens.** The free tier counts *requests*, not tokens,
   against a 1M-token context. Assuming one item per call made a two-hundred-pair job look
   like a week; batching ten per call made it one session.
4. **Test the branch beside the one you ran.** A shipped helper died on `set -u` with an
   empty array under macOS bash 3.2 — the default path worked, the flag beside it did not,
   and the failure was a single silent line.

---

## Stack

Python 3.11 · `uv` · `bm25s` · `sentence-transformers` (`bge-small-en-v1.5`) ·
cross-encoder reranking · Qdrant · FastAPI · Docker/Compose · Gemini via raw `httpx` ·
`pytest` (**1,118 tests**) · `ruff`

Artifacts: 22 files in `reports/`, every one
regenerable by a command.
