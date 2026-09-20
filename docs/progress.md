# Progress log

Append-only. One entry per checkpoint. Corrections are logged rather than
overwritten — a record that shows mistakes caught and fixed is more credible
than one with no mistakes in it.

---

## The chunk-size sweep, regenerated — the stale corpus was hiding a stronger conclusion

2026-09-20, after the repo went public. `reports/chunk_size_sweep.md` was the one artifact
still carrying the **pre-quarantine corpus** (13,706 chunks over 154 documents against the
current 13,423 over 152). Phase 8d had left it deliberately: it is a dated snapshot of a
run, so hand-editing its table would forge a measurement, and regenerating is a ~25-minute
`make sweep`. That was defensible while the repo was private. Once it was public, a reader
cross-checking 13,706 against the 13,423 in every other artifact just finds a contradiction.

Regenerating took three attempts, and the first two diagnoses were wrong:

1. `PermissionError ... downloading BAAI/bge-small-en-v1.5. Check cache directory
   permissions.` I read this as the sandbox blocking the HF cache. It was not: the model
   was already cached and the directory was writable.
2. So I forced `HF_HUB_OFFLINE=1`, which also failed — and the traceback named
   `httpx/_config.py create_ssl_context`. That is the documented certifi signature: `*.pem`
   is on the sandbox read-deny list, so the SSL context cannot be built **at import**,
   before any download decision is reached. Offline mode could never have helped.

The lesson is the ordering: an error that *names* a cache is not evidence about a cache.
Both wrong turns came from reading the message instead of the traceback's bottom frame.

### What the regeneration changed

The corpus-shape table moved as expected (13,706 → 13,423 at 1,024). Two things did not:

* **`title_lookup`'s primary metric came back as `hit@1`, not `hit@10`.** That is not the
  corpus — it is deviation #18 from Phase 8a, which the old snapshot predated. So the
  report had been carrying a metric the rest of the repo had already retired.
* **The README's chunk-size table was built on the old run**, and five of its figures were
  wrong. Corrected against the new JSON rather than edited by hand:

| claim | was | now |
|---|---|---|
| section `hit@10` across sizes | "flat at 0.966 from 1,024 onward" | **declines**, 0.967 → 0.917 |
| section `precision@10` | 0.236 → 0.109 | 0.248 → 0.100 |
| section `recall@10` | 0.809 → 0.932 | 0.753 → 0.900 |
| truncation at 1,024 | 0.6% | **0.03%** |
| 1,024 vs 2,048, denominator-free | 2,048 wins 22 to 11 | **2,048 wins 26 to 4** |

**Every conclusion survived, and two got stronger.** The argument for 1,024 was always
truncation plus refusal separability, and truncation at 1,024 is now essentially zero
(0.03% against 2,048's 9.81%). The argument that `recall@10` rises only because its
denominator shrinks is now *demonstrated* rather than inferred: the denominator-free
`hit@10` actively declines with chunk size instead of staying flat.

One new trap, and it is the same one this section already warns about one size down. The
five-figure table now favours **512** on four of five. Across the full denominator-free set
512 and 1,024 are a coin flip — 20 wins to 21, 19 ties — and 512 costs 63% more chunks to
index. Reading five favourable cells as a finding is exactly the cherry-picking the
numerical audit caught the first time, so the README now says so explicitly.

Suite: **1116 passed, 2 skipped**, unchanged.

---

## The demo video, first real build — two defects the frames showed and the tests did not

2026-09-20. Quota refreshed, so `bin/build-demo` ran end to end for the first time:
102 s, 2.7 MB, 22 cues. Then I watched the frames, and it was not shippable.

**The container was two days stale**, built before `/health` gained its `docs` field.
`narrate.py` fails loudly on a missing health field rather than narrating a stale number,
so this surfaced immediately. The guard worked; noted because it is the cheap kind of
failure that a "just re-run it" habit hides.

### Scene 3 rendered 87% empty

The frame whose whole job is to show the verification panes was black, with the citations
card just entering at the bottom edge, under narration describing the diff.

I was wrong about the cause three times — scroll ordering, then `scroll-behavior: smooth`,
then a non-instant scroll — and rebuilt the container for each. The frame never changed.
Pixel forensics did not settle it either. Driving Chrome over CDP did, in one call: at
capture time the page reports `scrollY: 610` and `citeTopViewport: 0`. **The deep link was
correct all along.** Headless Chrome's `--screenshot` renders from the document origin and
ignores a programmatic scroll; `--headless=new` honours it and never exits under
`--virtual-time-budget`, so it is not usable here.

The fix is not a scroll. The page publishes the card's document offset on
`body[data-cite-top]` and the build crops to it. A screenshot tool cannot ask a layout
engine where a card is, and a hardcoded offset against a page whose height depends on the
answer is exactly the guess the `#citations` deep link was added to remove. `--screenshot`
and `--dump-dom` run in one invocation, so the geometry and the pixels come from a single
page load — one generation request, which is the scarce resource at 20 per day.

Worth recording: two of the three wrong fixes were plausible enough to keep. I reverted
both, because a comment claiming a change fixed something it did not is the same defect
this log keeps finding, one level down.

### The captions covered the finding

Burned-in subtitles sat over the picture. On the architecture frame they covered the
EVALUATE lane — `192 configurations`, `the finding`, `43–75×` — the most important number
in the video, obscured in the frame built to show it.

The mechanism: ffmpeg renders an SRT through a 384×288 ASS canvas, so every style value is
scaled by 2.5 on the way to 720p. `MarginV=42` was 105 real pixels. The build now pins the
canvas to the video's size and **refuses rather than defaulting** if the header is absent,
because without a known canvas the style units are an unknown multiple of a pixel and the
failure mode is silent. Frames are scaled into a 624 px content area, top-aligned, with the
bottom 96 px reserved. Nothing can be covered.

### The tests were green throughout

Both defects were in shipped, "finished", 26-tests-passing tooling, and the full suite
stayed green while the video was unwatchable. Neither defect was reachable from any
assertion: one lived in Chrome's capture semantics, the other in ffmpeg's default canvas.
**The only thing that found them was looking at the output.**

22 tests added and break-tested: 10 mutations of `bin/build-demo` and 5 of `dashboard.py`,
all caught. One initially survived — the mutation had hit a *comment* that quoted the code
it explained, and the test's substring search was passing on the prose. The test now
excludes comment lines, which is the second time that exact shape has appeared here.

Suite: **1116 passed, 2 skipped**. The video itself is unbuilt — the debugging spent the
day's quota, and the probe refused before rendering rather than producing a stale take.

---

## Phase 8 gates, round two — five round-one fixes had introduced new defects

Round one produced 36 findings. Fixing them meant the code the gates reviewed no longer
existed, so Phase 8 was not gated until a second pass ran against what was actually
there. It found **27 more**: 16 from code review, 11 from the numerical audit.

The headline is the part worth keeping: **five of the round-one fixes introduced new
defects, and no round-one correction had propagated completely.** That is the argument for
re-gating after a fix pass rather than treating "fixed" as "verified".

### Guard regressions came first, because a blind guard is worse than none

| # | Fix from round one | What it broke | How it was confirmed |
|---|---|---|---|
| F1 | curate tie-breaking | `max()` over a set broke ties by iteration order, so the canonical spelling depended on `PYTHONHASHSEED` | identical output hash across 4 seeds after `_pick()` |
| F2/F3 | the saturation guard | the break-test re-derived the criterion instead of calling it, and `render_sweep` never got the banner | weakened `is_saturated` and watched the test fail |
| F4 | the `#citations` deep link | fixed at one end only, so it could never fire | node-executed URL resolution, replacing a string grep |
| F5 | the `_STRUCTURAL` name strip | ate decimals, so `74.6x`, `0.95` and `53.2%` stopped being checked | break-tested all three |
| F7/F8 | the mutation guard | two whole report sections rendered identically under mutation and so were never compared | 4 injected figures, all now caught |

F7/F8 needed two goes. Uniform multipliers preserved cross-stage *ratios*, so the rendered
"5 orders of magnitude" was byte-identical in both renders and the guard read a derived
figure as a hardcoded one. A per-stage scale fixed that and exposed the next layer: at the
original sentinel magnitude a mutated `retrieve.min_ms` rendered as `12.53 s`, which is
exactly the real `generate` p50. Hunting for multipliers that happen not to collide is a
birthday problem — the widened 1..1200 sweep would always eventually find one. The
sentinel band now starts at 1e5 ms, which puts every sentinel latency at three integer
digits of seconds where no real figure reaches two. **Disjoint by construction, not by
luck.**

### Claims that outran the implementation

- **`Judged: 25 of 24 flagged pairs.`** A ratio above 1, printed as progress. The
  numerator counted every verdict on file; the denominator counted what the *current*
  detectors flag. They diverged the moment a detector fix retired a flag on a pair that had
  already been judged. Now reported as two populations, and the one retained verdict is
  explained rather than hidden — a human read `fda-136987::gold::049` when a buggier
  detector flagged it, and the fixed detector now agrees with the verdict.
- **"whose own header marks it non-binding"** was hardcoded prose attached to a `max()`
  over the verdicts. True of the document that leads today, false for the
  ClinicalTrials.gov protocols one rejection behind it, and false for 23 of the corpus's
  115 FDA documents. Now read from the document's own text, and the clause *disappears*
  rather than lies when the leading document is a protocol. Ties also stopped resolving by
  dict insertion order.
- **`n_judged` printed as "flagged pairs"** — 25 published as a flag count when 24 were
  flagged. Same root cause as the ratio above, one section away.

### The narration was quoting a configuration the service does not serve

`bin/build-demo` spoke two figures that were wrong as literals:

- "a hundred and sixty documents" is the number *fetched*. 152 have a usable text layer and
  are the only ones in the index.
- "a factor of seventy five" is the `fixed` chunking's identifier margin. The service
  serves `structural`, at **43×**. Quoting the best configuration's number while
  demonstrating a different one is exactly what the evaluation harness exists to prevent,
  and the video was about to do it on camera.

Both now come from the machine: the corpus size from a new `docs` field on `/health`, the
margin from `reports/failure_modes.json` keyed by the chunking the service *reports it is
serving*. Point `/health` at `semantic` and the narration says 56×. An unmeasured chunking
fails loudly rather than quoting a neighbour's number. The same bare `75×` was corrected in
`docs/architecture.svg` and `docs/demo-script.md`; the README and case study already framed
it as a range, and the case study already named the trap.

### Two escape hatches, one of them a single character

The volatile-timing guard was inverted from a denylist to an allowlist in round one — the
right direction, because you cannot enumerate future wrong values. Round two found the
allowlist had a hole: `if f"`{value}" in line: continue` exempted **anything** wrapped in
backticks. Appending "retrieval p50 is `0.179 ms`" to the README passed with every test
green, which is the same escape the allowlist had just replaced a denylist to close. The
two real retraction citations are now enumerated by `(file, value)`, so an exemption cannot
transfer between files, and `0.012` came off `STABLE_MS` — its stated justification pointed
at a file the guard does not even scan.

The guard was also extracted from the test body into `scan(name, text)`. Both of its escape
hatches were found by hand-editing the README and re-running; a check that can only be
aimed at four committed files can only be break-tested destructively, and a destructive
break-test does not stay run. Seven tests now aim it at synthetic text.

### `metrics.py` was at 29% coverage, and `separability` at zero

The worst place in the project for a hole. Every retrieval figure in the README — the
identifier finding, the chunk-size decision, the 0.992 refusal separability that Phase 5's
refusal path was built on — is an average over six functions that were exercised only
transitively, through harness tests that would have been equally happy with a wrong
denominator. **Now 100%**, 44 tests, written against the documented rationales rather than
the arithmetic: the distinct-id count that stops recall exceeding 1.0, the `k` denominator
that stops a nearly-empty retriever scoring 1.0, the `min(len(relevant), k)` ideal that
stops nDCG penalising a perfect ranking, and the above-every-score threshold candidate
without which `separability` reports a worse-than-baseline optimum as the best achievable.
All 8 mutations of those rationales are caught by the intended test.

### A per-minute rate limit was reported as a spent day

`re.search("quota", reason)` in three places. A per-minute 429's body reads "you exceeded
your current quota", so a limit that clears in seconds was reported as an exhausted day —
opposite advice. `generation.py` already made the distinction correctly off the structured
`quotaId`; nothing carried the decision. `GenerationError` now has a `daily_quota` flag,
the service emits it as `refusal_source`, and the dashboard and build script branch on the
field with the substring left as a labelled fallback that says "retry", not "come back
tomorrow".

### Hand-edited verdicts were loaded with no validation

`corpus/curation_verdicts.json` is the one artifact here that cannot be regenerated, only
redone. Only `verdict == "reject"` is acted on, so a typo — `"rejct"`, or a stray capital —
leaves a pair a human rejected sitting in the accepted set with the worksheet still
counting it as judged. Silent in the dangerous direction. Now refused, loudly. `amendments`
is validated too: nothing reads it, which is precisely why it can drift from the verdict it
describes, and an entry recording a keep→reject the verdicts block never received would
leave the file *documenting* a rejection the pipeline does not apply.

### A mistake in my own verification, worth recording

The `metrics.py` break-test reported six false **SURVIVED** results. `if k <= 0:` and
`if k == 0:` are the same byte length, and the restore landed within the same second, so
CPython's `(mtime, size)` pyc invalidation never fired and pytest ran stale bytecode
compiled from the mutated source. `inspect.getsource` read the restored file and showed
code that could not have produced the observed output, which is what gave it away. Re-run
with `__pycache__` cleared between mutations, all 8 were caught. Every mutation harness
here now clears the cache.

### One finding did not survive, and the README was overclaiming

**The false positive.** A code-review finding held that `separability` was documented as
"checked by its own tests", which is false. The phrase appears nowhere in the repository —
I searched the whole tree. The finding's *substantive* half was right and worth having:
`metrics.py` was at 29% coverage with `separability` at none. Acting on the premise as
stated would have meant hunting a sentence to correct; acting on the measurement meant
writing 44 tests. Reproduce first, every time.

**And a claim of my own that outran the record.** The README's verification section said
"across nine review passes... zero false positives". Both halves were wrong: it is 25 gate
passes, and there are now two recorded non-survivors — the finding above and a packaging
fixture I mis-set myself. Rewritten to lead with the claim that *is* supported (zero
arithmetic errors in any generated artifact), name the one wrong statistic (the
hand-transcribed ρ), and record the non-survivors instead of implying there were none. A
verification section that overstates its own verification is the worst possible place for
this defect.

**The module table had drifted by twelve modules** — every one from Phase 5 onward,
including `citations.py`, the verifier this project is pitched on. Now complete, and
checked by comparing the table against `src/ragpipe/*.py` rather than by reading it.

### Where Phase 8 stands

| item | state |
|---|---|
| 8a `hit@1` metric fix | shipped |
| 8b `ragpipe bench` | shipped |
| 8c golden-set curation | shipped — 24 of 159 flagged, all judged, 8 rejected, **151 accepted** |
| 8d five failure modes | shipped |
| README, case study, architecture diagram | shipped |
| `bin/build-demo` | **built and tested** (26 tests); the recording is deferred |
| both gates, round two | passed — 27 findings, all reproduced and fixed |

Suite: **1094 passed, 2 skipped** (was 980 before the round-two fixes). `metrics.py` 29% →
100%. New test modules: `test_metrics.py` (44), `test_build_demo.py` (26).

**The video is the one carved-out exception.** `bin/build-demo` is finished and its guards
are tested; it needs a fresh free-tier quota to run, which is a calendar constraint rather
than an incomplete deliverable. Aryan's call: *"If we are out of daily quota, we can keep
the recording for some other time."*

### A correction to the Phase 8c entry below, not an overwrite

The 8c entry says "the golden set is 152 accepted, down from 159" and "7 rejected". Both
were true when written. An eighth pair (`fda-78268::gold::094`) was reclassified keep →
reject at the Phase 8 code-review gate — the lexicon was missing `recommend`/
`recommendations`, so an FDA recommendation frame scored 0 and took the lenient branch. The
current figures are **24 flagged, 8 rejected, 151 accepted**, and the change is recorded
with its reasoning in the `amendments` block of `corpus/curation_verdicts.json`, which the
loader now checks against the verdicts it describes.

### Standing record across 25 gate passes

**Zero arithmetic errors in any generated artifact.** Every real defect has been a claim
that outran the implementation, a check in the wrong place, or a check that never ran. The
one wrong statistic in the project remains a hand-written one — Spearman ρ published as
0.917 when it was 0.948, because `0.9167` is a `hit@1` table cell and I transcribed a cell
as a correlation.

---

## Phase 8d — the failure analysis, and three stale figures it exposed

**950 tests, lint clean.** New: `src/ragpipe/failures.py`, `ragpipe failures`,
`make failures`, `agentrag failures`, `tests/test_failures.py` (19 tests),
`reports/failure_modes.{md,json}`, and a README section leading with it.

Five failure modes, each backed by a committed artifact. The interesting part is not the
list — it is what building it as a *generated* report rather than prose turned up.

### Writing it caught three figures the corpus no longer supports

The rule was that a claim only appears if an artifact can produce its number. Applying
that rule flushed out three quoted statistics computed **before the post-quarantine
regeneration**, when the corpus held 13,706 chunks over 154 documents rather than the
current 13,423 over 152:

| where | figure | status |
|---|---|---|
| `citations.py` (×2), `dashboard.py` | line-number exposure "7,229 of 13,706 (52.7%)" | **recomputed: 7,141 of 13,423 (53.2%)** |
| `contextual.py` | the whole cost table, "$39.58 / $421.92" per-chunk | **recomputed: $39.02 / $416.36**; every figure was ~1.5% high |
| `retrieval.py` | heading-overlap "3,720 of 13,706; 28.6% of 13,000" | **labelled superseded, not replaced** |

The third is the honest one. I could recompute a heading-overlap number, but the original
measurement's definition of "carries the heading" is not recorded, and a freshly-invented
definition would produce a figure that *looks* like a correction while measuring something
slightly different. So it is marked as pre-quarantine with the reason, and recomputing it
properly is a re-run of the ablation.

Every conclusion survived. None of the arithmetic did, and **nothing would have caught any
of it**, because all three lived only in prose — which is the argument for the rule.

### A near-miss I have to record: I almost reported my own bug as the code's

While recomputing the contextual cost table I found `estimate_cost` returning identical
figures for all three architectures, and `assume_cache_hits` changing nothing at all — a
documented "genuine upper bound" parameter that appeared to be dead. I had the finding half
written.

It was my harness. `data/extracted/index.jsonl` holds *metadata* — `n_chars`, `n_pages`,
`n_sections` — and the document text lives in per-document files. I had fed `estimate_cost`
155 records with no `text`, so every prefix was zero tokens, nothing was cacheable, and the
cache branch could not fire either way. Loading the real text gives 13,140 cacheable groups
at per-chunk and a proper spread across architectures.

**The lesson is the one this project applies to subagents, turned inward: reproduce a
finding against a known-good input before believing it.** A measurement harness is as
capable of being wrong as the thing it measures, and mine was.

### The five modes

1. **Dense retrieval collapses on exact identifiers** — BM25 0.9553 against
   dense 0.0128, a factor of **75**, and naive RRF at
   0.1628 is *worse than sparse alone*. Consequence: the served retriever is
   BM25, and that is what the table supports rather than a simplification for the demo.
2. **Reference answers manufacture obligations** — the 7 curation rejections from Phase 8c.
   Consequence: this one corrupts the *evaluation*, not the product, which is why the
   golden set is now curated and the verdicts committed.
3. **The over-refusal rate was mostly a labelling failure** — 4 of 12 naively, 1 of 8 once
   the mention-labelled slice is excluded. Consequence: golden answers cannot be derived
   from the retrieval eval set. A metric that looked like a model defect was a property of
   the ruler.
4. **Answers reach past the span they cite** — 7 of 8 fully supported, 1 partial. A quote
   being real is not the same as a quote being sufficient, and tier 1 cannot see the
   difference by construction.
5. **The verifier's own tier certified a change in meaning** — "40 CFR" verifying against
   "21 CFR", across 53.2% of the corpus. Consequence: demoted to a
   non-verified diagnostic, and never green in the UI.

### Two failure modes are anecdotes, and the report says so

`wrong_chunk` and `line_number_ambiguous` are both **zero** in the shipped generation run
and both have been observed live — the first on the very first request ever served through
the container. A failure analysis that promoted an `n=1` observation to a rate would be
doing precisely what this project exists to avoid, so they sit in their own section labelled
as anecdotes, with the cause of the small sample named: the free tier's 20-requests-per-day
ceiling holds the generation sample at 16.

### The mutation guard, and four rounds of arguing with it

`test_no_number_survives_payload_mutation` renders from the real payload and from one whose
every leaf is a sentinel; any number identical in both is by construction not derived from
the data. Getting it to a state where it flagged only real problems took four corrections,
and all four are worth recording because each was the *test* being wrong:

1. Perturbing every dict key renamed the payload schema the renderer indexes by, so the
   report crashed with `KeyError` instead of rendering.
2. Leaving keys fixed made pair ids (`::gold::016`, `fda-75426`) appear identically in both
   renderings — the guard flagged its own fixture. Now only `obligations.rejected`, whose
   keys are data, gets perturbed keys.
3. Perturbing `refusal.per_slice` keys made `per_slice.get("unanswerable")` miss and print
   its `0` fallback, which registered as a hardcoded zero. That fallback is graceful
   degradation, not a defect.
4. Consecutive integer sentinels made *ratios* of perturbed counts round to 1.0, rendering
   "100.0%" — colliding with the real payload's 100.0% negative-judge rate. Sentinels now
   step by a large prime.

Break-tested afterwards: injecting `53.2% of 13,423 chunks` into a heading is caught with
both tokens named.

### Carried forward

- **`reports/chunk_size_sweep.md` still carries the pre-quarantine corpus** (13,706 chunks).
  It is a dated snapshot of a run, and hand-editing its table would forge a measurement, so
  regenerating it is a `make sweep` (~25 min) rather than an edit. The staleness test is
  scoped to live prose for exactly this reason.
- The heading-overlap figure in `retrieval.py` is labelled rather than recomputed, pending
  an ablation re-run.

## Phase 8c — the golden set is curated, and 8 answers invented obligations

**931 tests, lint clean.** New: `src/ragpipe/curate.py`, `ragpipe curate [--apply]`,
`make curate`, `agentrag curate`, `tests/test_curate.py` (34 tests),
`corpus/curation_verdicts.json`, `reports/golden_curation.{md,json}`.

**The golden set is 152 accepted, down from 159.** Every pair `plan.md` had
described as "machine-validated and spot-checked, **not curated**" since Phase 6 has now
been triaged, and every flagged pair read against its source.

### Target the blind spots, do not sample randomly

The five validation checks end in answer/evidence lexical *coverage*, which is a bag of
words, and Phase 6 had already measured precisely what that cannot see:

> "60 days" against "30 days", "may" against "shall", and "shall not submit" against
> "shall submit" all scored coverage 1.000 and were accepted.

Numbers were already handled by `unsupported_numeric_claims`. Obligation level and
polarity were not — and in regulatory text they *are* the meaning. FDA guidance states the
convention explicitly: **must** is a requirement, **should** is a recommendation, **may**
is an option. An answer that says "must" about a source that says "should" has
manufactured a legal obligation while remaining lexically identical to a correct one.

So: rank the modals, compare answer against cited source, and read everything that flags.
**24 of 159** pairs flagged (15.1%) — a readable number rather than a
sample, and targeted rather than random. A random sample of 25 would have found roughly
one of the seven defects.

### What was actually wrong: 7 rejections, all the same shape

| pair | source says | answer says |
|---|---|---|
| `NCT02942264::gold::016` | treatment **should** be discontinued; pregnancy reported within 24 h | **must** be discontinued, **must** be reported |
| `NCT04125745::gold::021` | capsules **should** be stored refrigerated 2–8 °C | **must** be refrigerated |
| `NCT04125745::gold::023` | "we **may** discontinue the study" | "the study **will** be discontinued" |
| `fda-75426::gold::085` | question **should** have minimal qualifiers | **must** contain, **must** not be leading |
| `fda-75426::gold::087` | votes **should** be read aloud | **must** be read aloud |
| `fda-75426::gold::089` | the Chair **should** first check with the DFO | the Chair **must** verify |
| `fda-78268::gold::097` | "You **should** include a Table of Contents" | "a Table of Contents **is required**" |

Two of them are safety instructions — a pregnancy discontinuation and a drug storage
temperature. Three come from one advisory-committee guidance whose own header marks it
non-binding. And `gold::097`'s **question** is leading in the same direction ("which
submissions *must always* include"), so both halves of that pair assert an obligation the
guidance does not.

The 18 keeps are as informative. Regulatory text obliges without modals constantly, and
each of these is a correct answer that a naive rule would have deleted: imperatives
("Ship via Federal Express"), bare protocol specifications ("slice thickness 5 mm skip 0
mm"), enumerated statutory duties ("(5) Have systems and processes in place"), CFR
documentation lists, and a clinical protocol's "will", which is directive rather than
descriptive. Every keep records *why*, so a later reader can disagree with a specific call.

### Six bugs in my own detector, all found by judging real pairs

This is the part worth keeping. Every one produced a **confident false positive**, and
every one was invisible until a human read the pair beside the source.

1. **`not` without word boundaries** matched inside "notified", so "the FDA must be
   notified" read as a *negated* obligation. Same class as the `\bcomment\b` regex Phase 6
   found — that one had a boundary it should not have, this one lacked one it needed.
2. **The polarity window looked only forward.** English negates backward just as often:
   "are **not** required", "is **not** needed". So the check was blind to the exact
   inversion it existed to detect.
3. **`max_modal_strength` counted negated modals**, so "bookmarks are not required" —
   correctly paraphrasing a source's "not needed" — scored *mandatory*.
4. **Fixing (3) changed nothing**, because `check_modal_strength` re-derived the maximum
   itself instead of calling `max_modal_strength`. A check that recomputes a rule the
   module already implements will not notice when the rule is corrected. This project's
   recurring defect in its purest form.
5. **An incomplete lexicon.** Bare `require` was missing while `required`/`requires` were
   present, and 8 of 159 sources use the infinitive: "MR scanning sessions **require**
   participants to lie flat" went unmatched, so the source's strongest modal fell back to
   an unrelated clause's "will". Separately, `requirement` — a **noun** — was ranked
   mandatory, flagging a pair whose own modals matched its source exactly.
6. **PDF word-splitting.** One source reads "the requir ed reporting", which `\brequired\b`
   cannot match. Fixed with a space-tolerant pattern — and then the fix for (3) silently
   **re-broke it**, because only the other matcher had been made tolerant. Caught by the
   test written for (6), which is the entire argument for pinning a fix with a test rather
   than trusting the fix.

### A check firing on 60% of the data is worse than no check

The first negation check compared negation-*counts* between answer and source and flagged
any asymmetry. It fired on **96 of 159 pairs** — because a fifty-word answer is being
compared against a thousand-character chunk, and the source is nearly always more negated.
That is not triage; it teaches a reviewer to skip the column. Rebuilt around **polarity
attached to a modal**, it fires on 2.

The same reasoning removed a whole check. A `modal_weakened` branch fired on 19 pairs and
contributed 45% of all flags. Reading all 19 found **zero** real defects, and a
sentence-level re-test confirmed why: 18 of 19 vanish when the answer is compared against
the sentence that actually addresses it. It was not detecting under-claiming — it was
detecting that regulatory documents contain obligations. Deleted, and pinned deleted by a
test, because re-adding it would silently restore a 45% noise floor.

Flag rate across the rebuild: **72% → 30% → 26% → 15.1%**, with the real defects intact.

### The chunk-level comparison got one direction backwards

`gold::023` — "we **may** discontinue the study" restated as "the study **will** be
discontinued" — was reported as `modal_weakened` at **low** severity, because the chunk
contains "required" in an unrelated clause and so out-ranked the answer overall. A
discretionary safety action restated as a commitment, filed under the harmless category.

Fixed by adding `check_modal_in_context`, which compares the answer against the single
most relevant source sentence. It catches 6 pairs the chunk-level check missed, of which
this one is a genuine defect. Both checks run: the chunk-level one is lenient about
multi-sentence answers, the sentence-level one is precise about direction, and neither
decides anything.

Also fixed while judging: the worksheet showed the window around the *source's strongest
modal*, which is frequently an unrelated clause — a pair about central venous lines was
presented beside a sentence about glomerular filtration rate. A worksheet that makes
judging harder than reading the raw pair is worse than no worksheet.

### Two commands, each correct, whose interaction erased the work

`ragpipe revalidate` re-runs the machine checks from each pair's *drafted* state —
deliberately, so relaxing a threshold can re-accept a pair rather than only reject more.
But the pairs a human rejects are precisely the ones that **pass** every machine check;
that is why they needed a person. So after applying the curation, `revalidate` reported
**152 → 159** and `--write` would have silently undone the entire pass and called it a
clean run.

`revalidate` now honours curation verdicts and says so on every run ("7 pair(s) held
rejected by human curation verdicts that the machine checks would have re-accepted"), with
a test that drives the real command against the real artifacts and fails if the accepted
count ever rises.

### Why the verdicts are committed

`corpus/curation_verdicts.json` is the **one artifact in this project that cannot be
regenerated, only redone**. Every report here rebuilds by re-running a command; a judgement
about whether an answer misstates a regulation is a person reading a source. Phase 6
shipped a faithfulness figure that was not re-derivable because the verdicts behind it were
never written down — this is that lesson applied before the fact. Each entry carries the
verdict, the reason, and the source wording, so a later reader can disagree with a specific
call rather than with the set as a whole. A test requires every reason to be substantive.

## Phase 8b — `ragpipe bench`: the latency figures become an artifact

**897 tests, lint clean.** New: `src/ragpipe/bench.py`, `ragpipe bench`, `make bench`,
`agentrag bench`, `tests/test_bench.py` (28 tests), and
`reports/serving_bench.{md,json}`.

### Closing a gap this project had already named twice

Phase 7 delivered plan deviation #8 on `/stats` and quoted three single observations:

    retrieve 0.44 ms | generate 6257 ms | verify 0.012 ms

They were correctly labelled `n=1` after a review — and they lived in **no**
`reports/*.json`. That makes them precisely what the Phase 7 entry names twice as the
recurring defect: *a measurement that runs somewhere the project cannot re-run it is a
measurement the project does not have.* The ANN report had the same problem and lost its
headline to it.

| stage | n | p50 | p95 | source |
|---|---:|---:|---:|---|
| `retrieve` | 540 | 0.173 ms | 0.246 ms | measured |
| `verify` | 24 | 0.006 ms | 0.201 ms | measured |
| `generate` | 16 | 12.53 s | 27.11 s | stored |

**These two local rows move on every run.** `retrieve` and `verify` are wall-clock timings and the payload retains no samples for them, so a re-run shifts them a few percent — this log records what one run measured, and `reports/serving_bench.md` is always the current figure. The README and case study now publish them as *bounds* for the same reason, after a re-run during the Phase 8 audit silently made a quoted `0.173 ms` wrong.

### The quota forced the design, and the design is the honest one

The three stages cannot be sampled the same way, so `n` is a **column** rather than a
footnote — one `n` for the table would be a lie about two of the rows.

- **retrieve** — local and free. 180 answerable queries x 3 passes, warmed first because
  `bm25s` builds lookup structures lazily and an unwarmed first call lands in the p95.
- **verify** — local and free, but it needs *citations*, and producing citations means
  generating. So it **replays the real `{chunk_id, quote}` pairs already stored** in
  `generation_eval.json` against the same chunks. Identical work to the serving path, no
  quota.
- **generate** — a third-party call against 20 requests per day. Measuring it here would
  spend the day's evaluation budget on a latency row, so it is read from the stored Phase
  5 outcomes and the `source` column says so. `--live-generate N` measures it fresh,
  **capped at 5** in the CLI rather than only in the help text, because a rejected request
  also spends a unit.

`total` is deliberately absent: summing medians measured in isolation is not a request
latency. `/stats` reports real end-to-end totals from served traffic.

### Three things found by sampling properly rather than once

**1. The stored generation figure is double what I had been quoting.** p50 **12.53 s**
against the ~6.3 s of one live request. Before publishing that I checked what `latency_s`
contains: `attempts == 1` on all 16 outcomes, so it is *not* retry waits, and the stored
run used the same model as the current default. The real explanation is spread —
**2.16 s to 27.11 s, a 12.6x range** — with only weak correlation between
output tokens and time (r = +0.32, n=16). So it is variance in somebody else's
infrastructure, and the ~6.3 s was one draw. The report now says to quote the range or
p50 *with* p95, and treats any single observation as an anecdote.

**2. A ratio can be arithmetically right and still false precision.** The first render
printed "**Generation is ~69,712x the cost of retrieval**". True division, indefensible
resolution: the numerator is the median of a 12x-wide distribution, so the "real" multiple
moves by an order of magnitude with the draw. Now stated as **orders of magnitude**, with
a test asserting the five-figure multiple does not reappear.

**3. Verification cost depends on the verdict it reaches.** The verify p95 was ~30x its
p50 on a deterministic function. Not cold start (warming it changed nothing) and not
citation count — six `exact` citations cost less than one `normalized` one:

| answers containing | p50 | slowest |
|---|---:|---:|
| `exact` only | 0.0050 ms | 0.0125 ms |
| any `normalized` | 0.1328 ms | 0.1977 ms |

`exact` is a substring search; `normalized` rebuilds the chunk whitespace-collapsed **plus
an index map back to the original character offsets**, which is O(chunk length) per
citation. Reported rather than smoothed away because it says which way real traffic skews:
`normalized` is the *common* case in PDF-extracted text, so served traffic sits at the
expensive end of that range — and still five orders of magnitude under generation.

### The renderer gets the mutation guard from day one

`render_report` quotes more derived figures than the ANN renderer did, and that module
shipped a stale literal twice. So `test_no_number_survives_payload_mutation` was written
alongside it rather than after a review found something: render from the real payload and
from one whose every leaf is a sentinel, and any numeric token identical in both is by
construction not payload-derived.

It immediately caught **its own scaffolding**: the perturbation replaced strings with
`f"vX{counter}"`, and `vX3` contains the numeric token `3`, which the guard then flagged
as a leak from the `"3 passes"` note. Sentinels are letters-only now. A guard sensitive
enough to catch its own test helper is the right sensitivity.

## Phase 8a — the saturated slice was a saturated *metric*

**869 tests, lint clean.** No retrieval re-run: every metric was already stored for all
192 runs, so this is a re-summary. `reports/retrieval_eval.{md,json}` regenerated;
`primary_metric_note` in the JSON records why.

### The carried-forward note was a conclusion, and it was wrong

Three documents carried this, in the same words, since the post-quarantine regeneration:

> **`title_lookup` is now saturated and no longer discriminates.** Every one of the 16
> retriever configurations scores exactly 1.000 on it. A slice on which nothing can lose
> measures nothing, so it should either be made harder (shorter or partial titles) or
> dropped from the headline table.

Both remedies would have been damage. Checking the claim against the stored report instead:

| metric | spread over 48 configurations | distinct values |
|---|---|---|
| `hit@10` — the primary metric | 0.9833 – 1.0000 | **2** (47 of 48 exactly 1.0000) |
| `hit@1` | 0.8500 – 0.9667 | 8 |
| `mrr@10` | 0.9218 – 0.9833 | 18 |
| `ndcg@10` | 0.7460 – 0.7960 | 39 |

**The slice discriminates. The metric was saturated.** And the reason is structural rather
than accidental: the queries are verbatim document titles, so retrieving the right document
*somewhere in the top 10* is trivial for any retriever — while ranking it **first** is not.
`hit@10` was measuring the easy half of the question.

The `hit@1` spread is **3.0 binomial standard errors** at n=60 (se ≈ 0.0387), so it is
signal, not sampling noise. Primary metric moved to `hit@1`; the slice stayed.

**`hit@1` over `mrr@10`**, which has more resolution, because the two order all 48
configurations near-identically — Spearman ρ **0.948**, identical top five — so the extra
resolution buys no different conclusion, and "was the right document ranked first" is a
sentence a reader can check against the table. Measured before choosing rather than argued.

### Why this one stings

The note did not misreport a number. Every figure in it was correct: `hit@10` really was
1.000 almost everywhere. What it got wrong was the **inference** — from one summary
statistic at one depth, to a property of the slice — and then it proposed acting on that
inference by *changing the ground truth* ("shorter or partial titles") to fix what was
actually a reporting choice. Rewriting the queries would have made the corpus worse and
hidden the real finding.

Same shape as Phase 4's "1,024 dominates" and Phase 5's "0.725 for any fusion": a claim
that outran the data it was drawn from, then propagated verbatim into three documents
because each copy looked like corroboration.

### A generic guard, not a fix for this one slice

`test_no_slices_primary_metric_is_saturated_in_the_shipped_report` checks **every** slice's
primary metric against the artifact on disk and fails if it takes fewer than two distinct
values across the configurations being compared — because a headline metric that cannot
move cannot support any ordering the report prints beneath it. It would have caught this at
the point the metric was chosen rather than three phases later, and it guards slices not
yet written. Break-tested: `hit@10` still fails it, `hit@1` passes.

### An incidental finding

`title_lookup` is the **only slice where dense retrieval is not beaten**: on `structural`,
dense scores 0.883 against BM25's 0.867. Stated with its size — that is **one query in
sixty**, comfortably inside the standard error, so it is not a win. It is the *absence* of
the collapse dense suffers on exact identifiers, where it loses by 43–75×. Worth a line in
the Phase 8 failure analysis: the one query type where a semantic model holds its own is
the one where the surface form of the query already matches the document.

## Tooling — `agentrag`, and bash 3.2 biting the untested path immediately

**`bin/agentrag`**, symlinked into `~/.local/bin` (also `make install-agentrag`). One entry
point for the stack: `start` / `stop` / `restart` / `status` / `logs` / `open` / `query` /
`stats` / `health` / `ann` / `serve` / `test` / `lint` / `doctor`, and a bare `agentrag`
lists them all.

It exists because bringing this up is four steps in a required order — colima, compose,
health wait, then knowing which port is which — and each wrong order fails in a way that
looks like the *app* being broken rather than the sequence: compose against a stopped VM
reports a daemon socket error, and an unhealthy Qdrant is indistinguishable from a Qdrant
whose healthcheck shell lacks `/dev/tcp`. It echoes every command it runs, so it does not
become the only way to operate the project.

**It shipped with a silent failure, found within minutes, in exactly this project's
recurring shape.** macOS ships **bash 3.2**, where `set -u` plus an *empty* array
expansion (`"${build[@]}"` with `build=()`) is a fatal `unbound variable`. So
`agentrag start --no-build` printed `==> starting api + qdrant`, died, and skipped both
the health wait and the status report — while `agentrag start` (the `--build` path, a
non-empty array) worked perfectly. One manual test of the default passed and the branch
nobody exercised was broken. Worse, `restart` had already run `docker compose down`, so
the visible outcome was a stack that quietly went away.

Now `tests/test_agentrag.py` (13 tests, no Docker or network needed) guards: the script
parses under `/bin/bash` specifically — not whatever bash is first on PATH, since
Homebrew's bash 5 hides this class entirely — no unsafe array expansion survives outside a
comment, help and the dispatch table still describe each other in both directions, an
unknown command exits 2 and still prints the menu, repo-root resolution works through a
symlink and ignores the caller's cwd, and the Gemini key is exported but can never reach
the terminal (`run` echoes its command, so the key must never be an argument).

Break-tested: reintroducing `"${build[@]}"` fails the guard.

Two smaller fixes found while testing it: the status and query output used f-strings with
escaped quotes inside the expressions, a syntax error before Python 3.12, so both silently
fell back to raw JSON; and `lsof` reporting `ssh` on ports 8000/6333 reads like a problem
but is exactly how colima publishes container ports, so `doctor` now names it as healthy
rather than leaving it ambiguous.

**Lesson, again: the default path passing says nothing about the branch beside it.** Three
Phase 7 gate findings were tests that passed against the defect they named; this one was a
whole code path with no test at all, and it took one invocation to expose.

## UI pass 2 — written for someone who has never seen this project, and seen in a browser

**867 tests, lint clean** (from 855; two skip under the sandbox, which denies the HF
model download and TLS read they need). The trigger was a screenshot: the page was
legible only to someone who already knew the system.

### Renamed for the reader: agentrag

The page and the OpenAPI title now say **agentrag** — what you actually type to run it.
`ragpipe` stays the Python package and import namespace. This is the one place the two
differ deliberately, and a test asserts the page says `agentrag` so they cannot silently
swap back.

### Six things a first-time visitor could not understand

Every one was a real complaint against a real screenshot, not a hypothetical:

| was | now |
|---|---|
| `k=5` dropdown | **"passages to read"**, with a tooltip on the trade-off |
| `retrieve only` checkbox, silently *checked* when no key was set | **"search only / skip the AI answer"**, plus a banner explaining that search still works |
| `retrieve p50 0.373ms / p95 0.373ms (n=1)` | "Your request took: search 0.64 ms" |
| "No answer requested. **retrieve_only** Retrieval ran and is shown below." | "**Search only.** You asked for the matching passages without an AI answer…" |
| explainer said "passages", table said "chunks" | one word, enforced by a test |
| route list + per-token-rate paragraph on first paint | folded into "For developers" |

`k` is a variable name from the retrieval code. `retrieve_only` and `no generator
configured` are API tokens. Neither belongs in a sentence aimed at a visitor, and
**`p50 … p95 … (n=1)` is the same defect this project corrected in its own reports**:
three pieces of jargon wrapped around a single observation and presented as a
distribution. The strip now earns the word "typical" only at `MIN_TYPICAL` samples and a
tail figure only at `MIN_TAIL`.

Two additions rather than fixes: an **empty state** ("What happens when you ask" — search,
answer, verify — plus a glossary), so the first paint teaches instead of showing a blank
page above a footer; and **example questions phrased as questions**, one of which the
corpus cannot answer, so a visitor can watch a refusal without having to invent one.

Jargon that remains — BM25, passage, verified — is now defined on the page, and a test
asserts each has a definition rather than a mention.

### Seen, not imagined

The browser extension was unavailable, but Chrome screenshots headlessly from the command
line with no extension at all:

```
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
  --screenshot=out.png --window-size=1280,1900 --virtual-time-budget=45000 \
  --force-device-scale-factor=2 "http://127.0.0.1:8000/?q=…&k=5"
```

The `?q=` permalink auto-runs the query, so a single command captures a fully populated
result view. This is how four defects were found that **no test in this repo could have
caught**: the select sat visibly lower than the input beside it (a label above it broke the
row's centre alignment), the example chip's hint rendered as blue monospace running straight
on from its label, the default checkbox was a white square in a dark theme, and an empty
`.status` paragraph reserved a line and left a visible void.

**One caveat worth recording so nobody quotes it.** `--virtual-time-budget` compresses the
client clock, so the "round trip" figure in any headless screenshot is not real elapsed
time: a capture read `0.07s` for a request whose server-side `generate` was **3,955 ms**.
The per-stage table is server-measured and accurate; the round-trip line is only
trustworthy in a real browser.

### A guard narrowed, deliberately, and break-tested

`test_the_page_does_not_promise_a_dollar_figure` asserted `"cost_usd" not in page`, which
forbade the page from *reading* the server's field. Wrong target: `/stats` returns
`cost_usd: null` with a note saying why, and surfacing that null is a feature. The risk is
the page **inventing a rate** — multiplying a token count by a number of its own. Split
into two: no currency glyph, and no client-side arithmetic on a token count. Both
break-tested, because "the test failed so I loosened the test" is the shape of a real
defect.

A second, smaller instance of the same care: the first vocabulary test tried to extract
quoted strings with a regex and matched across the entire CSS block, because `"` appears
there only sparsely. Replaced with a rule that actually separates prose from code — the
English word is preceded by whitespace, every legitimate use is part of an identifier —
plus a test that the rule would catch a regression, since a rule matching nothing passes
just as quietly.

## UI pass 1 — the diff is the product, and a cross-check found a false claim in it

**855 tests, lint clean** at the end of this pass (from 825); **867** after UI pass 2 above.

### What changed, and why each

**A word-level diff replaced two panes of plain text.** The old page put the model's quote
and the document's text side by side, which technically contained the answer and made the
reader hunt for it. The differences that matter on this corpus are one token wide, and they
are exactly the ones Phase 6 found lexical coverage scoring **1.000 and accepting**:

```
model : as required under [-40-] CFR 314.50 the sponsor shall report
source: as required under [+21+] CFR 314.50 the sponsor shall report
model : the sponsor [-shall-] submit a report
source: the sponsor [+may+] submit a report
```

LCS over whitespace-split tokens, `<ins>`/`<del>` on the side each token belongs to.
Quotes are a sentence or two, so the O(n·m) table is a few thousand cells.

**The clean case renders plain.** A whitespace-only difference — the common PDF case, a
doubled space left by stripping a line number — is a *real* match under `normalized`.
Marking those tokens put highlights on screen directly under a legend saying there was
nothing to mark. Found by executing the diff, not by reading it.

**Retrieved rows carry text and a `cited` badge.** `/query` now returns each hit's own
`text` (900 chars, with `n_chars`/`truncated` saying what was withheld) and, after
verification, whether the answer actually leaned on it. A retrieval demo showing only doc
ids and scores cannot be judged — "was that a good hit?" is a question about the text — and
"retrieval ranked ten things, the answer used two" is a ratio visible nowhere else.
`text`, never `embed_text`: the latter carries a prepended heading the model never sees as
body content, and a test asserts `embed_text` never reaches the page.

**Example queries, one of which should refuse.** A refusal is a feature of this system, so
the demo has to be able to show one without the visitor inventing an unanswerable question.

**Smaller:** live elapsed timer (generation is 5–7s and a static line reads as a hang);
score bars; expandable chunk panels; `?q=&k=&r=` permalinks; a cumulative p50/p95 strip
from `/stats`; `/` focuses the box; `aria-live` on the status and health lines;
`prefers-reduced-motion`; `focus-visible` rings; and a glyph (`✓ ! ✗`) beside every colour
so a verdict survives a colour-blind reader and a greyscale screenshot.

### The finding: a banner that described an event that never happened

The page was cross-verified by deriving its component inventory from the source — element
ids in markup vs ids the script writes, `fetch()` paths vs routes the app serves, and every
response field the page reads vs the fields present in **each** of the five `/query`
branches. Two things fell out.

One dead id (`kbdhint`, declared and never written) — now used to hide the `/` hint on
touch devices, where the badge advertises a key that does not exist.

And a real defect: the new banner said **"the model was told to answer only from the
retrieved text and reported that it could not"** for *every* refusal. The API distinguishes
three, and only one involved a model — `no_context` and `score_gate` refuse **before**
generating, spend no quota, and send no request. For two of three cases the banner was
describing an event that never occurred. `refusal_source` exists in the response for
precisely this distinction and the UI flattened it. Now each is described as what actually
happened, with a test per source.

**Lesson: an inventory derived from the code finds what reading it does not.** The banner
read fine. It was comparing the page's field reads against every response branch — a
mechanical check, not a careful re-read — that showed one branch reaching a paragraph
written for another.

### The diff is executed, not asserted

`tests/test_dashboard_diff.py` lifts `diffTokens`/`renderDiff` out of the shipped HTML
string and runs them under node: identical text produces no marks, each single-token change
is isolated to exactly one `<del>`/`<ins>` pair, whitespace-only differences report clean,
marks appear only on the side they belong to, and markup in either input is escaped rather
than rendered (both inputs are untrusted — one is a PDF extract, the other is model output,
and both flow through the diff into `innerHTML`).

Skipped when node is absent. It is a test-time convenience, not a dependency: nothing in
`ragpipe` needs a JS runtime, and adding one to a 569 MB container to test a string would
be the wrong trade.

Break-tested: disabling the LCS match arm fails 5 of the 11.

### A guard narrowed, deliberately

`test_the_page_does_not_promise_a_dollar_figure` asserted `"cost_usd" not in page`, which
forbade the page from *reading* the server's field. Wrong target: `/stats` returns
`cost_usd: null` with a note saying why, and surfacing that null is a feature. The risk is
the page **inventing a rate** — multiplying a token count by a number of its own. Split
into two checks: no currency glyph (unchanged), and no client-side arithmetic on a token
count. Both break-tested, so narrowing did not hollow it out.

Recording it explicitly because "the test failed so I loosened the test" is the shape of a
real defect, and the distinction between *reading* a server-computed figure and
*fabricating* one is the whole reason this project reports cost in tokens.

## Phase 7b — the API, the dashboard, the container, and a headline that did not survive

**Status: SHIPPED — both gates passed.** New artifacts: `src/ragpipe/{service,dashboard}.py`,
`Dockerfile`, `docker-compose.yml`, `ragpipe serve`, `ragpipe ann`, `tests/test_packaging.py`,
and a regenerated `reports/ann_recall.{md,json}`. **825 tests, lint clean** (from 727 — 98 new, almost all on code that had never
been executed). Two skip under the sandbox, which denies the HF model download and
the TLS read they need; they pass outside it.

**22 findings across the two gates (8 numerical, 14 code review), every one reproduced
before acting, all fixed. Zero false positives.** The numerical gate recomputed 227 figures
from the payload without importing the renderer: **zero arithmetic errors**, the fifteenth
consecutive audit with that result. Every real defect was again a claim, a check in the
wrong place, or a check that never ran.

### The `/query` route had never accepted a request

`service.py` was written in Phase 7a and never called. The first test found a 422 on a
well-formed body:

```
{"type": "missing", "loc": ["query", "req"], "msg": "Field required"}
```

`loc: ["query", "req"]` is the tell: FastAPI had classified the request body as a **query
string parameter**. The cause is the interaction of two individually reasonable choices.
`service.py` uses `from __future__ import annotations`, so every annotation is a string
FastAPI resolves against the *module* globals — and `QueryRequest` was defined inside
`create_app()`, where it is a local. FastAPI could not resolve the name, fell back to
treating `req` as a scalar, and the only working route was one nobody could call.

Worth recording because of *how* invisible it is. The module imports. It lints. It type
checks. `create_app` returns an app, the app starts, `/health` and `/stats` answer
correctly. Only a request to the one route that matters reveals it, and the reason no
request had been made is that the phase's status line said the service was "written but
untested" — which was true and read as a small remaining task rather than as "the central
feature is in an unknown state". Moved to module scope, with the reasoning in its
docstring so it does not get tidied back.

### The ANN report's headline was one build's luck

`reports/ann_recall.md` existed from Phase 7a with the claim:

> **At Qdrant's default search settings, approximation costs nothing: recall@k and top-k
> agreement are both 1.0000.**

It was produced by an ad-hoc script in a scratch directory. Wiring the same measurement as
`ragpipe ann` — the point being reproducibility, not new numbers — gave **0.9883 recall and
0.9000 agreement** on the identical 60-query sample. Neither figure reproduced.

This is the third time this project has found the same shape of defect: Phase 6 found three
golden-set checks with no call sites outside their own module, and the fix there was the
same as the fix here. **A measurement that runs somewhere the project cannot re-run it is a
measurement the project does not have.** Its numbers are unfalsifiable, and unfalsifiable
numbers drift into reports and stay there.

### Separating the two candidate causes, rather than picking one

The obvious explanation was run-to-run noise — and the report already said so, in a caveat
claiming "segment search order varies, so read the low end of the curve as approximate to
about a point". A controlled run tested it: sweep twice over one built index, then rebuild
and sweep again.

| | `ef=16` recall | `ef=128` recall | `ef=128` identical |
|---|---:|---:|---:|
| build 0, sweep 1 | 0.9117 | 0.9900 | 0.9000 |
| build 0, sweep 2 | 0.9117 | 0.9900 | 0.9000 |
| build 1, sweep 1 | 0.9267 | 0.9933 | 0.9333 |
| build 1, sweep 2 | 0.9267 | 0.9933 | 0.9333 |
| build 2, sweep 1 | 0.9233 | 0.9950 | 0.9667 |
| build 2, sweep 2 | 0.9233 | 0.9950 | 0.9667 |

**Two sweeps over the same index are bit-identical. Search is deterministic.** The variance
is entirely HNSW graph construction, which is randomised. So the existing caveat was wrong
twice over: wrong about the mechanism, and wrong to confine the looseness to low `ef` —
the spread reaches the top of the curve, which is exactly where the discarded headline came
from.

`ragpipe ann` now defaults to five builds and reports a median with the observed range.
Over five builds the default row is recall **0.9883 (0.9833–0.9883)** and agreement
**0.9000 (0.8500–0.9000)**.

The corrected claim is weaker and more useful: at default settings approximation costs
about a point of recall, but **reorders the top-k for roughly one query in ten** — and
rank order is what feeds fusion and reranking, both order-sensitive. The negative result
about latency is unchanged and slightly larger: ~7.4× the exact matrix product, of which
~0.78 ms is bare HTTP transport — and that split is now published as a **bound rather than
a partition**, because the transport baseline is a `GET` on an idle server while the timed
query is a `POST` carrying a 384-float vector against a live index, so serialisation that
belongs to transport is counted in the search remainder.

**What is not explained.** 1.0000 sits *outside* the five-build range, not at the top of
it, so build randomisation accounts for the spread but not for that value. The original
script is gone and cannot be inspected. The report states the residual rather than
inventing a mechanism for it.

### The determinism check is now re-run, not remembered

The build-versus-search finding could have become a sentence in a report. Instead
`ragpipe ann` re-runs it every time: on the first build it sweeps twice and records
`within_build_identical` in the payload. If search ever stops being deterministic, the
reported range is too narrow and the report says so on its own.

### Eight tests where the renderer had none

`render_ann_report` had zero tests, in a module already caught shipping a stale literal
(prose reading `ef=16 → 0.9567` above a table saying 0.9367). Two of the new tests are a
structural guard on that whole class:

1. Every `d.dddd` in the rendered markdown must be derivable from the payload.
2. Changing a payload figure must change the prose.

The second exists because the first passes trivially if the prose quotes nothing. Together
they make a hardcoded figure a test failure rather than a review finding — and writing them
immediately caught **me** doing it again: my first rewrite of the headline paragraph had
`0.9883`, `0.9000`, `0.9900-1.0000` and `13,423` typed into the renderer.

### Packaging: 973 MB, 516 MB of which the API never touches

The container was going to be ~1.2 GB. Measuring first showed the full dependency set is
973 MB installed with torch alone at 516 MB, and that the served path uses none of it —
`serve` is BM25 (Python + scipy) plus the verifier (pure Python) plus an HTTP call. A guard
that blocks `torch`, `transformers`, `sentence_transformers`, `qdrant_client` and
`anthropic` at import time confirmed the whole serve path still imports.

Split into extras (`embed`, `vectorstore`, `bedrock`); image is **569 MB**. The imports
were already lazy, so this only made the packaging honest about what runs where. It is also
the same finding as the ANN table: the parts of this pipeline that need a GPU-shaped
dependency tree are the parts that measurably did not pay for themselves at this scale.

Two follow-ons: `ROOT` was derived from the source layout (`parents[2]`), which resolves
inside site-packages once the package is installed — a container would have reported zero
chunks rather than failing to start, so `RAGPIPE_ROOT` now overrides it. And the Gemini key
was readable only from `.env.gemini`; containers pass secrets as variables, so the
environment is now checked first, with a test that an exported-but-empty variable falls
through instead of becoming the key.

### The compose healthcheck that marked a healthy Qdrant unhealthy

`test: ["CMD-SHELL", "exec 3<>/dev/tcp/..."]` — the Qdrant image ships no curl, wget or nc,
so the check has to open the socket itself, and `/dev/tcp` is a **bash** builtin while
`CMD-SHELL` runs `sh`. Result: `cannot create /dev/tcp/127.0.0.1/6333`, five failing
retries, and a container reported unhealthy while answering requests correctly. The image
does have bash; the check is now `["CMD", "bash", "-c", ...]`.

### The dashboard shows both sides, including when they match

*(This describes the first version. See **UI pass** above for what shipped after it.)*

One HTML string, no build step, served from `/` by the same process. Its centre is a
per-citation panel with **the model's quote beside what the document actually says**, plus
the verdict and the character span.

Both sides are rendered even when identical. Showing the source only on a mismatch would
make `verified` the single state in which the reader cannot check the work — and `verified`
is the claim most worth being able to check. `line_number_ambiguous` gets its own colour
and is never green, because Phase 5's gate caught that tier verifying a model's "40 CFR"
against a document's "21 CFR".

The page duplicates `VERIFIED_METHODS` in JavaScript, which can drift, so a test asserts
the two lists still agree: otherwise a green badge in the browser would go on claiming a
citation was verified after the report stopped counting it.

### Per-stage timing, from real requests

Plan deviation #8 asked for per-stage p50/p95 and cost per query. From one live request
through the container — **`n=1`, and the column says so**:

| stage | observed (n=1) |
|---|---:|
| retrieve — BM25 over 13,423 chunks | 0.44 ms |
| generate — Gemini, network | 6257 ms |
| verify — locate every quote in the source | **0.012 ms** |

An earlier draft of this table headed that column `p50`. A percentile over one observation
is just the observation, and this project has been caught once already presenting a mean
over one surviving sample as a headline. `/stats` is where real percentiles live, over
whatever the process has actually served; these three are a *shape*, and the free-tier
daily quota is why `n` is 1.

**Carried as a gap:** these figures live in no `reports/*.json`. That makes them exactly
the "measurement the project cannot re-run" this same entry names as a recurring defect —
identified by the Phase 7 audit gate, not yet closed.

"The verifier is free" is now a measurement. `/stats` reports these over requests actually
served rather than from a synthetic benchmark, and an unused stage returns `None` with
`n: 0` rather than `0.0` — a zero would read as "generation is instant", which is the most
flattering possible lie about this pipeline.

Cost stays in tokens. `cost_usd` is null with a note saying why, and is computed only if
`RAGPIPE_PRICE_IN` / `RAGPIPE_PRICE_OUT` are both set — one rate alone produces a number
wrong in a plausible direction, which is worse than no number.

### Two live requests, and one real failure caught in the wild

The first request ever served through the full path returned 4 citations: 3 `exact`, and
one **`wrong_chunk`** — a real sentence from the retrieved context, attributed to the wrong
chunk id. Precision 0.75 on a genuine query, unprompted. Exactly the failure the verifier
exists to catch, and a better demo artifact than a clean run.

### The two gates, and the five findings worth carrying

**1. A guard I wrote, described as protection, that could not see the thing it guarded.**
The worst finding of the phase, because it was self-inflicted in the act of preventing
this exact defect. I added a test requiring every `\d+\.\d{4}` in the rendered report to
appear in the payload, then wrote in this log that it *"caught **me**"* typing `13,423`
into the renderer. The regex cannot match `13,423` — no decimal point — and one instance
was still sitting at `vectorstore.py:322` while the test passed. It also could not see
percentages (`66.7%`) or millisecond figures (`3.39`), and the historical defect it cited
was *"0.9567 while only 66.7%"*, whose second half it would still miss today. Worse, the
allowed set was built from the *same* payload used to render, so a literal matching the
current run passed **by construction** and only became detectable on the next run —
exactly when it does damage.

Replaced with a mutation guard: render once from the real payload, once from a payload
whose every leaf is a distinct sentinel, and any numeric token appearing identically in
both outputs is by construction not a function of the payload. It needs no model of how
figures are derived, catches every token shape, and fails on the current run. Ten tokens
legitimately survive (the exact baseline's `1.0000`, Qdrant's documented
`indexing_threshold` default, `ef` bounds quoted in historical narrative, percentile
labels, list numbering) and each is allowlisted **with a reason**, plus a test that fails
if an allowlist entry stops being needed — a stale entry is a hole waiting for a
coincidence.

**Lesson: a guard's coverage claim needs testing as much as the thing it guards.** Three
of this phase's findings were tests that passed against the defect they named.

**2. The verifier's most important claim was reordering-blind.** `within_build_identical`
fed a report sentence reading "**two sweeps over the same index are bit-identical** at
every `ef`". It compared each `ef`'s mean recall and exact-match *count* — both invariant
under reordering and under swapping which queries succeeded. Two genuinely different
result sets could satisfy it, and rank order is precisely what this report's own headline
says matters. Now compares the ranked chunk ids themselves via a new `topk_lists`.

**3. The artifact kept the conclusion and threw away the evidence.** `aggregate_builds`
received five values per cell and persisted median, min and max — so two of five were
unrecoverable and nobody, including the audit gate, could check that the published median
was the median of anything. Identical in shape to Phase 6 shipping a faithfulness figure
whose judge verdicts were never written down. Every per-build value is now in the payload
under `*_by_build`, with a test asserting the published median equals
`statistics.median` of it. The same finding also caught the docstring disparaging means
while `statistics.median` computed one at every even build count.

**4. A zero-hit query spent quota and answered anyway.** `/query` had no short-circuit for
an empty context: the model received no chunks, was billed a daily quota unit, and
**produced an answer with a fabricated citation** that verified `unverified`. Ungrounded
output through the one path with no retrieval evidence at all — in the project whose
entire premise is that this cannot happen. The suite already guarded the same cost on the
*deliberate* path (`retrieve_only`) and not on the accidental one. Now refuses with
`refusal_source: "no_context"` before generating.

**5. Claims that described one build as though they described all five.**
`indexed_vectors` and `full_scan_threshold` were single variables reassigned inside the
build loop, so the report printed the *last* build's values in a sentence that had just
said "over 5 independent index builds" — and `assert_indexed` returns for any nonzero
count, so a partially-indexed build would have been overwritten and the report would still
have claimed the full corpus. Now collected per build, with the minimum reported and the
array persisted.

Also fixed: `--builds` defaulted to 3 while this log claimed five (and only the Makefile
passed five); the p50 convention differed between `cmd_ann`/`sweep_recall`
(`sorted[n//2]`, not a median at even n) and `aggregate_builds`/`Timings`
(`statistics.median`) — one convention now, named in the payload; the dashboard's central
"both sides always shown" decision was untested and a reviewer implemented the rejected
behaviour with the suite still green; the endpoint test asserted two hardcoded strings
instead of consulting the app, so it could not notice the page fetching a route that does
not exist; the dollar-figure guard caught only `$` followed by a digit, so a JS-concatenated
cost from an invented rate passed; `_float_env` accepted negative and NaN prices, and NaN
is not valid JSON, so `/stats` could emit a body no strict parser accepts; `Timings.record`
accepted unknown stage names and accumulated them where nothing would ever report them;
`sweep_recall` timed queries it then excluded from `n_queries`, so two figures in one row
described different populations; `from_dense` died on a bare `ModuleNotFoundError` without
naming the `vectorstore` extra; and `test_vectorstore.py` was unguarded on a core-only
install, where 14 tests errored and two failed *misleadingly*.

**The dependency-split claim is now a shipped test.** `tests/test_packaging.py` blocks
`torch`, `transformers`, `sentence_transformers`, `qdrant_client`, `anthropic`, `boto3` and
`botocore` in a subprocess, then imports the whole serve path, serves every route, and runs
a real BM25 retrieval. The claim had been verified once by hand in a scratch script —
the same pattern this entry names as a recurring defect, protecting the phase's entire
packaging story.

**One false positive from me, not the gates:** my first packaging fixture set
`embed_text` to `"x"`, and `prepend` mode indexes `embed_text`, so the corpus tokenized to
an empty vocabulary and bm25s raised an opaque `max() arg is an empty sequence`. My
fixture, not a defect — logged because the failure looked like one.

### Carried forward

- **Sparse vectors in Qdrant are still not implemented.** The locked decision says named
  dense *and* sparse with server-side fusion; what exists is dense-only. "Hybrid lives in
  one store" is not yet true, and the ANN result is the reason it has not been urgent.
- `ragpipe ann --builds 2` silently overwrites a 5-build report with a weaker artifact of
  identical shape. `n_builds` and the `*_by_build` arrays are recorded in both files, so it
  is self-describing rather than guarded.
- ~~**The per-stage serving latencies live in no `reports/*.json`.**~~ **Resolved
  (Phase 8b):** `ragpipe bench` writes `reports/serving_bench.{md,json}`, and it spends no
  quota — verification replays stored citations and generation is read from the stored
  Phase 5 run. See the Phase 8b entry above.
- ~~**`title_lookup` is saturated.**~~ **Resolved (Phase 8):** the slice was never
  saturated — `hit@10` was. Primary metric moved to `hit@1` (spread 0.8500–0.9667, 3.0
  binomial se, against `hit@10`'s 2 distinct values). No retrieval re-run; see the Phase 8
  entry above.
- **`ef=4` and `ef=8` are bit-identical on every field, in every build, in both runs.**
  Qdrant appears to raise `hnsw_ef` to at least the requested `limit` (`k=10`), so neither
  row measures `ef<k` and the illustrative "at `ef=4`" claim is really about `ef=10`.
- The `full_scan_threshold` caveat is unresolved: segment layout is decided per build, so
  it is now a candidate contributor to the build-to-build spread as well.

## Phase 7a — the serving path, and what the vector store actually costs

**Status: SUPERSEDED by Phase 7b (above).** This entry is kept as the record of what that
run measured and concluded. **Its headline figures have been retracted:** the measurement
was produced by an ad-hoc script, and re-running it from a shipped command over five
independent index builds reproduced neither the recall nor the top-k agreement reported
here. Read every number below as "what one unreproducible build reported", not as current.
The current figures are in Phase 7b and in `reports/ann_recall.md`.

### The measurement Phase 3 set up three phases ago

`dense.py` has searched exactly — a full matrix product — since Phase 3, and said why:

> An ANN index has its own recall curve, so an approximate dense row conflates two
> effects: how good the embedding model is, and how much the index gave up to be fast.
> [...] Measuring the ceiling first means Phase 7's ANN configuration can be checked
> against a known exact baseline.

Qdrant 1.19.0 over HTTP, HNSW `m=16 ef_construct=100`, 13,423 vectors × 384 dims, 60
queries at k=10, loaded **from the same cached vectors the exact baseline uses** so the
delta is attributable to the index and nothing else:

| `hnsw_ef` | index recall@10 | top-k identical | p50 ms |
|---:|---:|---:|---:|
| 4 | 0.8567 | 0.2000 | 2.64 |
| 8 | 0.8567 | 0.2000 | 2.82 |
| 16 | 0.9367 | 0.5167 | 2.77 |
| 32 | 0.9750 | 0.7833 | 2.74 |
| 64 | 0.9950 | 0.9500 | 3.13 |
| **128** (labelled "default" in this run — see note) | **1.0000** | **1.0000** | 3.15 |
| 256 | 1.0000 | 1.0000 | 3.36 |
| **exact (numpy, in-process)** | 1.0000 | 1.0000 | **0.48** |

**Note added by the Phase 7 audit gate:** this run labelled `ef=128` as Qdrant's default,
which the five-build run does not support. Measured over the same five builds, the row with
`hnsw_ef` *unset* (recall 0.9883 / agreement 0.8833) and the row with `ef=128` explicitly
set (0.9950 / 0.9500) differ, so 128 is not what an unset request gets. `ragpipe ann` now
measures the default as its own row with nothing set, rather than assuming which swept
value it equals.

Also: `ef=4` and `ef=8` are bit-identical on every recall and agreement field in both this
run and the five-build run, across all builds. Almost certainly Qdrant raising `hnsw_ef` to
at least the requested `limit` (`k=10`), which means both rows measure the same effective
setting and neither is really measuring `ef<k`.

### The headline is a negative result about a locked decision

**[RETRACTED — see Phase 7b]** **At default settings approximation costs nothing** — recall and top-k agreement are both
1.0000. **And the vector database is roughly 6× slower than the thing it replaces.**

Before blaming HNSW, the gap was decomposed: a bare HTTP round trip to the same container,
performing no search at all, measures **2.12 ms p50**. So it is ~2.1 ms of transport plus
~0.9 ms of search, against 0.48 ms of doing the matrix product in-process. **The finding is
about scale, not about Qdrant:** 13,423 × 384 is trivial, so there is no approximation
worth making and no index worth a network hop.

This is the same shape as Phase 3's "hybrid always wins is false" — take the received best
practice, measure it, and on this corpus it does not pay. It also vindicates the Phase 3
decision to keep the eval harness exact: had the ablation used ANN, every dense and hybrid
row would have carried invisible index error in exchange for nothing.

The more decision-relevant column is **`top-k identical`**, which degrades far faster than
recall — 0.8567 recall at `ef=4` but only 20% of queries returning the identical top-10.
Rank order is what feeds fusion and reranking, so recall alone would have understated what
low `ef` costs. Reading recall alone here is the same trap as reading `recall@k` alone
across chunk sizes in Phase 4.

### Three ways this measurement silently measured nothing

Each produced a complete table of perfect numbers, and each looked like a successful run.

1. **Qdrant local mode brute-forces every query.** The first sweep, against
   `location=":memory:"`, reported index recall **1.0000 at every `ef` from 4 to 256** — it
   was comparing exact search against exact search. Qdrant emits a `UserWarning` saying so,
   which was nearly scrolled past. `sweep_recall` now refuses to run against a local client
   and names how to measure it properly. The module docstring had also claimed local mode
   "uses the same indexing code as the server", which is true of storage and false of
   search.
2. **A server will not index a small collection.** `indexing_threshold` defaults to 20,000
   KB and this corpus is about 20 MB across several segments, so a default collection would
   also have answered every query exactly. The store now forces indexing, **and
   `assert_indexed` blocks until Qdrant confirms the vector count** — the setting having
   worked is a claim, the count is evidence.
3. **The report generator carried stale literals.** The prose said "at `ef=16` recall is
   still 0.9567 while only 66.7% of queries return the identical top-10" above a table
   reading 0.9367 and 0.5167 — the previous run's numbers, hardcoded into a renderer
   written minutes earlier. This is exactly the defect Phase 4's audit found four times.
   Every quantitative claim in that prose is now derived from the payload, including which
   row it picks as illustrative.

### Caveats recorded rather than resolved

- **HNSW is not bit-reproducible.** Two sweeps over the identical 60 queries gave `ef=4`
  recall 0.8617 and 0.8567, `ef=16` 0.9567 and 0.9367. Segment search order varies, so the
  low end of the curve is good to about a point.
- **`full_scan_threshold` is 10,000 KB**, so Qdrant may still full-scan individual segments
  below that size. `hnsw_ef` demonstrably moves recall, which proves HNSW is in use — but
  this likely explains why full recall arrives so easily, and only a larger corpus would
  separate the two.
- **Latency is a floor, not a forecast.** Single client, no concurrency, container on the
  same host. Useful for comparing settings and for the transport-versus-search
  decomposition; useless for capacity planning.
- **No crossover estimate.** Qdrant earns its place at a scale this corpus does not reach,
  and no number is invented for a corpus that does not exist.

### Also built, not yet exercised

`src/ragpipe/service.py` — FastAPI with three separately timed stages (`retrieve`,
`generate`, `verify`) and cumulative token accounting, which is plan deviation #8. Two
decisions worth noting: cost is reported in **tokens with `cost_usd: null`** unless real
per-million rates are supplied via environment, on the same reasoning that keeps
`price_per_mtok` at `None` in the generator registry; and the service **degrades to
retrieval-only without a generator key** rather than refusing to start, because the
retrieval half of this project is fully local and a demo that cannot boot without a
third-party credential is a worse demo.

### Environment

Docker is now available on this machine via **Colima** (`colima start`) plus the Docker
CLI, rather than Docker Desktop — free, no GUI, no admin prompt, and it avoids Docker
Desktop's commercial-licensing threshold (free only under 250 employees *and* $10M revenue)
should this project ever touch a work machine. Qdrant runs as
`docker run -p 6333:6333 qdrant/qdrant`.

## Phase 6c — the first two-tier faithfulness numbers

**Status: shipped — both verification gates passed.** `reports/generation_eval.{md,json}`
carries both tiers. **708 tests, lint clean.**

28 findings across the two gates (10 numerical, 18 code review), every one reproduced
before being acted on, every one fixed; **zero false positives**. The audit recomputed
~790 figures and found **zero arithmetic errors** — the fourteenth consecutive pass with
that result. Golden set: **159 accepted of 198** across 20 documents.

### The numbers

**Read the tier-2 figure as a range.** Three re-judgements of the *same* 8 answers by the
same judge at temperature 0 returned **0.875, 0.750, 0.875**. At n=8 one answer is 0.125,
and `title-0000` is the only *consistent* failure — the CDRH/CBER attribution below. So
the honest statement is 0.75–0.875 with one reproducible defect, not a point estimate.
Temperature 0 is not bit-reproducible on this API, which was already noted for the
calibration and is now measured for the headline.

`gemini-3.6-flash` generating, BM25 top-5, 16 queries; `gemini-3.1-flash-lite` judging.

| Measure | Value |
|---|---|
| Scored | 16/16, 0 errors |
| **Tier 1: citations located** | **22 of 22** (18 exact, 4 normalized) |
| — unverified / wrong_chunk / too_short | **0 / 0 / 0** |
| Macro citation precision | **1.000** |
| **Tier 2: answers fully supported** | **7 of 8 = 0.875** (range 0.75–0.875 over 3 runs) |
| — verdicts | supported 7, partial 1, unsupported 0 |
| Refusal on `unanswerable` | **4/4** |
| Refusal on answerable, excl. mention-labelled | 0.125 (1 of 8) |
| Separability AUC | **1.000** both pools |
| Judge calibration in-run | negatives 1.000, positives 0.750 |
| Tokens | 29,811 prompt / 4,215 output / **0 thinking** |

The single non-supported answer is a precise catch: `title-0000` attributed a guidance to
"the FDA's Center for Devices and Radiological Health (CDRH) and Center for Biologics
Evaluation and Research (CBER)", and the judge's reason is that the quotes do not say so.
Tier 1 passed it — the quotes were real — which is the entire argument for a second tier.

Tier 1 at 22/22 is a genuine change from Phase 5's 19/21, and the plausible cause is the
prompt work done for the golden set: the footnote, contiguity and no-repair rules were
added to the *drafting* prompt, but `answer.py`'s rules 3-5 were already close, and this
run used a different generator. Not attributed, because two things changed.

### The tier-2 unit was wrong, and the first number measured the unit

The first two-tier run reported `supported_of_located = 0.136` with **18 of 22 verdicts
`partial`**. That is not a faithfulness measurement. Each citation was being judged
against the *whole* answer, so an answer citing six quotes collects five partials by
construction. Of the 22, three came back `supported` (0.136 = 3/22) and the two
single-citation answers were among them, which is the pattern; the third was one quote
inside a multi-citation answer. An earlier version of this paragraph said "the two
single-citation answers were exactly the ones judged supported", which does not add up
to 0.136 — the audit gate caught the slip.

The code comment had *predicted* this ("`partial` is the expected result for one citation
of several") and the report printed the resulting rate anyway. Predicting an artifact is
not the same as accounting for it.

Fixed by making the unit **one item per answer, carrying all of its located quotes**: does
this evidence, taken together, support this answer. That is the question a reader has. The
same 8 answers rescored **0.875**. Per-citation judging is kept behind
`--judge-unit citation` as a diagnostic for localising a weak citation, with a warning in
the report that it is not a faithfulness rate.

### Two auditability failures, both caught by trying to re-derive the number

**Units silently mixed.** Under answer-level judging `n_supported` counts answers while
`n_located` counts citations. The report divided one by the other and printed
**7/22 = 0.318** as "the share of citations that survive reading". It is neither a share
of citations nor of answers. `Faithfulness` now carries its `unit`, the citation-level
ratios return `None` under answer-level judging rather than a wrong number, and the report
renders unit-appropriate tables.

**The verdicts were not persisted.** `write_agreement` was called with the *controls*
only, so the per-answer verdicts behind 0.875 were absent from disk. It surfaced
immediately on trying to regenerate the report from the payload: 0 of 8 supported. Now all
items are written, and the figure round-trips — re-deriving from
`judge_calibration.json` reproduces 7/8 = 0.875 exactly.

That round-trip is worth keeping as a habit rather than a one-off check. Both of these
defects were invisible in the output and instantly visible in an attempt to recompute it.

### `--rejudge` paid for itself the same day

Tier 2 costs a few requests; generation costs one per query. Re-running only the judge
against a stored report made correcting the unit cost **3 requests instead of 19**, and
the unit was corrected twice. `load_gen_eval` exists because everything tier 2 needs —
answer text, per-citation methods, matched spans — was already persisted for
auditability, so the capability was free.

### The calibration was measuring one document, and the worst one

The Phase 6 audit gate's most consequential finding. `build_control_set` sorted by
`pair_id` and took a prefix — and golden pair_ids are prefixed by `doc_id`, so **all 8
positives came from a single document**. That document was
`ctgov-NCT00567567-Prot_SAP_000`: the one Phase 6a singled out for spurious intra-word
spaces and a 10/10 → 5/10 drafting regression, and the one whose pairs
`MIN_ANSWER_EVIDENCE_COVERAGE` was calibrated on before being used to reject its own
`gold::009` at 10% coverage.

So the 0.750 positive rate was a measurement of the corpus's worst document, and Phase 6b
extrapolated it to "roughly a quarter of golden pairs". That extrapolation is withdrawn.

`golden.sample_for_review` had already solved this — it spreads across documents
round-robin, for exactly this reason, with a comment saying so. **The lesson did not
travel between modules.** Controls now spread the same way: 8 positives from 8 documents.

Fixing it introduced a second bug, which its own new test caught immediately. The negative
construction offset by `len(usable) // 2` and skipped same-document collisions; under
round-robin ordering with D documents interleaved that offset is a multiple of D, so
*every* candidate collided and the function returned **zero negatives**. Zero negatives
makes `negative_rate` None and `usable` False, so it fails safe — but silently, and the
calibration simply stops existing. Now each claim walks forward cyclically to the first
pair from a different document, which is correct under any ordering.

### Verification gate 1 of 2 — numerical audit

**~790 figures recomputed independently from raw data, no `ragpipe` imports. 783
reproduced exactly.** 4 numerical mismatches, 3 label/unit defects, 3 low-severity
wording issues, 5 unverifiable. **Zero arithmetic errors** — the fourteenth consecutive
pass with that result, and again every defect was a claim rather than a computation.

The audit independently re-verified **all 22 citations** with its own span-location logic
(zero disagreements on method), re-derived **0.875** from the per-item verdicts, and
re-ran **all six golden-set validation rules** over 198 pairs with its own
implementations — **594 per-pair comparisons, zero disagreements**, including every
embedded percentage inside every `reject_reason` string.

| # | Finding | Severity | Reproduced |
|---|---|---|---|
| 1 | `MAX_LINE_NUMBER_DENSITY` claims it excludes 11 of 152 documents; it excludes **21** | HIGH | yes |
| 2 | `0.725` quoted beside "on `structural`" is `semantic`; structural's best min-max is **0.667** | HIGH | yes |
| 3 | The calibration's controls all came from **one document** — the corpus's worst | HIGH | yes |
| 4 | `MAX_ANSWER_OVERLAP` runner-up claimed 0.304; actual **0.381** | MID | yes |
| 5 | "validated against the 32 persisted items" — the table sums to **16** | MID | yes |
| 6 | Phase 6a's evidence-methods column counts pairs in rows 1–2 and quotes in row 3 | MID | yes |
| 7 | "0.136 with the two single-citation answers supported" needs **three** supported | LOW-MID | yes |
| 8 | `p50` interpolated beside a nearest-rank `p95`, unstated | LOW | yes |
| 9 | "0.04–0.18 drafts well" — the tested band starts at **0.014** | LOW | yes |
| 10 | "the first 93 accepted pairs" names a population the next rule shrinks to 84 | LOW | yes |

**Finding 2 is the one worth dwelling on, because it is a fix that failed in the same
dimension it was fixing.** Phase 5's audit gate caught "0.725 for *any* fusion
configuration" and I corrected the *scope* to "min-max fusion" — without checking the
*population*. 0.725 is min-max on `semantic`; on `structural`, which is what every one of
these commands actually runs, min-max reaches 0.667. So the corrected claim was still
wrong, one level down, and it still flattered the same comparison. Now fixed in seven
places with the chunking named on every figure.

**Finding 3 changed a number and a conclusion**, and is written up in its own section
above. Fixing it also introduced a zero-negatives bug that its own new test caught
immediately.

**Finding 1** is a ~2× understatement: 11 is the count at density ≥ 0.6415, not at the
0.35 threshold that ships. The gate's own supporting figures all reproduced exactly
(median 0.0478, p90 0.5122, the 101-character digit-free run), which is what made the
outlier obvious.

### What the audit could not verify, and what happened to it

Five items. Four are populations that no longer exist on disk (pre-regeneration batches,
Phase 5's overwritten `19/21`) and are correctly labelled as historical. The fifth was
actionable: the **short-fragment proxy figures** ("22nd of 152, 0.0843 against a median of
0.0589") were computed ad hoc, never committed, and did not reproduce under **32 plausible
definitions** — the document's rank ranged 5th to 127th. Those figures are now
**withdrawn**, on the same standard this project already applied when `citations.py`
deleted Phase 4's hand-counted line-number figures. The conclusion they supported survives
unchanged: under every definition tried the document is unremarkable, so the phenomenon is
not a separable class and a detector would quarantine good documents.

### Verification gate 2 of 2 — code review

**18 findings, all confirmed and reproduced, all fixed. Zero false positives.** No
arithmetic error anywhere — the fourteenth consecutive pass with that result. What is new
is the *shape* of the worst findings: three of them were checks that **did not run at
all**, and one was a safeguard that ran and did the opposite of what it promised.

**708 tests, lint clean.** The golden set moved 157 → **159** accepted (re-validation from
the drafted state can restore a pair, not only reject more).

| # | Finding | Severity |
|---|---|---|
| 1 | `--rejudge` with a **failed** judge shipped the previous run's faithfulness figure | HIGH |
| 2 | Unjudged items were scored as tier-2 **failures** | HIGH |
| 3 | Coverage was blind to 1–2 digit numbers, modals and negation; and three checks were never called | HIGH |
| 4 | `MAX_LINE_NUMBER_DENSITY` excludes 21 documents, not 11 (also found by the audit) | HIGH |
| 5 | `line_number_density` could not see **leading** line numbers — the commoner layout | HIGH |
| 6 | A constructed "negative" can be genuinely supported; `doc_id` inequality is insufficient | MID |
| 7 | The citation-unit scorer scored refused answers the judge was never asked about | MID |
| 8 | `usable: true` off one control per side; `build_control_set` under-delivers silently | MID |
| 9 | The calibration artifact recorded no judge model, and two commands overwrite one path | MID |
| 10–11 | `--rejudge` crashed without `--judge`; and judged empty claims on a pre-Phase-6 report | MID |
| 12 | Check order let "boilerplate" mask a fabricated quote | LOW-MID |
| 13–16 | Four rationale claims false or overstated of the shipped code | LOW-MID |
| 17 | Eight tests that cannot fail or do not test their name | MID |
| 18 | Dead `if True:` hiding an unconditional call | LOW |

#### The three checks that never ran

`drop_redundant`, `flag_factual_risk` and `sample_for_review` had **zero call sites
outside `golden.py`**. `cmd_golden` called only `validate_pairs` and `yield_report`. So:

- the redundancy gate never ran in the pipeline — the single `redundant:` reject on disk
  was produced by an ad-hoc script, and a second `make golden` could not have caught a new
  duplicate;
- the factual triage this log presents as *"the check that answers 'is the reference
  answer right'"* **never executed**;
- Phases 6a and 6b both claim re-validation is "free to re-run", which was true of a
  throwaway script and of no shipped code path.

All three are now wired through `_finish_golden`, and `ragpipe revalidate` exists — it
re-runs every check against the pairs on disk with no API calls, dry by default. Running
it reported the redundancy rejection and a clean factual triage for the first time.

#### Coverage was blind to exactly the content that matters

`_content_words` dropped tokens of ≤2 characters, and `_COVERAGE_STOPWORDS` contained
`shall`, `should`, `must`, `may` and `not`. Measured against a real-shaped quote
("...shall submit the report to the Agency within 30 days..."):

| golden answer | coverage before | after |
|---|---|---|
| "within **60** days" | **1.000 — accepted** | 0.875 |
| "**may** submit" | **1.000 — accepted** | 0.875 |
| "shall **not** submit" | **1.000 — accepted** | 0.889 |
| roles swapped (Agency↔sponsor) | **1.000 — accepted** | 1.000 |

Digits are now kept regardless of length and the modals and negation are out of the
stoplist. **But note the last row, and the fact that 0.875 still clears the 0.45 floor.**
Coverage is a bag of words: one word in eight barely moves it, and word order is not
modelled at all. So it does **not** catch negation, modal weakening or role reversal, and
no threshold on it will. That limit is now written into the constant rather than implied
away — the constant previously named only paraphrase and vocabulary reuse. Numbers are
additionally covered by `unsupported_numeric_claims`, which is now actually called.

#### A safeguard that did the opposite of its promise

`_run_tier2` only *wrote* `payload["faithfulness"]` when calibration passed; it never
removed it. On the fresh-run path the payload has no such key, so the promise held. On
`--rejudge` the payload is **the previous report loaded from disk** — so a rubber-stamp
judge that rejected 0% of constructed negatives shipped the stale `7 of 8 = 0.875`, beside
its own failed-calibration paragraph, with exit code 0. The reviewer reproduced it
end-to-end with a fake judge.

It failed on the one path Phase 6c presents as the phase's cost saving, and it falsified
three separate statements: the `--judge` help text, the `_run_tier2` comment, and this
log's *"omitted from the report entirely"*. Now `payload.pop("faithfulness", None)`.

#### Unjudged treated as failed — a doctrine this project states three times elsewhere

`answers_with_citations` was incremented *before* the judgment lookup, so a missing verdict
counted as not-supported. Dropping 3 of 8 verdicts turned **0.875 into 0.500**; dropping 7
of 22 citation verdicts turned 1.000 into 0.682. One swallowed request does that, and
`_run_tier2` swallows a `GenerationError` per batch.

The same file already says *"Errors are counted, never averaged in"*, refusals are already
excluded from citation precision, and `judge.Agreement` already excluded unjudged controls
from its denominator — with a test asserting exactly that. **Two contradictory treatments
of the same concept, in the same phase.**

#### Eight tests that could not fail

Worth listing because the pattern is now this project's most reliable defect:

- `test_a_single_oversized_chunk_does_not_loop_forever` asserted `== []` — it **enshrined a
  bug as intended behaviour** under a name about loop termination.
- `test_stopwords_do_not_inflate_coverage` paired its answer with text sharing *no token*,
  so coverage was 0.0 whether or not stopwords were filtered.
- `test_budget_bounds_every_real_document` asserted `DRAFT_CHAR_BUDGET == 40_000`.
- `test_refusals_contribute_nothing` used `checks=[]`, so it held for the wrong reason and
  let finding 7 through.
- `test_sampling_reaches_the_end` used 765 *identical* chunks, so the trim loop never fired.
- The line-number fixture put numbers at line *ends*, hiding finding 5 for the whole phase.
- `test_threshold_keeps_every_judge_supported_pair` restated the constant against a
  hardcoded 0.471 while the per-pair data sat unread on disk.
- `test_judge_default_differs_from_the_generation_default` was true but enforced nothing;
  `--generator X --judge X` self-graded silently. Now warns.

All eight rewritten to assert the property their name claims, several with an explicit
"could this fixture even fail" guard.

#### A latent bug found by a test failing for the wrong reason

Adding the leading-line-number test raised `JSONDecodeError`. Cause: `str.splitlines()`
splits on U+2028, U+2029, form feed and U+0085, none of which terminate a JSONL record —
and `data/chunks/structural.jsonl` contains **six literal U+2028 characters**, so
`read_text().splitlines()` reports **13,429 lines for 13,423 records** and tears six of
them mid-string. `golden.load_golden` used that pattern. It was safe only because
`write_golden` escapes non-ASCII — a property of the writer, which the reader should not
depend on. Both now iterate the file.


### Caveats that travel with these numbers

- **n = 16 queries, 8 answering.** The free tier allows 20 requests per day per model.
  0.875 is 7 of 8; one answer either way moves it by 0.125.
- ~~**The golden set is spot-checked, not curated.**~~ **Resolved (Phase 8c):** all 159
  accepted pairs were triaged for obligation level, polarity, actor and quantity; the 25
  flagged pairs were each read against their source and judged, and **8 were rejected for
  restating a recommendation as a requirement**. Verdicts are committed in
  `corpus/curation_verdicts.json`. Set is now 152 accepted.
- **The judge's in-run positive rate was 0.750**, and Phase 6b established that this
  reflects golden pairs whose answers outrun their evidence rather than judge error. It
  cleared the negatives gate at 1.000, which is the one that decides usability.
- **Temperature 0 is not bit-reproducible.** The same controls scored positives 1.000 in
  one run and 0.750 in another. The negative side held at 1.000 across all runs.

## Phase 6b — tier 2 of the verifier, and calibrating the instrument first

**Status: judge built and calibrated; usable, with a golden-set caveat it surfaced.**
Artifacts: `src/ragpipe/judge.py`, `reports/judge_calibration.json`, `ragpipe judge` /
`make judge`. **649 tests, lint clean.**

Tier 1 answers "is this quote really in the document". Phase 6a produced the case that
shows what it cannot answer: a golden pair whose quote was real, locatable, and correctly
attributed, supporting an answer that called Structured Product Labeling "an HL7
standard" when the cited chunk never mentions HL7. Every deterministic check passed and
the answer still did not follow. That needs reading, so tier 2 uses a model.

### An uncalibrated judge is worse than no judge

A judge emits confident verdicts whether or not it works, and those verdicts land in a
faithfulness table nobody re-derives. So it is run first against items whose answer is
known:

- **Presumed positives** — golden pairs that passed tier 1 and numeric triage.
- **Constructed negatives** — a real answer paired with a *different document's*
  evidence. These are unambiguous: no reading makes them supported.

The negatives are the half that discriminates. **A judge answering "supported" to
everything scores 1.000 on positives and 0.000 on negatives**, and a positives-only
calibration would call it perfect. Nothing else in this project would have caught it.

The judge model also deliberately differs from the drafting and answering defaults. The
risk is concrete rather than theoretical: the same model wrote the answer and the
reasoning behind it, so self-agreement would partly measure self-consistency.

### The calibration inverted the obvious reading

`gemini-3.1-flash-lite`, 16 positives and 16 negatives, 4 requests, temperature 0 —
reproduced exactly on a second run.

| | rate |
|---|---|
| Constructed negatives correctly rejected | **1.000** (16/16) |
| Presumed positives called supported | **0.750** (12/16) |

The first instinct is that 0.750 fails the judge. It does not. Reading its four
disagreements shows it was right every time:

> *"The quote provides the dosing instructions for mesna but does not mention the
> condition of microscopic or gross hematuria"*
> *"The quote does not mention Dr. Robert C. Seeger or the Neuroblastoma Biology
> Reference Laboratory"*

These are the HL7 defect again: **golden answers that synthesise across a whole chunk
while citing a single span.** So 0.750 is a *golden-set* measurement, not a judge
measurement, and roughly a quarter of pairs are affected.

### Three bugs in the calibration logic, each caught by its own test

The gate was wrong three times before it was right, and every correction came from a
test rather than from review.

1. **Both rates gated symmetrically.** The original rule required positives *and*
   negatives to clear 0.8, so it declared a working judge unusable on the strength of
   defective reference data. The rates are not equally informative: a low negative rate
   is unambiguous, a low positive rate is ambiguous between a bad judge and bad
   references. Now gated on negatives, with positives reported and flagged.
2. **Gating on negatives alone had a hole.** A judge answering "unsupported" to
   everything rejects every constructed negative perfectly and would have passed. Closed
   with a low sanity floor on positives (0.4) — well below the 0.8 warning, so the real
   0.750 case passes while a reject-everything judge does not. The band between remains
   genuinely ambiguous, which is why `--show` prints the disagreements.
3. **Rates divided by judged items only.** A batch where 15 of 16 items went missing
   could report a perfect rate from one survivor. `MIN_JUDGED_SHARE` requires most
   controls to come back.

Two more defects worth recording because they are this project's recurring shapes:

- **Batch interleaving that did not interleave.** Controls are meant to be spread across
  requests so one bad response cannot destroy the calibration. The implementation sorted
  on `(pair_id suffix, item_id)`, and because every golden pair_id ends in the same three
  digits that degenerated to an alphabetical sort — every negative ahead of every
  positive, exactly the grouping it existed to prevent. Its own test caught it.
- **A test that could not fail.** `test_agreement_is_recomputable_from_the_file` ended in
  `if False else True`. Replaced with one that round-trips through disk and re-derives
  the rates, and which fails if they drift.

### What the calibration is for, beyond the number

`write_agreement` persists every per-item verdict, reason, claim and quote — not just the
aggregate. The first calibration had to be re-scored after the gate was corrected, and
without the per-item data that would have meant re-spending quota to recover results
already paid for. It is the same lesson as storing claimed quotes in the generation eval:
**an aggregate that cannot be recomputed has to be trusted.**

The reasons matter more than the rates. They are what separated "the judge is broken"
from "the golden answers outrun their evidence", and no aggregate can carry that.

### Acting on it: a free proxy for what the judge charges quota for

The judge's finding was actionable, and the cheapest way to act on it turned out not to
involve the judge at all. The defect — an answer asserting content its cited quote does
not contain — is largely measurable as **lexical coverage**: what share of the answer's
content words appear in its evidence.

Validated against the **16 judged positives** among the 32 persisted calibration items —
the negatives are not presumed-supported and so cannot calibrate a coverage threshold.
Persisting the per-item verdicts is the only reason this was checkable at all:

| judge verdict | coverage median | range |
|---|---|---|
| supported (n=12) | 0.888 | 0.471 – 1.000 |
| partial / unsupported (n=4) | 0.225 | 0.000 – 0.611 |

Best single-threshold accuracy **0.938 at 0.471**, against a 0.750 majority baseline. The
shipped threshold is **0.45, deliberately below the optimum**, so it keeps every pair the
judge accepted and rejects 3 of the 4 it did not. Erring toward keeping pairs is right
because the judge is the arbiter and this is a pre-filter.

Its limits are stated in the constant, because n=16 is small: one known miss at 0.611, and
it is lexical, so a heavily paraphrased correct answer scores low while a wrong answer
reusing the quote's vocabulary scores high. It does not replace tier 2; it makes tier 2
cheaper by removing the obvious cases before spending a request.

**Effect, measured rather than assumed.** Re-validating locally rejected 12 of 169 pairs
(7%) — noticeably less than the judge's 25%, exactly as a conservative threshold should.
Re-calibrating on the cleaned set moved the positive rate **0.750 → 0.8125**, which
clears the warning threshold, so `inspect_golden_set` is now false. A modest,
proportionate improvement, and the honest reading is that the free filter caught the
worst third of the problem and the judge is still needed for the rest.

The drafting prompt now also requires evidence to **cover the whole answer** — supply as
many quotes as the answer has claims, or drop the claim — rather than "at least one
quote".

Eight existing tests failed when the check landed, all because their fixtures had answers
that outran their quotes. That is realistic of the defect and useless as a fixture for
anything else, since every one of them then rejected for coverage before reaching the
behaviour under test. Rewritten to be internally consistent.

### Composing the two tiers

`ragpipe gen-eval --judge <model>` (or `make gen-eval-full`) now runs both tiers in one
command. The composition is where the design decisions are, and they are all about
denominators.

**Only located citations are judged.** Asking whether a quote that does not exist supports
a claim spends a request on a question tier 1 already answered, and mixing the two would
make the tier-2 rate uninterpretable.

**Three figures, kept apart**, because a located-but-unsupporting citation is a different
defect from a fabricated one:

| figure | denominator | what it answers |
|---|---|---|
| `supported_of_located` | citations that passed tier 1 | of the quotes that are real, how many back the claim |
| `supported_of_claimed` | everything the model claimed | end to end, both tiers |
| `answer_level_rate` | answers with at least one located citation | share whose *every* citation located and supported |

`supported_of_located` is the tier-2 headline. Dividing by all claimed citations instead
would fold tier-1 failures into a tier-2 number. The answer-level rate is stricter and
closer to what a reader experiences: one bad citation in four is a bad answer.

**`partial` is reported separately, never folded in.** The claim judged is the whole
answer rather than a sentence, because splitting an answer into sentences and attributing
each to a citation needs an alignment step that would itself be a source of error. The
consequence is that `partial` is the *expected* verdict when an answer with four citations
is judged against one of them, so counting it as either success or failure would be wrong.

**Controls ride along with every run**, and if the judge fails calibration the
faithfulness block is **omitted from the report entirely** rather than printed with a
caveat. A number in a report gets read; a caveat beside it may not.

An answer with no located citations is excluded from the answer-level rate rather than
scored zero — the same reasoning that keeps refusals out of citation precision.

### Deferred
- **Running it.** Wiring, scoring, the report section and 15 tests are in place; the run
  itself needs a fresh daily quota.
- **A second judge** for agreement-between-judges, now cheap: the interface takes any
  registered model and quota is per model.

## Phase 6a — the golden answer set: drafting harness and calibration

**Status: harness shipped, first 100 pairs drafted, awaiting curation.** Artifacts:
`src/ragpipe/golden.py`, `corpus/golden.jsonl`, `ragpipe golden` / `make golden`.
**589 tests, lint clean.** No verification gate yet — that runs when the set is curated.

### The design decision that paid for itself immediately

The drafter is required to supply verbatim `evidence` quotes for its own reference
answer, and validation runs **the Phase 5 tier-1 verifier over them**. A pair whose own
evidence cannot be located in the chunk it cites is rejected before a human reads it.

This costs nothing — no model call, no quota — and it caught 7 of the first 100 pairs.
A golden set sits upstream of every generation metric, so a hallucinated reference
answer does not fail loudly: it makes a correct model look unfaithful, permanently, in a
table nobody re-derives. Holding the gold to exactly the standard the model under test
is held to is the cheapest possible insurance.

It also means validation is **free to re-run**, which is what made the calibration loop
below possible without burning quota on every iteration.

### Batched drafting, and the arithmetic that made it feasible

Deviation #15. The free tier allows 20 requests per day per model, counted per *request*
against a 1M-token context. An earlier estimate in this project put Phase 6 at a week by
assuming one pair per call. Ten pairs per call makes 200 pairs ~20 requests — one
session. Same error the Phase 4 contextual-retrieval estimate made before grouping, and
it is now written into the plan so it is not made a third time.

Prompt size is bounded by an **attention** budget, not a context limit. Chunks per
document run median 46, p90 234, max 765, and the largest document renders to ~870,000
characters — about 217k tokens, comfortably inside the window. But handing a model 765
excerpts and asking for ten questions anchors it on the opening few, so
`select_chunks_for_drafting` stride-samples to 40,000 characters: median 30 excerpts per
call, max 68, and 72 of 152 documents still pass whole. Striding rather than truncating
because a prefix of a 765-chunk protocol is its title page and table of contents, which
supports almost no answerable question.

### Three calibration rounds, each driven by reading the rejects

| round | prompt | accepted | evidence methods |
|---|---|---|---|
| 1 | baseline | 36/40 (90%) | 2 exact / 34 normalized |
| 2 | + footnote and contiguity rules | 35/40 (87.5%) | **21 exact** / 14 normalized |
| 3 | + no-repair rule | **93/100 (93%)** | 29 exact / 70 normalized (quotes, not pairs) |
| 3 re-validated | + boilerplate filter | **84/100 (84%)** | — |

The evidence-methods column is not a single unit down the page: rows 1–2 count one method
per accepted *pair*, row 3 sums to 99 because eight pairs carry two quotes. Flagged by the
audit gate; the "ten-folded the share of byte-exact quotes" reading holds either way
(2/36 = 5.6% → 21/35 = 60%).

**Round 1's four rejects were near-misses, not fabrications** — best-window word
similarity 0.85–0.90 on three of them. Diagnosed precisely: inline superscript footnote
markers extract as separate digit tokens (`partners 2`, `information, 8`) and the
drafter either joined them to the neighbouring word or dropped them; plus one quote
spliced across a gap.

**Both were fixed in the prompt, deliberately not in the verifier.** A tolerance for
dropped digits is exactly what Phase 5 added and had to withdraw — it scored a model's
"40 CFR" as a verified citation of a document's "21 CFR", because "dropped a footnote
marker" and "changed a regulation number" are the same edit. Round 2 took both FDA
guidances from 8/10 to 10/10 and roughly ten-folded the share of byte-exact quotes.

**Round 2 also regressed one clinical protocol from 10/10 to 5/10, and that was a corpus
finding.** All five rejects had one cause: the source text contains spurious intra-word
spaces — `communi cable`, `requir ed`, `coll ected`, `vi able`,
`9-dimethylaminomethyl-10-hydroxycamptot hecin` — and the drafter was silently
*repairing* them, which makes the quote unlocatable. A fifth text-layer defect class,
the mirror image of `space_collapsed`.

**It did not earn a detector, and that negative result is the point.** A short-fragment
share proxy over all 152 indexable documents put the offending protocol unremarkably
mid-pack rather than as an outlier.

The specific figures that sentence used to quote (22nd of 152, 0.0843 against a median of
0.0589) are **withdrawn**: the proxy was computed ad hoc and never committed, and the
Phase 6 audit gate could not reproduce them under 32 plausible definitions of
"short fragment" — the document's rank ranged 5th to 127th depending on the definition.
This project already deletes Phase 4's hand-counted line-number figures for exactly that
reason, and the same standard applies here. **The conclusion is unaffected and is what
matters:** under every definition tried the document is unremarkable, so it is not a
separable class and a gate on this signal would quarantine good documents. The phenomenon is a mild,
widespread property of PDF extraction, not a separable class of bad document, and gating
on a signal that does not separate would quarantine good documents. Fixed in the prompt
instead ("do not repair the text"), which took the protocol to 9/10.

### Boilerplate: measured, filtered, and still not solved

Reading the round-3 sample surfaced a defect no verifier would catch: three of eight
pairs asked where to mail written comments. Such pairs verify perfectly and measure
nothing — the answer is identical across dozens of documents, so retrieving the *right*
document is not required, and a model scores by recognising a template. Two pairs were
literally the same question phrased twice.

Measured over the 93 pairs accepted *before this rule existed*, evidence-document spread
separates cleanly: **82 in exactly 1 document, 2 in exactly 2**, then a gap to 3, 6, 9,
10, 14, 30, 30, 39, 39. (The rule then rejected those nine, leaving 84 — so "93 accepted"
named a population that includes the pairs it was about to discard.) The
worst is the standard nonbinding-recommendations disclaimer, in **39 documents**. The
threshold is >2 rather than >1 on purpose: the corpus deliberately contains 8 complete
draft/final guidance pairs that legitimately share long passages, so a quote in exactly
two documents is plausibly a real version-currency question.

Re-validating the existing pairs — free, no quota — removed exactly the 9 predicted.

**The limitation, stated because a fresh sample still shows it:** spread catches
*identical* boilerplate, not *near-duplicate* boilerplate. "Dockets Management Staff
(HFA-305)" and "Division of Dockets Management (HFA-305)", "Room 1061" and "rm. 1061"
are different strings, so each appears in one or two documents and survives the filter.
Embedding-based near-duplicate detection would catch it, and is deliberately not built
before a human has looked at a batch — this is what the curation pass is for.

### Cross-pair redundancy: the near-duplicate case spread cannot catch

Two of the 84 pairs asked where to mail written comments, of different documents, and
their answers were the same FDA docket address — one via `Dockets Management Staff
(HFA-305)`, one via `Division of Dockets Management (HFA-305)`. Both had evidence in
exactly 2 documents, right at the spread threshold, so both survived it. Spread asks
"does this exact string appear in many documents", and wording drift ("Room 1061" versus
"rm. 1061") makes each variant look document-unique.

The redundancy is *between pairs*, so it is measured between pairs. Jaccard over
lowercased word tokens, threshold 0.7: the duplicate scored **0.750** and the next
highest overlap among all 3,486 pairings was **0.304**, a wide empty gap. Exactly one
pair was removed. Embeddings were not needed — these are near-identical boilerplate, not
paraphrases.

**Final: 83 of 100 accepted.** 7 unverifiable evidence, 9 boilerplate by document
spread, 1 redundant by answer overlap.

### Curation: what actually happened, stated precisely

Deviation #5 specifies **human-curated**, and that is not what this batch received. A
review file was produced (`reports/golden_review.md`, 84 pairs, 12 pre-flagged by
mechanical heuristics), and the response was that it looked fine, with no pairs named
for removal. That is a **spot check, not a curation pass**, and the distinction is worth
recording because "150–200 LLM-drafted and human-curated pairs" is a stronger claim than
the provenance currently supports.

What that leaves unverified is specifically **factual correctness of the reference
answers**. Automation proves the evidence quote exists in the chunk it cites; it cannot
prove the answer follows from the quote. On regulatory content that is domain judgment.
Until a real curation pass happens, generation metrics built on this set carry that
caveat, and Phase 8 should either obtain one or describe the set as machine-validated
and spot-checked.

One mechanical failure in the review tooling is worth logging for the same reason the
gates are: the flagging regex used `\bcomment\b`, which cannot match "comments"
because `\b` requires a non-word character and `s` is one. The administrative-boilerplate
flag therefore matched **zero** pairs and presented the batch as cleaner than it was.
Fixed, it matched 5. A filter that silently matches nothing is indistinguishable from
clean data — the same shape as a test that cannot fail.

### Drafting to target: 169 pairs, and three more findings

Drafted from 20 documents in 25 requests across two models. **169 of 198 accepted
(85.4%)**, shapes spread 20-35 each. Within the 150-200 target.

**Output truncation cost a whole batch, silently.** `max_output_tokens=8192` was enough
for ten terse pairs and not for ten verbose regulatory ones: the response truncated
mid-JSON and the schema rejected it, losing all ten. Raised to 32,768 — these models
allow 65,536, and output is billed on what is produced rather than what is permitted, so
a generous cap costs nothing. The document went 0/10 to 10/10.

**Line-numbered documents cannot be drafted from, and two prompt attempts did not fix
it.** `fda-172169` and `fda-189693` yielded 0/10 and 1/10 against 7/10-10/10 elsewhere.
Every reject was a near-miss with a dropped line number, similarities up to 0.98.

The tempting explanation — that the instruction is impossible because every line carries
a number — is **false**, and measuring it saved a third prompt attempt: the median
longest digit-free run inside a chunk of the worst offender is **101 characters**, four
times the 24-character quote floor. The model can comply and does not.

So those documents are skipped as *drafting sources* while remaining fully indexed and
retrievable. The threshold (`MAX_LINE_NUMBER_DENSITY = 0.35`) is **bounded by
observation, not measured from a gap** — the density distribution is continuous, median
0.05 and p90 0.51 with no natural break. What is known is that 0.04-0.18 drafts well and
0.57-0.65 does not; 0.18-0.57 is untested, and 0.35 sits inside that untested band. It
excludes 11 of 152 documents. The real fix is upstream — strip line numbers during
extraction — deferred because it re-invalidates the ablation table.

**A by-model comparison that looked real and was not.** Acceptance was 83.0% on
`gemini-3.6-flash` and 72.7% on `gemini-3-flash-preview`, which reads as a model
difference. It is not: the two line-numbered documents contributed 18 of the second
batch's 30 rejects, and excluding them it scores **88.9%**. Document composition, not
generator quality. Recorded because attributing it to the model was the easy and wrong
conclusion, and it is the third time this project has caught itself doing exactly that.

### Factual triage: the check that answers "is the reference answer right"

Automated verification proves a quote exists; it cannot prove the answer follows from
it. One class of that gap *is* checkable without a model, and it is the class that
matters most in regulatory text: **numbers and identifiers asserted in an answer but
absent from the cited source.** A wrong quantity is materially wrong, and answers here
are dense with them.

Support is checked against the whole cited chunk rather than the evidence quote, because
a drafter legitimately summarises more than it quotes. Written-out numerals resolve, and
so do "a year" / "each year" / "annually", both of which arrived as false positives from
real data before being handled.

Over 171 pairs it raised **2 flags — one real, one false positive**, and the real one is
instructive. An answer stated that Structured Product Labeling "is an HL7 standard";
"HL7" appears nowhere in the cited chunk, though it does appear elsewhere in the
document. So the reference answer drew on material outside its own evidence. Scoring a
model against it would demand knowledge the retrieved evidence never contains. Two such
pairs were dropped.

That single defect is a **preview of exactly what tier 2 is for**: the quote was real,
locatable, and correctly attributed, and the answer still did not follow from it. Tier 1
cannot see that. This check catches the numeric subset for free; the rest needs a judge.

### What is deferred

- **A real curation pass** on factual correctness beyond the numeric subset.
- **The remaining ~100 pairs**, to reach the 150–200 target. Quota-paced.
- **Unanswerable pairs**, which are constructed rather than drafted: asking a model to
  invent a question its own corpus cannot answer yields questions the corpus answers
  obliquely, which are the worst possible refusal tests. Built the way `evalset.py`
  builds its own — synthesise a citation and verify its absence.
- **Tier 2 of the verifier** and the generation metrics, which are the rest of Phase 6.

## Corpus regeneration — post-quarantine golden set and ablation

**Status: done.** Artifacts regenerated: `extraction_report.md`, `chunking_report.md`,
`corpus/evalset.jsonl`, `retrieval_eval.{md,json}`. **542 tests, lint clean.**

Nominally "re-run the pipeline so the evalset matches the corpus". It turned up a real
defect instead, and regenerating alone would have made things worse.

### The evalset generator never filtered on quarantine

`cmd_evalset` built from every extracted document. Its docstring justified this —
ground truth is character spans, so it must not depend on how the corpus was chunked —
and that reasoning is correct but does not extend to quarantine. A span in a document
that *no* strategy indexes resolves to no chunk under every retriever, so the query is
unscorable by construction and silently shrinks the slice it belongs to.

Before this run, 2 of 180 answerable queries were unscorable. Regenerating took it to
**7** (6 `section_lookup`, 1 `title_lookup`) — worse, and necessarily so: quarantining
three documents instead of one means more queries aimed at documents that no longer
exist in the index. The task as written ("regenerate the evalset") would have shipped a
more broken golden set than it started with.

The pool now passes through the same `_is_quarantined` predicate the chunker uses, so
the golden set and the index cannot disagree about which corpus exists. **All 180
answerable queries now resolve under all three chunking strategies.**

There was also **no test module for `evalset` at all**, which is a large part of why a
five-phase-old gap went unnoticed. `tests/test_evalset.py` now exists, including a
guard that confirms an unfiltered pool really does leak — without it the main assertion
would pass for the wrong reason.

### Corpus shape

| | before | after |
|---|---|---|
| Documents extracted | 155 | 155 |
| Quarantined | 1 (`character_spaced`) | **3** (+2 `broken_encoding`) |
| Indexable | 154 | **152** |
| `structural` chunks | 13,706 | **13,423** |
| `fixed` chunks | 11,775 | 11,527 |
| `semantic` chunks | 11,722 | 11,468 |
| Answerable queries unscorable | 2 | **0** |

The two `broken_encoding` documents are the Caesar-shifted glyph-code text layers found
in Phase 5. Their detectors were wired into `extract.py` and the quarantine predicate at
the end of Phase 5 but had never run; this is the run where they took effect. The
extractor's control-char shares (0.13096 and 0.03574) reproduce the corrected
whole-document figures from the Phase 5 audit gate exactly — independent confirmation
that the basis correction was right.

The quarantine reason line was also reporting a bare "text quality flagged at
extraction" for anything that was not character-spaced, which after Phase 5 was most of
them. It now names the class and its ratio: *"text layer is raw glyph codes, no usable
ToUnicode map (13.1% control characters)"*.

### What moved in the ablation

**116 of 192 runs changed.** The headline conclusion did not:

| Measure | before | after |
|---|---|---|
| `exact_identifier` recall@10, BM25 vs dense (`fixed`) | 0.955 / 0.013 | **unchanged** |
| `exact_identifier` recall@10, BM25 vs dense (`structural`) | 0.951 / 0.022 | **unchanged** |
| Sparse-over-dense ratio on identifiers | 74.8x / 42.8x | **unchanged** |
| Refusal separability, BM25 | 0.992 / 0.983 / 1.000 | **unchanged** |
| `section_lookup` n | 58 | **60** |
| `section_lookup` recall@10, BM25 (`structural`) | 0.876 | 0.828 |
| `section_lookup` recall@10, dense (`structural`) | 0.864 | 0.739 |
| `title_lookup` hit@10, BM25 | 0.983 | 1.000 |

`exact_identifier` is byte-identical because its queries are drawn from CFR citations
and none of the three quarantined documents carried any — so that slice's sample really
is the same set. Worth stating, because "identical to four decimals" first looked like a
run that had not happened.

**The `section_lookup` drop is not attributable to the quarantine.** Both the corpus and
the queries changed, and two previously-excluded queries are now in the denominator, so
this is a different and slightly harder sample rather than a measured regression.
Separating the two would need the old golden set scored against the new corpus, which is
not worth a run.

**`title_lookup` is now saturated**: all 16 retriever configurations score exactly
1.000. A slice nothing can lose on measures nothing. Flagged in the README and left for
Phase 8 to either harden (partial or shortened titles) or drop, rather than kept as a
row of identical numbers.

### Claims withdrawn rather than updated

Two figures were not merely restated:

- **`retrieval.py`'s heading-mode table** moves with the golden set as well as the chunk
  target: 0.892/0.774/0.511 at 2,048, 0.876/0.631/0.408 at 1,024 pre-quarantine, and
  0.828/0.587/0.495 now. The docstring records all three and states that cross-target
  comparison is confounded.
- **README's claim that each decision effect is larger at 1,024 than at 2,048** is
  withdrawn. Two of the three effects shrank (+0.467 → +0.333 and +0.346 → +0.255) and
  the heading effect is now *below* its 2,048 figure — but the 2,048 numbers were
  measured on the previous golden set, so neither direction is supportable without a
  re-run. Identifier tokenization held at exactly +0.402. Withdrawing beat picking
  whichever reading still flattered the conclusion, which is the failure this project
  has now logged three times.

## Phase 5 — Grounded generation, quote-and-verify citations, refusal

**Status: shipped — both verification gates passed.** Artifact:
`reports/generation_eval.{md,json}`. 23 findings across the two gates, every one
reproduced before being acted on, every one fixed; zero false positives. Tier 2 of the
verifier (LLM-as-judge) is deferred and named as such below — Phase 5 ships the
deterministic half.

### The generator changed, and the citation design changed with it

The locked decision was Claude, chosen for native citations that return character
offsets into the supplied document — that made tier-1 verification a span check.
First-party access is blocked by organisation policy, and the available Bedrock role
permits only models AWS has retired (21 Claude model IDs probed: every permitted
model 404s, every existing model 403s, no intersection). The project cannot depend
on it.

Gemini on the free tier is the substitute, and it has no citation feature. So tier 1
now **locates** the model's quoted span in the source by search rather than trusting
reported offsets — plan deviation #14, with the locked decision revised in place.

This is stronger in one respect and weaker in another, and both are worth stating.
Stronger: nothing the model claims about position is trusted; position is recovered
independently, and the check works for any generator, which turns "citation
verification rate by model" into a table row rather than a vendor property. Weaker: a
model not trained to emit verbatim spans paraphrases, and a paraphrase is correctly
unverifiable — so the failure rate rises, and that rate is the measurement, not a
number to prompt away.

### What shipped

`citations.py` (tier-1 verifier, no model), `generation.py` (provider interface +
Gemini transport), `answer.py` (prompt, response envelope, refusal), `gen_eval.py`
(+ `ragpipe gen-eval`, `make gen-eval`). Two new `pdfcheck` detectors, wired through
`extract.py` and the chunk-time quarantine predicate. **539 tests passing, 2 skipped**,
`make lint` clean.

Verification has six outcomes rather than a boolean, because a boolean conflates
failures with different causes and different fixes:

| Outcome | Meaning |
|---|---|
| `exact` | byte-identical to the cited chunk |
| `normalized` | matches modulo whitespace runs (PDF artifacts) |
| `line_number_ambiguous` | **not verified.** Failed the strict check, but would match if interleaved legislative line numbers were discounted — a hint that extraction, not the model, is the likely cause |
| `wrong_chunk` | real text, wrong provenance — a plumbing defect |
| `unverified` | not in the document — fabricated or paraphrased |
| `too_short` | under 24 chars — not evidence of anything |

Only the first two count as verified. The third was briefly counted as verified and the
code-review gate demonstrated that this blessed meaning changes; see gate 2.

### Measured — `gemini-3.5-flash-lite`, BM25 top-5, 16 queries (4 per slice)

| Measure | Value |
|---|---|
| Scored | 16/16, 0 errors |
| Citations verified | **19 of 21** |
| Citation precision, macro (mean over answers with ≥1 citation) | **0.875** |
| Citation precision, micro (verified / all claimed) | **0.905** |
| Refusal on `unanswerable` | **4/4 (100%)** |
| Refusal on answerable slices | 0/12 |
| Separability AUC, all answerable | **0.958** over 48 pairs |
| Separability AUC, `exact_identifier` only (Phase 3's population) | **1.000** over 16 pairs |
| Buckets | exact 17, normalized 2, `line_number_ambiguous` 1, `unverified` 1, `wrong_chunk` 0, `too_short` 0 |
| Tokens | 28,741 prompt / 4,170 output / **0 thinking** |
| Latency | p50 3.24s, p95 4.10s (nearest-rank over 16 = the slowest query) |

Per-slice precision: `section_lookup` 1.000, `title_lookup` 0.875,
`exact_identifier` 0.750.

Macro and micro differ because answers claim between one and nine citations; both are
reported rather than one being presented as *the* precision. The two AUC rows differ
for a reason that vindicates Phase 3 rather than this phase: pooling all answerable
slices lets long, high-IDF section headings and document titles into the positive
class, and `reports/retrieval_eval.md` records that the same pooling once reversed
which chunking strategy looked better. The restricted row is the one comparable in
population to Phase 3 — though still not in statistic, since that was accuracy.

**n = 16 is small and the report says so.** It is not a sampling choice: the free
tier allows 20 requests per day per model, so 16 queries with retry headroom is what
fits. Quota being per *model* is what makes a by-generator comparison affordable at
all.

### The one unverified citation is a true positive, and a good advertisement

`ident-0002` cited a 159-character span. Every fragment of it exists in the cited
chunk. The divergence is one word:

    source:  '...and 21 CFR 890.380 (Motorized Three-wheeled Vehicle) should follow...'
    model:   '...in  21 CFR 890.380 (Motorized Three-wheeled Vehicle) should follow...'

Substituting `in` back to `and` makes it an **exact** match, which pins the
divergence to that single token. The model almost certainly did it to make the quote
read grammatically standalone. A human reviewer would very likely have missed it; the
verifier rejected it deterministically. That is the entire argument for tier 1.

### `too_short` went to zero because the prompt states the floor

An early ad-hoc run produced three citations whose quote was the string `'The '`.
After the system prompt began stating the 24-character minimum (interpolated from
`citations.MIN_QUOTE_CHARS` rather than written twice), the bucket is empty across
both real runs. `wrong_chunk` is also zero, so no misattribution was observed at all.

### Two independent refusal mechanisms, kept apart

Score-gated refusal (before generation) and model-decided refusal (the `refused`
field) have different costs and different blind spots, and `refusal_source` records
which fired. Blending them would let a strong threshold disguise a badly behaved
model. The score gate is **off by default** because the mechanism is
retriever-specific: Phase 3's best-threshold separability accuracy is 0.983 for BM25
alone on `structural`, against 0.667 for min-max fusion and 0.942 for rank fusion plus
reranking on the same chunking, so the threshold must be recalibrated against whatever
Phase 7 serves. (0.725 is min-max's best *anywhere*, on `semantic` — quoted beside "on
structural" until Phase 6's audit gate caught it.) This
run's 0.958 is pairwise AUC over 4 unanswerable queries; Phase 3's figure is
best-threshold *accuracy*. **They are different statistics and must not be compared.**

### Findings

1. **The retrieval golden set is not an answerability label.** In an earlier run on
   `gemini-3.6-flash` — **not** the shipped run tabulated above, where `ident-0000` is
   answered with one verified citation — this query looked like an over-refusal: gold
   chunk retrieved at rank 1, model refused. The gold
   span is `'21 CFR 1.980(k)'`, sitting inside a footnote listing citations for
   "post administrative detention recordkeeping". The document *cites* the regulation
   and never states its requirements, so refusing was correct. All 60
   `exact_identifier` queries ask what a regulation requires against a corpus that
   mostly just mentions it. `gen_eval` therefore reports the answerable refusal rate
   with and without that slice and declines to call either an over-refusal rate.
   Same root cause as the known `ident-0031` hazard. **This constrains Phase 6: the
   golden answer set cannot be derived from the retrieval evalset.**

2. **Interleaved legislative line numbers break naive verification.** FDA draft
   guidances extract as `'...current thinking of the Food and Drug 8 \nAdministration'`.
   A model quoting the sentence drops the `8`, and no amount of whitespace
   normalization finds it — `8` is a token. In the first clean run one `title_lookup`
   query had **all seven** of its citations scored `unverified` for this reason alone,
   with `title_lookup` precision at 0.750 against `exact_identifier`'s 1.000. Phase 4
   had recorded this artifact as a hypothesis about BM25 false matches; its real cost
   was here. (Phase 4 quoted 34.4% of chunks and 1.4% of tokens, but flagged those as
   coming from a hand pilot with no rule in code, hence unfalsifiable as written. The
   Phase 5 audit gate confirmed they do not reproduce: three reasonable
   reimplementations give 37.5–53.2% of chunks and 0.79–2.28% of tokens. Right order of
   magnitude, so the figures are not repeated here as fact.) Fixed with a third matching tier that
   drops only *line-adjacent* bare integers — stripping every integer would be unsafe
   in a corpus full of `21 CFR 211.192` and `within 15 days` — reported under its own
   bucket so the tolerance can never hide inside `normalized`. In the shipped run it
   fires exactly once, on exactly that pattern.

3. **Two more text-layer defect classes, found by reading a verified citation.** A
   quote came back `exact` and unreadable:
   `'Informedconsentforeachpatientwillbeobtainedpriortoinitiating...'`. Measured
   across 154 documents:

   | Class | Signal | Affected | Worst legit | Mildest corrupt | Median | p95 |
   |---|---|---:|---:|---:|---:|---:|
   | character-spaced (Phase 4) | single-char token share | 0 new | 0.159 | — | 0.033 | 0.067 |
   | broken glyph encoding | control-char share | **2** | 0.00053 | 0.03574 | 0.000 | 0.000 |
   | collapsed spaces | run-on alpha tokens ≥25ch | **1** | 0.00029 | 0.01794 | 0.000 | 0.000 |

   These are per-document figures over each document's **whole extracted text**. An
   earlier version of this table reported 0.00057 / 0.03622 / 0.00026 / 0.01759, which
   came from concatenating each document's chunks and so double-counted the 15%
   inter-chunk overlap — arithmetically fine, but not the population the label claimed,
   and not the basis `extract.py` calls the detectors on. Caught by the audit gate.
   Classification is unchanged.

   Both new classes have a median and p95 of exactly zero, so thresholds sit *near* the
   geometric midpoint between worst-legitimate and mildest-corrupt — 0.005 and 0.002
   are 1-significant-figure roundings of the true midpoints 0.004355 and 0.002300.
   The detectors classify 151 ok / 2 `broken_encoding` / 1 `space_collapsed`, no false
   positives. **That is detector output, not corpus state:** no re-extraction has run,
   so both broken-encoding documents are still in the shipped index — 283 chunks,
   2.06% of 13,706 — and `data/extracted/index.jsonl` still labels them
   `text_quality: 'ok'`. Quarantining them is listed under Deferred. The worst case is
   a Caesar-shifted text layer — `IIHFWLYHQHVV` is `EFFECTIVENESS` shifted by 3 — from
   a PDF with no usable `ToUnicode` map. Both broken-encoding documents score 0.066
   and 0.069 on single-char share, far below the 0.40 gate, which is exactly why
   Phase 4 missed them.

   **`space_collapsed` is deliberately not quarantined.** Its citations verify at
   exact offsets, so the text is degraded but readable; discarding it would throw
   away recoverable evidence. It degrades BM25 tokenization and is reported as a
   warning instead.

4. **The free tier's real limit is 20 requests per day per model, not per minute.**
   The 429 body says `"Please retry in 4.411926172s."`, which reads exactly like a
   rate limit; the authoritative field is
   `quotaId=GenerateRequestsPerDayPerProjectPerModel-FreeTier`. Because a rejected
   request still spends a quota unit, retrying into an exhausted daily quota is pure
   loss — so a per-day 429 is now not retried at all, and the error names the two
   real options (different model, or wait for reset).

### Mistakes made in this phase, and what they cost

The standing hazard has been that every real defect is a *claim*, never the
arithmetic. This phase produced four of my own, and they are logged because a history
with no mistakes in it is less credible than one with them.

1. **Misdiagnosed the rate limit as per-minute.** I read `limit: 20` plus a
   4-second retry hint and wrote a comment, a constant (`FREE_TIER_RPM`), and a test
   asserting a per-minute cap. All three were wrong; the structured `quotaId` said
   `PerDay`. Cost: a day of quota on `gemini-3.6-flash`, and a wrong rationale
   committed to three places before the probe that settled it.
2. **Reported a process as stopped when it was not.** `pkill` failed under the
   sandbox with "Cannot get process list", and I read the fallback `echo` as
   confirmation. The first eval run continued to completion, competing with its
   replacement for the same quota and writing a report of 3 scored queries out of 80.
3. **Stated a test count I had not read** ("513"; it was 495 at the time).
4. **Called `ident-0000` an over-refusal** before reading the gold span, which turned
   out to make the refusal correct. Finding 1 above is the corrected version.
5. **Built a citation-matching tier that verified meaning changes** — the largest of
   the five. Having written in the same file that "a citation verifier that tolerates
   changes in meaning is theatre", I added a tier that scored a model's "40 CFR" as a
   verified citation of a document saying "21 CFR". The reasoning that led there was
   sound in shape (an artifact was causing false negatives, so widen the match) and
   wrong in execution (the widening was not bounded by anything that distinguishes a
   line number from a regulation). It survived my own review because I tested it on the
   case it was built for and not on the case it would break. Caught by gate 2, and the
   two tests I wrote to "guard" it could not fail.

Two guards came directly out of these. The report now refuses to present itself as
evidence below an 80% scored fraction, naming the actual counts — the first run's
artifact was arithmetically perfect and headlined `mean citation precision 1.0`
computed from a single answer. And `QueryOutcome` now stores per-citation method,
quote, and matched text, because a verifier that says "unverified" without showing
the text cannot be debugged without spending quota to re-run.

### Verification gate 1 of 2 — numerical audit

**138 figures recomputed independently from raw data, no `ragpipe` imports. 126 match.
Zero arithmetic errors: all 107 figures in `reports/generation_eval.{md,json}`
reproduce exactly.** Consistent with the twelve prior passes, every defect was a label
or a claim. Eight findings, all fixed; **zero false positives.**

The audit re-verified all 21 citations from `data/chunks/structural.jsonl` with its own
span-location implementation and **agreed with the shipped verdict on 21 of 21**,
including the line-number tier firing exactly once. (It was called `line_numbered` and
counted as verified at the time the audit ran; gate 2 below renamed it to
`line_number_ambiguous` and demoted it. The audit's agreement was on the *matching*,
which still holds — only the verdict attached to it changed.) That check was only possible
because per-citation detail is persisted; it is the strongest evidence the verifier
works, and it is why the storage cap has since been raised (see below).

| # | Finding | Severity | Reproduced |
|---|---|---|---|
| 1 | The Phase 3 separability cross-reference was wrong three ways | HIGH | yes |
| 2 | The whole defect table was measured on chunk concatenations, not documents | MED-HIGH | yes |
| 3 | The report asserted two refusal rows "differ" when both were 0.0% | MED | yes |
| 4 | Finding 1 described a refusal from a different run than the shipped table | MED | yes |
| 5 | "2 quarantine" described detector output, not corpus state | MED | yes |
| 6 | Phase 4's line-number figures re-cited as fact, not reproducible | LOW-MED | no — see below |
| 7 | `ident-0002` quote length 155 vs actual 159 | LOW | yes |
| 8 | "sit at the geometric midpoint" overstated a rounding | LOW | yes |

**Finding 1 is the one that mattered, and it was load-bearing.** The claim "Phase 3
measured this separability at 1.000 for BM25 alone and at best 0.725 for any fusion
configuration" was the stated justification for shipping the score gate off by default.
All three parts were wrong:

- **Metric.** Phase 3 stores no AUC. Its separability is *best-threshold accuracy* over
  60 answerable / 60 unanswerable queries. Phase 5's 0.958 is pairwise AUC. Presenting
  them as the same statistic ("*this* separability") was a category error.
- **Population.** 1.000 is `semantic`. On `structural` — the chunking Phase 5 actually
  retrieves over, and every cited `chunk_id` is `::structural::` — BM25 alone is
  **0.983**.
- **Scope.** 0.725 is the max over **min-max fusion only**. Over *any* fusion
  configuration it is **0.9417** (`structural`, RRF + rerank). Understated by 0.217.

Phase 3's own entry says it correctly — "**min-max fusion** recovers much less". Phase 5
dropped the qualifier, and the generalisation happened to strengthen the argument being
made: the wider the sparse-versus-fusion gap looks, the more retriever-specific the
mechanism appears. That is the same failure as Phase 4's "1,024 dominates" — reading a
comparison in the direction that flatters the conclusion already reached. The
conclusion survives (the gate is still retriever-specific and still needs
recalibration), but the margin is 0.983 vs 0.942, not 1.000 vs 0.725. Corrected in six
places: `answer.py`, `gen_eval.py` (docstring and report prose), `cli.py`, the
`Makefile`, and here.

**Finding 2 is the cleanest instance yet of the project's recurring defect class.** The
figures were arithmetically perfect and described a population the label denied. Five
independent values reproduced to five decimal places under exactly one alternative text
construction — concatenating each document's chunks, which double-counts the 15%
inter-chunk overlap — while the comment claimed "per document, whole text". It matters
beyond labelling because `extract.py` calls these detectors on whole document text, so
the thresholds were calibrated on a basis they will never see. Re-measured properly;
classification is unchanged (same 151/2/1, same documents), but the worst-legitimate
anchors moved 7.5% and 10%, and the stated ratio margins with them.

**Finding 6 was acted on without independent reproduction**, and that is worth stating
plainly rather than counting it as verified. The audit could not reproduce Phase 4's
"34.4% of chunks / 1.4% of tokens" under three reasonable reimplementations
(37.5–53.2% and 0.79–2.28%). I did not recompute it a fourth time, because Phase 4's own
entry already records those figures as coming from a hand pilot with no rule in code —
"unfalsifiable as written". Under either outcome the fix is the same: stop restating
them as unqualified fact. The defect was **provenance decay** — the caveat did not
travel with the number into the next phase.

Two things the audit flagged that are conventions rather than errors, both now
documented where they are used: the reported `p95` latency over 16 observations is
nearest-rank, so it is definitionally the maximum; and 6 of 21 stored quotes hit the
160-character cap, meaning the audit could only prefix-check them. The cap is now
`MAX_STORED_QUOTE_CHARS = 2000`, comfortably above a chunk, so future runs are fully
re-verifiable rather than prefix-verifiable.

### Verification gate 2 of 2 — code review

**15 findings. All confirmed and reproduced; zero false positives.** The reviewer found
**no arithmetic error anywhere in Phase 5** — every offset computation and every figure
was correct, which is now the thirteenth consecutive pass with that result. What was new
this phase is the kind of claim that failed: **two of the defects were claims about
safeguards, where the code did not implement the safeguard it documented.**

**Finding 1 is why the gate exists.** The `line_numbered` tier added earlier in this
same phase was blessing quotes that change what the document says. Reproduced against
live corpus text (`fda-106721::structural::00009`):

    document: 'Fish and Fishery Products or 21 \nCFR part 120, Hazard Analysis'
    model:    'Fish and Fishery Products or 40 \nCFR part 120, Hazard Analysis'
    -> scored line_numbered, verified=True, citation_precision=1.0

Both the source's `21` and the quote's `40` sit at a line boundary, so both are deleted
and the two strings reduce to the same text. A model asserting **40 CFR** where the
regulation is **21 CFR** passed verification. So did a quote that *omits* a stated
duration: source "must respond within 30 \ndays", quote "must respond within days".

The premise that failed was mine: "a line number by construction sits alone at a line
boundary." pypdf splits real identifiers across line breaks — Phase 1 recorded 63
occurrences of exactly this (`21 CFR 812.2(c)` extracting as `21 CFR 71 \n812.2(c)`) —
so content numbers sit there too. Measured exposure: **7,229 of 13,706 chunks (52.7%)
lose digits** under that normalization, 86,485 digit characters corpus-wide, and 15 of
the 21 quotes in the shipped run contain a literal newline. The precondition was
ordinary behaviour, not an edge case.

It also falsified the module's own stated principle three paragraphs above it — "a
citation verifier that tolerates changes in meaning is theatre". It was right, and I
broke it inside the same file.

**Resolution: the tier is demoted to a non-verified diagnostic** (`line_number_ambiguous`).
A hit records that the failure is *consistent with* an extraction artifact rather than
fabrication, which preserves the triage value that motivated it, without letting it
count as provenance. It cannot be repaired by local pattern matching, because "the model
dropped a line number" and "the model dropped a content number" are structurally
identical; separating them needs document-level evidence of a line-numbering scheme,
which is deferred with the measurement above as its justification. The shipped figures
moved accordingly: **20/21 verified → 19/21, macro precision 0.917 → 0.875**.

**Finding 4: the Phase 5 detectors were dead code.** `QUARANTINE_QUALITIES` declared
`broken_encoding` unindexable, `text_quality_label` assigned it — and nothing called
either. `extract.py` still emitted only `ok | character_spaced`, and
`cli._is_quarantined` compared against the single string `"character_spaced"`. So the
two Caesar-shifted documents stayed in the shipped index (283 chunks, 2.06%) while a
constant said they could not be. The existing test asserted frozenset *membership*, so
it passed while the opposite was true of the pipeline. Now wired: `extract.py` calls
`text_quality_label` and persists both new ratios, `_is_quarantined` tests membership in
`QUARANTINE_QUALITIES` and re-applies every live threshold, and the new tests assert
behaviour rather than declaration. Takes effect at the next re-extract.

The remaining thirteen, all fixed:

| # | Finding |
|---|---|
| 2 | The tier's stated "three bounds" included one that is not a bound — a quote with an altered number can never match the stricter tiers, so "both failed" is precisely the dangerous state |
| 3 | Two tests asserting exactly the safeguard that did not exist, both unable to fail: single-line fixtures meant no line-number pattern could fire |
| 5 | The Phase 3 separability claim (also found independently by the audit gate) |
| 6 | `separability_auc` pooled all answerable slices while claiming to "recompute" Phase 3's figure, which deliberately pooled `exact_identifier` only — the exact pooling `retrieval_eval.md` records as having reversed a conclusion. Both pools now reported |
| 7 | Two unconditional report sentences contradicting the tables under them |
| 8 | `make gen-eval` still said "20 requests/minute" and passed `--per-slice 20` — 4× the daily budget, i.e. it reproduced the original failure |
| 9 | The per-day substring fallback fired even when a *per-minute* violation was present, turning a retryable rate limit into an immediate hard failure |
| 11 | `too_short` documented as counting as neither verified nor fabricated; it counts in the precision denominator exactly like `unverified` |
| 12 | `Retry-After: -3` and `nan` reached `time.sleep` and raised `ValueError` out of the retry loop, where the blanket handler recorded a retryable 503 as a permanently lost query |
| 13 | The documented retry floor was applied only to the body-derived delay, not the header path — and a test asserting exactly `7.0` had pinned the divergence |
| 14 | The report's bucket table iterated a hardcoded 6-tuple while its total summed every key, so a new bucket would be omitted from the table while inflating its denominator |
| — | `p95` over 16 observations is nearest-rank, i.e. the maximum; macro and micro precision were printed side by side unlabelled; the `index_map` contract is false for synthesized separators; `propertyOrdering` does not stop the model front-loading prose (citations are emitted *after* `answer`); two `pdfcheck` fixtures sat at 180 tokens under a 200-token guard and so passed via fail-open |

**What the reviewer attacked and could not break**, which is worth recording because it
is the part that held: 30,000 fuzzed inputs against `normalize_whitespace` found zero
offset-contract violations for real characters; 34,000 successful `locate_quote` matches
all satisfied `matched_text == source[start:end]` with no off-by-one at either boundary;
`_pace` is bounded across retries; every `gen_eval` denominator is correct and nine
degenerate cases return `None`/`[]` rather than dividing by zero; and the specific
"harmless anchoring" claim I had asked it to break held over 31,106 fuzzed matches.

**Regression coverage was checked by mutation, not by assertion count.** Re-adding
`LINE_NUMBER_AMBIGUOUS` to `VERIFIED_METHODS` now fails 5 tests; before this gate it
failed none. **539 tests passing, 2 skipped, `make lint` clean.**

### Deferred

- **Tier 2 (LLM-as-judge: does the span support the claim?)** is not built. Tier 1
  answers "is this quote real"; it does not answer "does this quote support the
  sentence it is attached to". `ident-0000` is answered by
  `gemini-3.5-flash-lite` and refused by `gemini-3.6-flash` on identical context,
  which is precisely a tier-2 question.
- **A document-level line-numbering detector.** The only principled way to recover the
  `line_number_ambiguous` cases as genuinely verified: detect that a document numbers
  its lines (a monotonic sequence with unit increments) and only then discount those
  integers. Justified by the measurement in gate 2 rather than by plausibility.
- **A larger sample.** 16 queries is the daily free-tier ceiling per model.
- **The by-generator comparison table**, which the provider interface exists to make
  cheap and which per-model quota makes affordable.
- **Quarantining the 2 broken-encoding documents**, which needs a re-extract and
  therefore an ablation regeneration (~50 min). Recommended to batch with the
  evalset regeneration, which already requires that same run.

## Phase 4 — Chunk-size sweep, contextual retrieval, and a corpus defect

**Date:** 2026-08-17
**Status:** **shipped** — both verification gates passed; 21 findings across the two
reproduced and fixed. Contextual retrieval is **built and tested but not run**, blocked
on API access (see below); recorded as incomplete rather than dropped.
**Artifacts:** `reports/chunk_size_sweep.md`, regenerated `reports/retrieval_eval.md`

### Headline: the chunk size this project shipped was measurably wrong

Phase 1b picked a 2,048-character target (≈512 tokens) from the Phase 0 estimates and
never revisited it. The sweep — 512 / 1,024 / 2,048 / 4,096 chars at a fixed 15%
overlap on `structural`, so size is the only variable — says it was too big.

**The trap first, because it is the interesting part.** `recall@10` on section lookup
rises cleanly with chunk size (BM25: 0.809 → 0.876 → 0.897 → 0.932). Read alone that
says "bigger is better". It is an artifact: larger chunks mean fewer chunks per
section, so recall's *denominator* shrinks. The denominator-free metric does not move
at all — `hit@10` is flat at **0.966** from 1,024 onward for every retriever — and
`precision@10` falls from 0.236 to 0.109. Larger chunks find exactly the same
sections; they just need fewer of them to cover one.

| target | ident `hit@1` | ident `hit@10` | section `hit@10` | separability | truncated |
|---|---|---|---|---|---|
| 512 | **0.900** | 0.967 | 0.948 | **0.992** | **0.0%** |
| **1,024** | 0.850 | **0.983** | **0.966** | 0.983 | 0.6% |
| 2,048 *(old default)* | 0.833 | **0.983** | **0.966** | 0.942 | 10.4% |
| 4,096 | 0.700 | 0.967 | **0.966** | 0.908 | 31.1% |

**This is a trade, not a domination, and the first version of this entry got it wrong.**
The five figures above genuinely favour 1,024 and every one was verified exactly by the
audit. But across *all* denominator-free metrics (`hit@1/5/10/20`, `mrr@10`) over three
answerable slices and four retrievers, **2,048 wins 22 comparisons to 1,024's 11**, with
27 ties. `mrr@10` — which `metrics.py` calls "what a user actually sees" — prefers 2,048
on **8 of 12** slice x retriever pairs, and section `hit@1` is 0.672 at 1,024 against
0.724 at 2,048. Generalising from five favourable metrics to "every undistorted measure"
was cherry-picking: the same selective reading the entry accuses `recall@10` of, one
level up. Caught by the numerical audit against the sweep's own JSON.

1,024 still ships, on two narrow grounds:

- **Truncation, 10.4% -> 0.6%.** At 2,048 a tenth of chunks lose text before a vector
  exists — a correctness defect, not a ranking preference.
- **Refusal separability, +0.042**, which Phase 5's refusal path depends on.

The cost is top-of-ranking precision: smaller chunks fragment a section across more
pieces, so the single best chunk is less complete. If Phase 5 needs rank-1 precision
more than the refusal threshold, 2,048 is the better pick and the data is already here.
512 remains a documented alternative (best `hit@1` and separability, zero truncation) at
a cost of 0.017 section `hit@10`.

**Two mechanisms worth naming.** Truncation is a *function of chunk size*, not just of
the model: the embedding window is a fixed 512-token budget, so a 4,096-char target
means a third of chunks are clipped before a vector exists. And refusal separability
degrades monotonically with size (0.992 → 0.908) — longer chunks dilute a match, so
BM25's top score separates answerable from unanswerable less sharply. That is a direct
constraint on Phase 5's refusal threshold, arriving from an unexpected direction.

### A corpus defect that had been in the index since Phase 1

While inspecting a real generated prompt before spending anything, an excerpt read
`Offi ci al Titl e: A n O p e n -L a b el`. One document extracts to character-spaced
noise:

| | |
|---|---|
| Documents affected | **1 of 155** (`ctgov-NCT03386721-Prot_SAP_000`, share 0.723) |
| Share of corpus characters | **6.7%** (801,075 of 11,986,402) |
| Corruption density | 99.7% of substantial lines above the 0.40 threshold; 1 of 8,448 lines is clean |
| Repairable | No — intra-word and inter-word gaps are both a single space, so word boundaries need dictionary re-segmentation |

**Root cause: `pdfcheck` verified text-layer *presence*, never *quality*.** A
glyph-positioned PDF has a perfectly good text layer that happens not to be words, so
it passed every emptiness check for four phases. It is also the largest document in the
corpus, so it carried 5.0-6.6% of chunks depending on strategy.

Fixed with `single_char_token_share` at a threshold of **0.40** — 2.5x above the worst
legitimate document (0.159, an FDA index page of short entries and numbers) and well
below the corrupt one (0.723), so the threshold is not finely balanced. Quarantine
follows the existing policy for scanned PDFs: text that cannot be read cannot be
retrieved, and indexing it silently puts a floor under every metric that is
indistinguishable from the retriever underperforming.

Caught only because a real prompt was inspected before submitting a batch. Trusting the
estimator would have paid to write blurbs for ~500 chunks of noise *and* confounded the
contextual-retrieval measurement with an invisible floor.

### Contextual retrieval: built, tested, not run

`src/ragpipe/contextual.py`, 57 tests, no spend. The engineering worth keeping is the
cost architecture.

The standard recipe — Anthropic's own cookbook included — is **one call per chunk**
with the document as a cached prefix. Measured against this corpus:

    architecture      requests   all cache hits   all cache misses
    per-chunk           13,706          $39.58            $421.92
    grouped 40/call        425           $6.81             $15.62   <- this module
    one call per doc       154           $6.04              $6.04

Two independent reasons the grouped form wins, and only the first is about money:

1. **Batch parallelism undermines prompt caching.** A cache entry becomes readable only
   once the first response begins; a batch fires everything at once, so N concurrent
   requests sharing a prefix can all miss. The per-chunk architecture's 8x saving is a
   hope, not a guarantee — its real cost sits in a **$40–$422** band you do not
   control, and the spread *is* the risk. Grouping narrows the same band to
   $6.81–$15.62 and makes caching a bonus instead of load-bearing: 425 requests, not
   13,706.
2. **Bounded output.** One call per document would put 542 blurbs in a single response
   for the largest protocol (765 blurbs at the shipped 1,024 default, ~61K output tokens), where quality drifts.
   40 per call caps it near 3,200.

Grouping costs almost nothing against the all-in-one extreme because the cached
document re-send is charged at 0.1x. Pick the group size for *quality*; let caching
absorb the structural cost.

**A prompt pilot was run by hand** (no API) on three real chunks of an FDA device
guidance. The design works: blurbs added the searchable terms the excerpts lacked
entirely — "Predetermined Change Control Plans", "PCCP", "510(k)", "marketing
submission" — and the boilerplate clause fired correctly, labelling a table-of-contents
fragment in one flat sentence rather than inventing significance. The pilot also
surfaced a second artifact: legislative line numbers interleaved inside sentences
(`"if— \n215 (A) the device remains safe"`), present in **34.4% of chunks but only 1.4%
of tokens**. Recorded with a hypothesis rather than acted on — bare integers like `314`
could cause BM25 false matches against CFR part numbers, but `exact_identifier` sits at
0.983 `hit@10`, leaving little room. First suspect if Phase 8's failure analysis shows
identifier false positives.

### Blocked: API access

Contextual retrieval needs a metered model and no credential is available.

- First-party OAuth (`ant auth login`) is restricted by the parent organisation.
- A Bedrock credential was supplied and diagnosed precisely: **21 Claude model IDs and
  16 non-Anthropic ones probed.** Every model the IAM policy permits returns **404
  (retired by AWS)**; every model that still exists returns **403 (denied by policy)**.
  There is no intersection — a stale policy, not a misconfiguration. The 404s prove
  authorisation *passes* from that network, which also rules out VPN or region as the
  variable.

The module supports first-party (batched) and Bedrock (synchronous) paths, and both are
tested against fake clients, so this is one run plus a table regeneration whenever a
working credential exists — not new work. Recorded here rather than quietly dropped.

**If it runs on a non-Claude writer** (a free tier was discussed), two things must be
stated: the blurb writer is no longer Claude — a change to a locked decision, though
only for that role, since Phase 5's answer generator must stay Claude for
character-level citations — and a weaker writer makes any measured gain a *lower bound*
on what contextual retrieval can do.

### Corpus after quarantine and the size change

| Strategy | Phase 3 (2,048) | Phase 4 (1,024, quarantined) |
|---|---|---|
| `fixed` | 6,628 | 11,775 |
| `semantic` | 7,404 | 11,722 |
| `structural` | 9,413 | 13,706 |

Halving the target roughly doubles the chunk count, as expected.

**A prediction of mine that the data refuted, recorded because it was wrong.** `fixed`
and `semantic` land within 53 chunks of each other (11,775 vs 11,722), and I inferred
from that they had converged behaviourally — that at 1,024 chars most semantic
topic-shift cuts fall where fixed-size chunking would have cut anyway, making the
comparison uninformative. The regenerated table says no:

| BM25, slice / metric | `fixed` | `semantic` | gap |
|---|---|---|---|
| `exact_identifier` hit@10 | 1.000 | 0.983 | 0.017 |
| `section_lookup` hit@10 | 0.810 | 0.879 | **0.069** |
| `section_lookup` recall@10 | 0.530 | 0.574 | 0.044 |

`semantic` is consistently better on section lookup. **Equal chunk counts do not imply
equal chunk boundaries** — the two strategies cut the same corpus into the same number
of pieces in materially different places, and the retrieval difference survives. The
inference from count to behaviour was unfounded, and the only reason it did not reach
the README is that the table was regenerated before the claim was published.

### Ablation table regenerated at 1,024 chars

Every Phase 3 number moved, and the headline claims got *stronger*:

| Claim | at 2,048 | at 1,024 |
|---|---|---|
| Dense-vs-sparse gap on identifiers | 31x | **43-75x** (0.955 vs 0.013 on `fixed`) |
| Identifier-aware tokenization | +0.392 | **+0.402** |
| Prepending section headings | +0.381 | **+0.467** |
| Section-aligned boundaries | +0.187 | **+0.346** |
| Truncation (`fixed`) | 20.2% of chunks | **0.77%** |

Each design decision matters *more* at the smaller chunk size, which is the mechanism
working as expected: less surrounding text to dilute whatever signal the decision adds.

The truncation row is stated post-quarantine on both sides. An earlier version compared
25.5% against 0.77%, but 25.5% included the corrupt document — so ~5.3 points of that
"improvement" was the quarantine rather than the size change. Post-quarantine `fixed` at
2,048 is 20.2%, and the size change alone accounts for 20.2% -> 0.77%.

Two things worth carrying into Phase 5. Min-max fusion recovers much less refusal
separability than the 2,048 figures suggested — best anywhere is 0.725 (`semantic`,
dense weight 0.7) against BM25's best anywhere of **1.000** (`semantic`), where the
earlier table showed 0.842 — so **no fusion configuration tested comes close to
sparse-alone separability**. (An earlier version compared min-max's best-anywhere
against BM25 on `fixed` alone, 0.992: two different populations, which is the exact
class the Phase 3 audit already caught once.) And reranking on `structural` reaches 0.948
recall@10 against a 0.963 ceiling, still essentially at its ceiling, so further gain
there needs a better first stage rather than a better reranker.

### Verification: numerical audit gate

The stricter gate the plan assigns to Phases 2, 3, and 5 — run here because Phase 4
changed the corpus. The auditor reimplemented `structural`/`fixed` chunking,
`normalize_for_embedding`, `cache_key`, identifier atoms, BM25, all three fusion
methods, span-to-chunk relevance, and every metric, **with zero `ragpipe` imports**.

Its validation anchor is worth stating: **its independent chunker reproduces
`data/chunks/structural.jsonl` byte-exactly** — 13,706 chunks, identical id set, zero
field differences. Every derived figure rests on that.

**2,050 of 2,055 figures reproduced exactly.** The five misses are all one cause
(below). Confirmed clean: the entire sweep corpus shape (16 cells), the truncation
percentages against a real 512-token BGE window, every decision delta
(+0.4022 / +0.4673 / +0.3458), the 74.76x and 42.79x dense-vs-sparse gaps, the 12-cell
min-max separability grid, the rerank ceiling verified with real cross-encoder forward
passes, all 18 README headline cells, and every cell of `chunking_report.md`. Dense rows
reproduced from the cached vectors with **zero cache-key misses**, which independently
validates the normalizer and key construction.

**And it confirmed the central claim's *mechanism*, not just its numbers.** Mean relevant
chunks per `section_lookup` query falls 4.638 -> 2.759 -> 1.707 -> 1.276 as the target
grows: recall's numerator is saturated (`hit@10` flat) while its denominator shrinks
3.6x. The rise is arithmetic.

Eleven findings, all reproduced, all fixed.

**HIGH — my headline conclusion was cherry-picked.** "1,024 dominates 2,048 on every
undistorted measure … nothing is traded away" is refuted by the sweep's own JSON. Across
all denominator-free metrics over 3 answerable slices x 4 retrievers: **2,048 wins 22
comparisons, 1,024 wins 11, 27 tie.** `mrr@10` prefers 2,048 on 8 of 12 pairs, and
section `hit@1` is 0.672 at 1,024 against 0.724. The five figures I tabulated do favour
1,024 — all verified exact — but generalising from them was the same selective reading
the entry accuses `recall@10` of, one level up. Also unqualified: "both answerable
slices" when there are three, and `hybrid rrf`/`exact_identifier` is *not* identical
(0.283 vs 0.333). Now stated as a trade with two named grounds (truncation as a
correctness defect; separability for Phase 5) and an explicit revisit condition.

**HIGH — a hardcoded prose range contradicted the JSON it points readers to.** The
rerank-ceiling note said "0.606-0.740"; the true span is **0.491-0.592**. Now computed
from the payload.

**MEDIUM x4 — figures correct at 2,048, reprinted as current at 1,024.** The
per-section chunk means (1.70/2.38, actually 2.76/3.57); three truncation figures
(max 2,464 chars, the 1.16 chars/token floor, the 2,108->1,819 example — all impossible
at the new target); the min-max comparator (0.725 "best anywhere" against BM25's 0.992
on `fixed` alone, when BM25's best anywhere is **1.000** on `semantic` — population
mixing, the class the Phase 3 audit already caught once); and the truncation headline
(25.5% -> 0.77% spans two different corpora, since 25.5% included the corrupt document;
post-quarantine at 2,048 is **20.2%**, so ~5.3 points of the "improvement" was the
quarantine). Four of these were literals inside `eval_report.py`, reprinted on every
run — now computed or removed.

**MEDIUM — the golden set was never regenerated after the quarantine.** Two
`section_lookup` queries are drawn from the character-spaced document, so they resolve
to no chunk and `metrics.aggregate` drops them. Correct behaviour — but the summary said
`n=60` while all 48 result rows said 58, with nothing to explain it. The report now
reconciles the two explicitly. **Regenerating `corpus/evalset.jsonl` against the
post-quarantine corpus is required before Phase 6**, which builds golden answers on it.

Fixing that accounting, I made the same class of error inside the fix: counting the 60
`unanswerable` queries as "unscorable" and reporting 62 missing when the answer is 2.
They are by design not retrieval-scored — accuracy is undefined on an empty relevant
set — and drive separability instead. Caught by reading the generated output.

**LOW-MEDIUM x2 —** the sweep table printed two different token definitions in one row
(`chars/4` estimate beside a real-tokenizer percentage, the estimate running ~14% high),
now labelled; and `contextual.py` contradicted itself on request count (317 vs 425).

**LOW — the "542 blurbs" example was stale *and* mis-attributed.** 542 was the 2,048
figure, and the document is an FDA compliance-policy index, not a protocol. At 1,024 the
largest is 765 blurbs (~61K output tokens). Plus: 0.018 -> 0.017; "~5% of chunks" is
5.0-6.6% by strategy; "zero clean lines" is 1 of 8,448; "the two strategies" when there
are three.

**LOW — the tie-break guarantee overclaimed.** `retrieval.py` said its `chunk_id`
tie-break made metrics "reproducible from the same inputs". It makes *ordering*
deterministic, not *membership*: `bm25s` selects the top-k by `argpartition` before the
sort runs, so a score tie straddling the k boundary is still arbitrary. That is the
entire cause of the 5 non-reproducible figures out of 2,055 — two `recall@20`/
`precision@20` pairs and one rerank ceiling. Documented with the real fix (retrieve
k+ties, truncate after sorting) and why it was not taken: only window-edge metrics are
affected and no headline figure depends on them.

**Recorded as unverifiable rather than verified:** `semantic` chunk *boundaries* (needs
per-sentence embeddings and a per-document percentile — the artifact's metrics all check
out, but the algorithm is unaudited), MinHash near-duplicate *detection*, the
line-number "34.4% of chunks / 1.4% of tokens" figure (from a hand pilot, no rule in code
to reimplement, so unfalsifiable as written), and the contextual-retrieval dollar amounts.

### Verification: code review gate

Ten findings, every one reproduced before being acted on, all fixed with regressions.
Test suite 377 -> **384**.

**HIGH — the cost estimator's "pessimistic" bound was not a bound.** The plan says to
quote it when deciding whether to spend, and it failed two ways. `user_content()` sets
`cache_control` unconditionally, so a cache *miss* is still billed as `cache_creation`
at 1.25x — billing misses at 1.0x understated the true worst case by 17%. Worse, a
single-group document pays 1.25x once under "optimistic" and 1.0x once under
"pessimistic", so **the pessimistic figure came out lower than the optimistic one for 67
of 154 documents.** Reproduced exactly ($0.1000 vs $0.1300 on a one-group document),
fixed, and verified: 0 of 154 documents now invert, and the corrected band is
$6.81–$15.62 batched on Haiku.

**MEDIUM — the module docstring's headline cost table did not reproduce.** (The audit
separately found the largest-document figure was both stale and mis-attributed: 542
blurbs was the 2,048 figure, and the document is an FDA compliance-policy index, not a
protocol. At 1,024 the largest is 765 blurbs.) It named an
exact configuration and claimed $4.35; rebuilding that configuration and running the
module's own estimator gave $5.97, and the shipped 1,024-char corpus is $6.81. The
request counts were stale too (317/9,413 vs 425/13,706). The docstring is portfolio-facing
prose asserting numbers the code disagreed with — now generated from the same estimator
it documents.

**MEDIUM — a cache breakpoint that could never fire.** The system prompt carried
`cache_control` with the comment "caches once for the whole run". At ~290 tokens it sits
far below the minimum cacheable prefix — **4,096 tokens on Haiku 4.5, 512 on Opus 5** —
so nothing could ever cache, silently (`cache_creation_input_tokens: 0`, no error). The
breakpoint is removed and the minimum is now a per-model field, because it also bounds
the *document* breakpoint: 33 of 154 documents are below the Haiku minimum and never
cache at all. `estimate_cost` reports `cacheable_groups` so "grouping makes caching a
bonus" is checkable rather than assumed. The minimum is also **not monotonic across
generations** — the cheaper writer needs the larger prefix.

**MEDIUM — the sweep and the chunk pipeline used different quarantine rules.** `cmd_chunk`
checked the stored verdict *and* re-applied the threshold; `cmd_sweep` checked only the
verdict. They agree today only because extraction writes both from one computation — but
extraction is content-hash cached, so **retuning `CHARACTER_SPACING_THRESHOLD` without
`--force` silently put different corpora behind two reports that claim to describe the
same one.** Now one shared `_is_quarantined` predicate.

**LOW-MEDIUM — the sweep report hardcoded "15%"** while `--overlap-ratio` is
configurable, so a report could assert a control it had not applied. Now read from the
payload. **LOW-MEDIUM — `single_char_token_share` fails open on whitespace-free text**: a
120K-character document with no spaces collapses to one token and scores 0.0, so a
differently-corrupt glyph run would pass the very check added to catch glyph-positioned
PDFs. Recorded as a documented limitation rather than a speculative second detector.

**LOW — two figures were rounded wrong** (0.161 -> 0.159, 0.729 -> 0.723), here and in
`pdfcheck.py`. **LOW — `apply_blurbs` was not idempotent**: re-applying double-prepended
the blurb, contradicting the "partial runs degrade coverage, never corrupt" guarantee on
exactly the resume path a metered job needs. **VERY LOW — `parse_blurbs` could
misassign** on a duplicate `chunk_id` within a group, unreachable from
`chunking.build_chunks` but a hole in an absolute guarantee.

**Claims the gate verified and I had asserted correctly:** the document block is the
cacheable prefix and its bytes are identical across every group of a document (tested
against unicode, quote, and tab titles, and four `metadata` shapes); `apply_blurbs` never
touches `text`/`start`/`end`; `custom_id` uniqueness survives 9 adversarial doc_ids x
20,000 group indices including the 5-digit overflow; `CostReport` arithmetic and all
current prices; every per-model request-shaping comment; quarantine scoping across
strategies; the sweep writes nothing to `data/chunks/`; ragged sweep rows render without
column shift; and all 20 figures in the `DEFAULT_TARGET_CHARS` comment table match
`chunk_size_sweep.json` exactly.

One scoping limitation worth recording: `is_character_spaced` *is* fooled by legitimately
token-dense content — a single-digit table scores 1.000, math notation 1.000,
one-letter-per-line verse 1.000 — but none occurs at document scale here, because the
metric is whole-document and real prose dominates (max legitimate 0.159). A per-page
variant would be the fix if that ever changes.

### Verification

Test suite 292 -> **377**. New coverage: character-spacing detection (7 tests,
including the verbatim corrupt string and the worst legitimate document as a
false-positive guard), contextual retrieval (57), and the sweep report (9 — centred on
the recall-denominator trap, so a future reader cannot be misled by the column the
sweep itself warns about).

Both gates still to run on this phase.

---

## Phase 3 — Dense retrieval, fusion variants, reranking, semantic chunking

**Date:** 2026-08-13
**Status:** **shipped** — both verification gates passed; all 18 findings reproduced
and fixed, ablation table regenerated against the fixed code
**Artifact:** `reports/retrieval_eval.md` (ablation table v1)

### Headline result: the project's central claim is false on one slice

The thesis a RAG portfolio project is supposed to demonstrate is "hybrid retrieval
beats dense-only." Measured on this corpus, the interesting finding is the opposite,
and it is worth more than a table where every number goes the right way.

| Slice | Chunking | BM25 | Dense (bge-small) | Hybrid (RRF) |
|---|---|---|---|---|
| `exact_identifier` recall@10 | `fixed` | **0.992** | 0.032 | 0.171 |
| `section_lookup` recall@10 | `structural` | 0.892 | 0.903 | **0.934** |
| `section_lookup` recall@10 | `fixed` | 0.587 | 0.434 | 0.561 |
| `title_lookup` hit@10 | `fixed` | 0.983 | 1.000 | 1.000 |
| `unanswerable` separability | `fixed` | **0.975** | 0.875 | 0.533 |

The chunking column was added after the numerical audit. Without it this table mixed
populations — `structural` for section lookup, `fixed` for the other three rows — and
read as one system's per-slice profile, which made hybrid look like a uniform win. It
is not: on `fixed`, hybrid *loses* to BM25 on section lookup (0.561 vs 0.587). Exactly
the "figure computed over one population, presented beside prose implying another"
class the Phase 0 audit flagged, recurring in the summary rather than the code.

**Dense retrieval scores 0.032 against BM25's 0.992 on exact identifiers — a 31x
gap.** This validates the Phase 0 corpus choice with a measurement rather than an
argument: `21 CFR 314.50` and `21 CFR 314.70` are nearly identical as strings, so they
land almost on top of each other in embedding space while referring to entirely
different regulations. Dense retrieval is structurally incapable of the task, and it
is not a tuning problem.

**Naive fusion is worse than sparse alone on that slice: 0.171 versus 0.992.** The
mechanism was traced on a single query rather than inferred. For
`What are the requirements of 21 CFR 1.980?`:

- the correct chunk is at **BM25 rank 1**, and is **absent from dense's top 100**
- every one of the fused top 5 is irrelevant, and **all five appear in both lists**
- the winner sits at BM25 rank 4 / dense rank 2, scoring `1/64 + 1/62 = 0.0317`
- the correct chunk scores `1/61 = 0.0164` from one list only

RRF rewards agreement between retrievers. When one retriever has no signal on a query
type, that "agreement" is noise outvoting evidence. Weighting dense at 0.7 — exactly
what the source guide prescribes — makes it worse still, at 0.074.

**RRF also destroys the refusal signal.** On `fixed`, separability collapsed from 0.975
to 0.533 against a 0.500 majority baseline. RRF discards score magnitude by
construction, so the fused top score is near-constant whether or not an answer exists.

Min-max keeps magnitude, and the first version of this entry claimed it "retained
0.842" — but 0.842 is `structural`, while the paragraph is about `fixed`, where min-max
tops out at **0.525** against a 0.500 baseline. The audit caught it. Measured across
all three strategies:

| Min-max, dense weight | `fixed` | `structural` | `semantic` |
|---|---|---|---|
| 0.1 | 0.508 | 0.600 | 0.533 |
| 0.5 | 0.525 | 0.742 | 0.633 |
| 0.7 | 0.525 | **0.842** | 0.733 |

So the tradeoff `fusion.py` documents is real but narrower than stated: keeping
magnitude *can* preserve the refusal signal, and does on `structural`, but no fusion
configuration tested recovers BM25's 0.975. The constraint on Phase 5 is therefore
**validate the refusal threshold against the retriever that actually ships**, rather
than inferring it from the fusion method.

### Truncation was a silent retrieval defect

Text past the context window is never embedded, so it can never be retrieved — but it
fails as mediocre metrics rather than as an error. Measured against bge-small's
512-token window before normalization:

| Strategy | chunks truncated | corpus tokens dropped |
|---|---|---|
| `fixed` | 33.1% | 16.8% |
| `structural` | 18.8% | 16.0% |

Chunk size was not the cause — no chunk exceeds 2,464 characters against a 2,048
target (2,298 for `fixed` and `structural`). **Tokenization density was.** Regulatory text runs 4.54
chars/token at the median, 2.10 at the 5th percentile, and **1.16 at the floor** — and
the floor is table-of-contents dot leaders, where `....................` costs roughly
one token per character. One chunk consumed 1,819 tokens for 2,108 characters of pure
navigational filler, evicting everything after it.

My first hypothesis was that these were schedule-of-assessment tables, which the Phase
0 corpus table had predicted would be the hard case. Inspecting the actual worst
chunks disproved it: they are ToC dot leaders. Collapsing leader runs reclaimed 11.2%
of tokens and cut loss to 9.6%.

This is safe only because Phase 1b kept `text` (byte-exact, what citations resolve
against) separate from embedding input. Normalization touches the latter alone, so no
span in the golden set moved. That split was built for heading prefixes and paid off
again here.

### Semantic chunking, now implemented

Deferred since Phase 1b for want of an embedding model. Splits at troughs in
consecutive-sentence cosine similarity, thresholded at a **percentile of each
document's own distance distribution** rather than an absolute distance — absolute
distances are not comparable across documents, so one fixed threshold shatters a
tightly-worded regulation and leaves a heterogeneous protocol whole.

| Strategy | chunks | median chars | overlap inflation | heading tokens | build time |
|---|---|---|---|---|---|
| `fixed` | 6,628 | 2,115 | +16.6% | 0 | ~1s |
| `semantic` | 7,404 | 2,055 | **0%** | 0 | **~3min** |
| `structural` | 9,413 | 1,600 | +10.2% | 100,631 | ~1s |

`semantic` is the only strategy whose index tokens equal its corpus tokens exactly: no
overlap by design, because overlapping across a boundary re-introduces the bleed the
method exists to remove. It is also the only one whose cost scales at *chunking* time,
and the sentence embeddings are discarded afterwards since chunks are re-embedded as
units. That asymmetry is the honest argument against it: `structural` is free, so
semantic has to beat it to be worth anything.

### A token discrepancy that was not a bug

`semantic` reported fewer corpus tokens than the other two strategies, which looked
like lost text. Checked per document against the source without going through the
chunking code:

| | gap chars | docs affected | **non-whitespace lost** |
|---|---|---|---|
| before the runt fix (8,001 chunks) | 1,522 | 116 | **0** |
| after the runt fix (7,404 chunks) | 272 | 1 | **0** |

Sentence splitting yields whitespace-only spans where a blank-line block sits between
sentences, and `build_chunks` correctly declines to make those into chunks; `fixed` and
`structural` never hit it because their windows always contain prose. Absorbing the
runts merged most of those spans into their neighbours, which is why the gap shrank by
82% as a side effect of an unrelated fix. The residual 272 characters account for
exactly the 68-token difference against the other strategies (2,996,532 vs 2,996,600).

This reframed the invariant. The tiling tests asserted the wrong thing — spans tile
exactly, but *assembled chunks* legitimately do not. The test now asserts the version
that matters: no non-whitespace character is ever lost.

Worth recording as a process note: the first version of this entry cited the 1,522/116
figures after the runt fix had already changed them. Measured before a fix, quoted
after it — the same class of error the verification gates keep finding, caught here by
re-running the measurement rather than trusting the note.

### Verification: code review gate

Ten findings, **every one reproduced before being acted on, zero false positives**.
For the first time in this project the arithmetic was not the problem — the *claims*
were. All ten are fixed with regression tests.

**HIGH — `_oversize_split` produced runt chunks the docstring claimed were
impossible.** The function began as a copy of `chunking._windows` and the copy dropped
its closing `_absorb_runts` call. Because cuts snap *forward*, a span of `target + 2`
split into `target + snap` plus a **1-character chunk**. Result: **597 of 8,001
semantic chunks (7.5%) below `MIN_CHUNK_CHARS`, smallest a single character** —
`'M'`, `'. '`, `'! '` — each consuming a BM25 posting, a cache row, and a matrix row,
and each retrievable. Two of the three `chunk_semantic` return paths reach that
function without passing through `_spans_from_cuts`, so absorbing there was not
enough. `reports/chunking_report.md` was already printing `Chars min = 1.0`: the
artifact contained the counter-evidence to its own documentation. After the fix:
**7,404 chunks, 0 runts, min exactly 200.**

**MEDIUM — the embedding cache key omitted the section heading.** Keyed on
`content_hash`, which hashes `text`; but under `heading_mode="prepend"` the embedded
string is `heading + body`, and only the *mode name* was in the key. Two groups in
`structural` had byte-identical bodies under different headings, collided, and the
second silently received the first's vector. Impact today was 4e-5 cosine and no
published number moved. The latent half was worse: `content_hash` also **case-folds**,
invisible under bge-small's uncased WordPiece but wrong under **bge-m3's cased
SentencePiece — the plan's locked destination model**. 35 groups in `semantic` differ
only by case or whitespace, so adding the intended model would have silently produced
wrong vectors. The key is now the hash of the exact prepared string.

**MEDIUM — the rerank recall ceiling described a different list than the one
reranked.** `search` scores `max(candidate_k, k)` candidates while the ceiling used
`candidate_k`, so any `--candidate-k` below the harness's `max(K_VALUES)=20` yields
rows whose recall exceeds their own printed ceiling — which the report prose declares
to be a wiring bug. The default of 50 hid it; the first `--candidate-k 10` sweep row
would not have. Now computed over the effective window.

**Four false claims in my own comments** — the category this gate keeps finding:

1. `fusion.py` claimed vanilla RRF's *ranking* is window-independent. Its **scores**
   are; its ranking is not, since widening the window admits new documents — and
   `HybridRetriever`'s own docstring 130 lines below made exactly that point. One
   portfolio-facing module contradicting itself.
2. The `try/finally` around `cache.flush()` claimed to preserve vectors "already
   computed" across a later failure. There is no later step that can fail: truncation
   counting runs before encoding, and a crash inside encoding is all-or-nothing.
   Verified — an embedder raising after internally encoding 90 of 100 texts leaves
   **0 vectors on disk and no files written**. The construct bought nothing.
3. `count_truncation` read `st.tokenizer`, which loads the full `SentenceTransformer`,
   defeating the lazy-load rationale on every eval run. Now loads `AutoTokenizer`
   alone.
4. `semantic.py` called `target` a "hard cap". It is `target + 449`: forward snapping
   adds up to 250 and absorbing a runt extends its predecessor by up to 199. The
   largest semantic chunk is 2,464 chars against 2,298 for the other strategies.

**Three lower-severity real defects:** `hits`/`misses` counted on different
denominators, so the printed cache line read as an accounting of every chunk while
omitting within-call duplicates (9,409 encoded against 9,413 requested, with nothing
explaining the gap — now `hits + misses + deduped == len(keys)`); `get_or_encode`
validated returned vector *count* but not *dimension*, so a wrong-width vector was
accepted and written, after which every later run discarded the whole file and
re-embedded the corpus silently forever; and the "append-only, never invalidates"
claim holds only for a single writer, since `flush()` rewrites the entire array.

**One of my own tests asserted nothing.**
`test_normalize_never_applied_to_citation_text` ended in `or True`, so it would have
passed even if the retriever had mutated chunk text. Rewritten to check both halves.

**Findings the reviewer investigated and disproved** — recorded because a gate that
only ever confirms suspicions is not independent:

- The duplicated BM25 construction in `_retrievers_for` was checked for divergence and
  had none: over all 9,413 structural chunks both indexes return identical
  `(chunk_id, score, rank)` triples. The fusion rows genuinely are comparable to the
  baseline they are read against. Consolidated to one object anyway, since two call
  sites can drift and one cannot.
- Min-max fusion with **negative** scores is correct — min-max is affine-invariant, so
  an all-negative cosine list normalizes identically to the same list shifted
  positive. No sign bug.
- Normalization cannot reach `text`: `normalize_for_embedding` is a pure `str -> str`
  reachable only through `prepare_document`/`prepare_query`, and no assignment to
  `chunk["text"]` or `Chunk.text` exists anywhere in `src/`. Citation offsets are safe.
- Unit-norm vectors survive the cache round trip: all committed vectors have
  norm 1.000000 to 6 dp, so dot product is cosine through the cached path too.
- Semantic span tiling holds under **8,000 fuzz cases** over an adversarial alphabet
  (terminal punctuation, brackets, dot leaders, tabs, CJK, accents, astral-plane
  emoji) crossed with 5 targets and 2 embedder behaviours, plus 19 handcrafted edge
  cases: zero gaps, zero overlaps.
- Cache corruption resilience holds: truncated `.npy`, garbage bytes, malformed
  `.keys.json`, key/vector length mismatch, and dim mismatch all degrade to a cold
  cache with no exception and no leaked `.tmp` files.

### Verification: numerical audit gate

The stricter gate the plan assigns to Phases 2, 3, and 5. The auditor reimplemented
BM25 (Lucene variant), the identifier tokenizer and atom encoder, cosine retrieval,
all three fusion methods, every metric, the text normalizer, and the blake2b cache-key
construction — **with zero `ragpipe` imports** — then recomputed 178 of 192 report rows
cell by cell.

**All 8 headline claim groups reproduce exactly.** Ten review passes now, and the
arithmetic has never been wrong. Verified to 4 decimal places: the 31.04x
dense-vs-sparse gap (0.9917 / 0.0319), all four truncation fields across three
strategies, all 24 chunking-stats fields, the whole `fixed` weight sweep, the
single-query RRF trace including `1/64 + 1/62 = 0.031754`, and the +0.392 / +0.381 /
+0.187 decision deltas. Also clean: 192/192 rows match between the markdown and the
JSON, every non-unanswerable row has n = 60, every separability pool is exactly 60/60
against a 0.500 baseline, no recall exceeds 1.0, **no row exceeds its own ceiling**,
all 23,436 cached vectors are unit-norm and reproduce from an independent key
construction, and all 60 unanswerable citations are genuinely absent from all 155
documents. Row labels were checked for a mirrored weight pair and are correct.

Confirmed on the rerank finding: the `fixed` `bm25 + rerank` ceiling really is
**exactly 1.000** — every relevant chunk for all 60 `exact_identifier` queries was
inside BM25's top 50, and recall@20 for that row is 0.955. So the 0.992 -> 0.891 drop is
**entirely the cross-encoder demoting correct chunks**, not a first-stage miss.

Eight findings, all reproduced. Two were real bugs in retrieval; the rest were stale or
mislabelled prose.

**MEDIUM-HIGH — the rank-fusion rows were not reproducible, because the BM25
out-of-vocabulary guard never fired.** `bm25s.tokenize` on a *string* query silently
drops unseen terms, so a fully out-of-vocabulary query returns a non-empty token list,
an empty id list, and an **all-zero score vector** — and the top-k of a constant array
is an arbitrary k documents. The guard checked only for empty tokens, and its comment
asserted "bm25s raises rather than returning an empty ranking", which is false for
string queries. Reproduced exactly: `title-0012` is `NEPA_Final_Guidance`, one token
under bm25s' `\b\w\w+\b` pattern because underscore is a word character, out of
vocabulary, returning **100 hits all scoring 0.0**. RRF then consumed that noise as
ranks 1..100. The auditor ruled out tie-shuffling and device differences, and showed
the committed run had seen yet another arbitrary ordering. 1 query in 240, affecting
rank-1 metrics on `title_lookup` fusion rows only; hit@10 stayed 1.000 either way, so
no conclusion moved.

Fixed by dropping zero-score hits outright — a zero BM25 score is the *absence* of
evidence, not weak evidence, and feeding it to fusion as a rank lets a document that
matched nothing outrank one that matched, purely by appearing in two candidate lists.

**LOW, same family — `BM25Retriever` did not break score ties on `chunk_id`**, though
`fusion._ranked` and `CrossEncoderReranker.search` both do and both cite determinism as
the reason. Query `section-0059` has an exact tie at 6.600398 between a relevant and an
irrelevant chunk, so rank-1 metrics depended on `bm25s`' internal ordering. recall@10
was unaffected. Both fixes required regenerating the ablation table.

**There was no `tests/test_retrieval.py` at all** — the retriever was covered only
incidentally, through whatever the eval harness happened to exercise, which is how two
reproducibility bugs survived to the audit. Now 12 tests, including the exact
`NEPA_Final_Guidance` case and stability across input order.

**MEDIUM — the headline tables in `README.md` and this file silently mixed chunking
strategies.** No strategy column: rows 1, 3, 4 were `fixed`, but `section_lookup` was
`structural`. Read as one system's per-slice profile, it made hybrid look like a
uniform win. It is not — on `fixed`, hybrid **loses** to BM25 on section lookup, 0.561
against 0.587. This is the same "figure computed over one population, presented beside
prose implying another" class the Phase 0 audit flagged, recurring in the summary
rather than in the code. Both populations are now shown.

**MEDIUM — "min-max keeps magnitude and retained 0.842" was a different strategy from
its own paragraph.** The paragraph is about `fixed` collapsing 0.975 -> 0.533; `fixed`'s
best min-max separability is **0.525** against a 0.500 baseline, essentially nothing.
0.842 is `structural`. The sentence's load-bearing claim was false for the strategy
under discussion, and it was a claim about Phase 5's refusal path — so the corrected
version is deliberately narrower: no fusion configuration tested recovers BM25's 0.975,
and the refusal threshold must be validated against whichever retriever ships.

**MEDIUM — "the largest chunk is 2,298 characters … none exceed 2,400" was false** once
`semantic` landed at 2,464, and stale in three places (`README.md`, this file,
`embedding.py`) while `eval_report.py` had already been corrected — so the repo
contradicted itself. **LOW-MEDIUM — a mean quoted as a median**: "2.4 chunks under
`fixed`" is the mean; the median is 2. **LOW — 95.8% and 41.8% printed side by side on
different denominators**: 95.8% is of all chunks, 41.8% is of *labelled* chunks; over
all chunks it is 40.1%. Plus stale figures in `tokenize.py` (the atom example predated
uppercase marking), a vector count from the pre-fix two-strategy cache, and a ceiling
range given as 0.6–0.7 where the measured range is 0.606–0.740.

**Recorded for Phase 5, not a defect now.** `ident-0031`'s ground-truth span in
`fda-148646` points at the literal text `21 CFR 720` while the cited canonical is
`21 CFR 312.3` — the documented line-number-contamination repair records the raw
match's offsets. All four resolved chunks do contain `312.3`, so no metric is affected.
But Phase 5's citation verifier will assert `text[start:end]` against the cited
identifier, and this span will not match. Handle it there.

**Could not verify:** 14 of 24 cross-encoder rerank rows (all 18 recall ceilings and 10
full rows were recomputed exactly; the rest were ~50 minutes of remaining forward
passes), the pre-fix historical figures quoted in this entry for states no longer on
disk, and the build times.

### Also fixed this phase

- **A partial re-chunk silently narrowed a committed artifact.**
  `ragpipe chunk --strategies semantic` rewrote `reports/chunking_report.md` to cover
  `semantic` alone, dropping `fixed` and `structural` from the comparison table. The
  chunk files survived, so nothing looked broken — the artifact just quietly stopped
  being a comparison. The report now re-derives any strategy it did not just build
  from chunks already on disk.
- **`eval_report.py`'s separability table declared 8 columns and supplied 6.**
  `Baseline` and `Gain` were headers with no data under them.
- **`retrieval.py`'s docstring carried stale numbers**, claiming `strip` scored 0.486
  for a +0.406 heading effect. The committed report says 0.511 and +0.381. Also added
  the missing note that all three heading modes score identically for `fixed`, which
  has no headings to strip — the reason the confound stayed invisible until the
  strategies were compared.

### Standing hazard, updated

Across ten verifier runs: **zero false positives, arithmetic wrong zero times.** The
numerical audit recomputed 178 of 192 rows from raw data with independent
implementations and found every headline figure exact to 4 decimal places.

Every real defect has instead been a *claim*: a docstring, a comment, a report's prose,
a table missing the column that made it honest, or a test that asserted nothing. This
phase produced one of each — a copied function whose docstring promised a safeguard the
copy had dropped, a guard whose comment described a failure mode that does not occur for
the input type it handles, a summary table that mixed two populations, a separability
figure quoted from a different chunking strategy than the paragraph discussed, and
`assert ... or True`.

The refined lesson: the arithmetic is defended by tests, so it holds. What nothing
defends is the sentence *next to* the number. Two specific habits have caught the most:
re-measuring after a fix rather than trusting a note written before it, and requiring
every reported figure to name the population it was computed over.

A third now joins them, from this phase's two retrieval bugs: **check that a component
is reproducible before trusting any number derived from it.** Both defects produced
plausible metrics from non-deterministic rankings, and neither would have been caught by
comparing a number against a previous run of the same broken code.

---

## Phase 2 — Retrieval eval harness and the first ablation

**Date:** 2026-08-13
**Status:** **shipped** — both verification gates passed, all findings fixed
**Artifact:** `reports/retrieval_eval.md`, `corpus/evalset.jsonl`

### Headline results

Two design decisions, both now measured rather than asserted.

**Identifier-aware tokenization is worth +0.392 recall@10** on the
exact-identifier slice (0.992 vs 0.600 for `fixed`). Default BM25 tokenization
shatters `21 CFR 314.50` into `['21','cfr','314','50']` — and drops the `.5`
entirely from `21 CFR 314.5` — so distinct regulations collapse into identical bags
of the most common tokens in regulatory text. The slice built to demonstrate that
lexical retrieval beats dense would have scored 0.60, and the conclusion drawn
would have been "this corpus is unsuitable" rather than "fix the tokenizer".

**Structural chunking's advantage on section lookup decomposes into two effects:**

| Configuration | recall@10 |
|---|---|
| `structural` + headings | 0.892 |
| `structural` − headings | 0.774 |
| `fixed` | 0.587 |

So +0.187 comes from section-aligned boundaries and +0.118 from prepending the
heading to the indexed text. The first version of this report attributed all
+0.305 to boundaries, which was wrong — see the correction below.

Headings make **zero** difference on the exact-identifier slice (0.992 both ways)
and atoms make zero difference on section lookup, which is the behaviour a correct
factorial ablation should show: each intervention moves only the slice it should.

### Corrections

**The eval set's ground truth was chunking-dependent, which invalidated every
cross-strategy comparison.** Relevant *chunk ids* were stored, so scoring a
strategy the set was not built on fell back to document-level truth: `structural`
reported recall@10 of 0.993 against `fixed` at 0.123 on identical queries, with
`fixed` simultaneously scoring *higher* on precision@10 — the tell. Ground truth is
now `(doc_id, start, end)` character spans, resolved to chunk ids at scoring time
against whichever chunk set is under evaluation. Chunking-agnostic by construction,
and it will hold for every dense, fused, and reranked row added in Phase 3.

**The strategy comparison was confounded by heading prefixes.** `structural`
chunks carry the section heading in `embed_text` (95.8%) and `fixed` chunks do not
(0%), so indexing `embed_text` unconditionally measured boundaries and headings
together. On the `section_lookup` slice — where the query *is* a heading — that is
close to writing the answer into the index. `use_headings` is now a retriever-level
flag and the ablation runs the full factorial, which is what produced the
decomposition above.

**Dedup was non-deterministic.** `shingles()` used Python's builtin `hash()`,
which is salted per process, so signatures, LSH buckets, cluster membership and
every reported similarity changed on each run — while three separate docstrings in
the module claimed determinism, and the committed cluster files could not be
reproduced. Now a stable BLAKE2b digest; three consecutive runs produce identical
signatures.

**"Verified exactly" was false.** Candidate pairs were screened *and decided* by
the 128-permutation MinHash estimate, whose standard error at J=0.85 is ~0.032 — so
4 of 46 shipped duplicate pairs were actually below the stated threshold. The
estimate now screens with a margin and exact Jaccard decides. Cluster
`similarity_floor` is also now the true minimum pairwise similarity across the
cluster rather than only against the canonical member: union-find is transitive, so
two members may never have been compared, and one cluster's real floor was 0.781.

### Verification gate — results

Two independent agents. All 32 result rows reproduced exactly against an
independent reimplementation, and all 37 hand-computed metric assertions passed —
`metrics.py` is correct, including the nDCG IDCG cap. Ground truth is essentially
clean: 0 answerable queries resolve to an empty relevant set, 0 cited regulations
occur outside their ground-truth spans, and all 60 synthetic unanswerable citations
are genuinely absent from every document.

**But two reported conclusions were wrong**, both in the interpretation layer
rather than the arithmetic.

#### The heading effect was understated 3.2x

`bm25 -headings` selected `text` over `embed_text` — and a section's `text`
*begins with its own heading*, because the section span starts at the heading line.
So 41.8% of `structural` chunks still carried the heading in the supposedly
heading-free condition. The variable was never isolated.

Three modes are needed, not two:

| Heading mode | `section_lookup` recall@10 |
|---|---|
| `prepend` (heading added to every chunk) | 0.892 |
| `source` (`text` as-is — heading still in 41.8%) | 0.774 |
| **`strip`** (heading removed everywhere) | **0.511** |

The real heading effect is **+0.381**, not the +0.118 previously reported. The
direction survives; the magnitude was off by more than 3x, and anyone sizing the
decision from +0.118 would have concluded headings were not worth the index cost.

#### Separability was measured against the wrong pool, and the winner flipped

`separability()` was passed every answerable query — 180 against 60 unanswerable —
while the report's own prose claimed the two groups were "phrased identically". That
is true only of the 60 `exact_identifier` queries. The other 120 are section
headings and document titles: long, high-IDF strings that score highly for reasons
unrelated to whether a citation resolves.

| | Old (all slices) | Corrected (like-for-like) |
|---|---|---|
| Best accuracy | 0.879 | **0.975** (`fixed`) |
| Majority baseline | 0.750 | 0.500 |
| Gain over baseline | +0.129 | **+0.475** |
| Best chunking | `structural` | **`fixed`** |

So the headline number was both weaker than it looked *and* ranked the wrong
strategy first — on the table that is the stated evidence for whether Phase 5 can
refuse on a score threshold. The baseline is now reported alongside, because an
accuracy figure without one is unreadable.

#### Also fixed

- `title_lookup` is fully saturated (0.983 on every configuration) and now carries
  an explicit callout that no ordering in it is evidence.
- `recall@k` across chunking strategies overstates the gap, because recall has a
  per-strategy denominator: the median section is 1 chunk under `structural` and 2.4
  under `fixed`. A caveat now points readers to `hit@k` for cross-strategy claims,
  which shows the same direction at roughly half the margin.
- `recall_at_k` counted positions rather than distinct ids, so a retriever returning
  a duplicate could have scored above 1.0. Unreachable today; a fused retriever in
  Phase 3 could have hit it.
- `separability()` never considered a threshold above the maximum observed score, so
  the degenerate "predict everything unanswerable" corner was unreachable.
- `atomic_token` lowercased, colliding `(a)` with `(A)` — while `identifiers.py`
  explicitly preserves subsection case because they are different subdivisions. Zero
  collisions in this corpus; now impossible by construction.

### Open items carried forward

- Phase 1b and Phase 2 verification gates not yet closed.
- Five chunking-report framing defects outstanding (task #11), including
  "section-aligned 95.8%" which actually counts chunks *carrying* a section label —
  genuinely boundary-aligned is 28.9% exact / 67.9% sharing one boundary — and an
  "identifier ceiling" that reports chunk density when the true ceiling is 100%.
- `title_lookup` saturates at hit@10 = 0.983 across every configuration. Recorded
  as non-diagnostic rather than presented as evidence.
- No tests for `dedup.py`; findings 1, 2, 10, 11 and 12 were all unguarded.
- Missing parameter validation: `--target-chars` below `MIN_CHUNK_CHARS` collapses a
  document into one chunk, and `--overlap-chars` ≥ target degrades to one span per
  character.

---

## Phase 1b — Chunking strategies and duplicate clustering

**Date:** 2026-08-13
**Status:** **shipped** — both verification gates passed, all findings fixed
**Artifact:** `reports/chunking_report.md`, `data/chunks/`

### What shipped

| Module | Responsibility |
|---|---|
| `chunking.py` | `fixed` and `structural` strategies, boundary snapping, runt absorption |
| `dedup.py` | MinHash + LSH near-duplicate clustering, exact-Jaccard decision |
| `chunk_report.py` | Phase 1b checkpoint report |
| `cli.py: chunk` | `make chunk` |

**Semantic chunking deferred to Phase 4, not faked.** It needs embedding-similarity
troughs and no embedding model exists until Phase 3. It is registered in the
strategy table with an explicit `NotImplementedError` explaining why, so the
comparison table cannot be mistaken for complete. A third row backed by a stub
would have been worse than two honest ones.

**Chunks carry two text fields.** `text` is the byte-exact source slice, because
Phase 5's citation verifier asserts cited spans exist in the source and a decorated
chunk cannot support that. `embed_text` prepends the section heading, because a
chunk reading "...within 15 days" is far more retrievable carrying "Subpart C —
Adverse Event Reporting". Same seam contextual retrieval needs in Phase 4.

### Numbers

| Metric | `fixed` | `structural` |
|---|---|---|
| Chunks | 6,628 | 9,413 |
| Index tokens (what gets embedded) | 3,493,403 | 3,403,848 |
| Corpus tokens (union of source spans) | 2,996,600 | 2,996,600 |
| Overlap inflation | +16.6% | +10.2% |
| Heading tokens | 0 | 100,631 |
| Both cuts on a section boundary | — | 3,964 of 9,015 labelled (44.0%) |
| Distinct CFR citations reachable | 521 | 521 |
| Duplicate clusters | 5 | 35 |

### The near-duplicate assumption was wrong

`docs/plan.md` deviation #4 overrode the guide's dedup approach, arguing that
regulatory boilerplate is legitimately near-identical so duplicates must be
*clustered* rather than skipped. The machinery is correct. There is almost nothing
to cluster: 6 chunks marked duplicate for `fixed` (0.1%), 36 for `structural` (0.4%).

The threshold is not the reason. In a 900-chunk sample **nothing exceeded 0.7**
Jaccard and only two chunks exceeded 0.5; the distribution falls off smoothly, so
there is no cluster hiding below the 0.85 bar. Lowering it to manufacture duplicates
would be fitting the method to a desired conclusion.

**The boilerplate is real, but lives at paragraph granularity.** Of 8,724
paragraph-like segments, 182 form exact-duplicate groups and 35 of those span
documents — the classic FDA disclaimer appears in 14 copies across 14 documents. At
~2,000-character chunks that text is diluted by the unique content around it, so
chunk-level Jaccard never approaches threshold. The guide's `cosine > 0.95`
chunk-level dedup would have found the same near-nothing, at more expense, since it
needs embeddings first.

Not acted on: 122 paragraphs of 8,724 is 1.4%, and there is no metric yet to say
whether removing it helps retrieval. That is a Phase 3 experiment, not a speculative
optimisation.

### Corrections

**Dedup was non-deterministic.** `shingles()` used Python's builtin `hash()`, salted
per process, so signatures, buckets, cluster membership and every similarity figure
changed on each run — while three docstrings claimed determinism and the committed
cluster files could not be reproduced. Now BLAKE2b. A subprocess test guards it,
because a same-process test structurally cannot detect per-process salting.

**"Verified exactly" was false.** Candidate pairs were screened *and decided* by the
128-permutation MinHash estimate, whose standard error at J=0.85 is ~0.032, so 4 of
46 shipped pairs were actually below threshold. The estimate now screens with a
margin and exact Jaccard decides.

**`similarity_floor` was not a floor.** Measured only against the canonical member,
it reported 0.891 for a cluster whose true minimum pairwise was 0.781. Now the exact
minimum across every pair. One cluster legitimately reports 0.781 — single-linkage
transitivity, with pairs at 0.891 and 0.876 both clearing the bar and the outer two
never compared directly. Documented in the report rather than hidden.

**The LSH bucket cap silently dropped duplicates.** `sorted(members)[:200]` excluded
high-sorting ids from every band; a 260-item near-identical set lost 7. Buckets now
hold one representative per exact-duplicate group, which is what the cap was
reaching for. All 260 now cluster.

**Two sizing bugs.** An 11-character chunk from an overlap step-back leaving a runt
trailing window, and `pending_idx or 0` mislabelling an absent section as section 0.
A two-document smoke test missed the first; only the corpus-wide check found it.

**A regression I introduced while hardening against section gaps.** The front-matter
region was covered both by a pre-existing special case and by the new gap loop —
**104 documents ended up with duplicate spans.** One loop now covers every
uncovered region. Also: my claim that the fix "recovered previously-dropped text"
was wrong. Measured properly, genuine mid gaps are 0 and tail gaps are 0; the
480,803 characters I counted were all leading front matter, already handled. The fix
is defensive hardening, not recovered content.

**Missing parameter validation.** `--target-chars` below `MIN_CHUNK_CHARS` made every
window a runt and collapsed a whole document into one chunk; `--overlap-chars` at or
above the target advanced one character at a time. Both now raise.

### Report framing, five defects

All 23 reported figures were arithmetically correct; the framing was not.

| Defect | Fix |
|---|---|
| "Est. tokens" double-counted overlap while the prose reassured on the wrong axis (the 4-chars-per-token constant is shared; the double-count rate is not) | Split into index tokens, corpus tokens, overlap inflation, and heading overhead |
| **Then, in the fix:** I labelled index tokens an embedding cost while summing `len(text)`, understating `structural` by its entire 100,631-token heading overhead — an asymmetric error hitting only the strategy the column exists to compare | Computed over `embed_text`, the field actually indexed |
| "Section-aligned 95.8%" counted chunks *carrying* a label; sub-split windows inherit their parent's index | Four-way breakdown against real section offsets |
| **Then, in the fix:** the boundary envelope was derived from the labelled chunks themselves — self-referential, reducing to "was this group emitted as one chunk" | Boundaries now read from the extracted documents, with the definition stated because two readings are plausible |
| "Identifier ceiling" reported chunk density; the true ceiling is 100% (all 521 distinct citations reachable) | Ceiling reported first, prevalence labelled as haystack density and flagged non-comparable across strategies |
| Prose called within-document clusters overlap artifacts | Corrected: 15% overlap cannot reach 0.85, and one cluster of adjacent windows still scores 0.880 with the shared region deleted — repetition clustered them, not overlap |

`dedup.py` had **zero tests**, which is why five of its defects were unguarded. It
now has 22.

---

## Phase 1a — Text extraction, structure, and identifiers

**Date:** 2026-08-13
**Status:** **shipped** — both verification gates passed, all findings fixed
**Artifact:** `reports/extraction_report.md`, `reports/extraction_stats.json`, `data/extracted/`

### What shipped

| Module | Responsibility |
|---|---|
| `extract.py` | PDF → single text blob with page offsets, section structure, identifiers |
| `identifiers.py` | Regex + structural canonicalisation for CFR / U.S.C. / FR / docket / NCT / ICH |
| `extract_report.py` | Phase 1 checkpoint report |
| `cli.py: extract` | `make extract`, incremental by content hash, `--force` to override |

Text is stored once as a blob with a page offset map rather than duplicated per
page, because chunking slices by character offset and Phase 5's citation verifier
needs spans into the same coordinate system.

Incrementality reuses the Phase 0 sha256 pins as the change-detection key: a
document is re-extracted only when its manifest hash differs from the hash
recorded in its extraction output. One artifact, three jobs now.

### Numbers

| Metric | Value |
|---|---|
| Extracted | 155 / 155, zero failures |
| With ≥1 section | 147 (94.8%) |
| Structure source | 88 outline, 59 heuristic, 8 none |
| Sections per document | 0 / 16 / 139 / 966 (min/median/p90/max), 7,368 total |
| Characters per document | 694 / 40,816 / 187,891 / 801,075 — 11.99 M total |
| Characters per page | 564 / 2,341 / 3,043 / 6,951 |

### The exact-identifier question, answered

Phase 0 left open which identifier class could carry the exact-identifier eval
slice — the slice that demonstrates lexical retrieval beating dense retrieval. It
had assumed dockets, because 93 documents list one in *metadata*. Measured over
the full corpus body text:

| Identifier | Docs | Coverage | Occurrences | **Distinct** |
|---|---|---|---|---|
| **`cfr`** | **107** | **69%** | **2,553** | **575** |
| `usc` | 73 | 47% | 319 | 166 |
| `fed_register` | 45 | 29% | 143 | 87 |
| `docket` | 26 | 17% | 41 | 29 |
| `registry` | 20 | 13% | 56 | 40 |
| `ich` | 20 | 13% | 55 | 16 |

**Decision: the slice is built on CFR citations.** 575 distinct citations across
69% of the corpus is ample; dockets offer 29 across 17%. `identifiers.PRIMARY_KIND`
records the choice in one place so the eval harness and the report cannot disagree.

`distinct` is the load-bearing column, not `occurrences`: 2,553 mentions of 575
citations supports 575 questions, not 2,553.

### Findings

**PDF outlines cannot be trusted merely because they exist.** The design assumed
`outline` meant authored headings with real nesting. Two documents produced
**1,618 and 1,224 "outline entries" for 25 pages each** — these are accessibility
*structure trees* exposed through the same bookmark API. Every text run became an
entry, including single hyphens and empty strings; median heading length was 3
characters, and 1,593 of 1,618 sections were zero-length because all resolved to
page 1.

Caught by this phase's own quality flags (sections per 1,000 characters), before
either verifier ran. Left unfixed it would have handed structure-aware chunking
thousands of degenerate one-character sections on those documents, and Phase 3's
ablation would have shown structure-aware chunking losing badly for a reason with
nothing to do with chunking.

`outline_is_usable()` now validates on four bounds — entries per page,
substantive-title ratio, single-page concentration, minimum entry count — and
records the accept/reject reason in the output rather than silently downgrading.
**9 of 97 outlines were rejected**, removing ~2,900 degenerate sections (10,281 →
7,368 total).

One borderline case is recorded rather than tuned away: `fda-175603` has 460
entries over 57 pages (8.1 per page) and just failed the 8.0 threshold. Its
heuristic fallback produced 387 sections, which is also high. Tuning a threshold
to accommodate one document is overfitting, so it stays flagged for inspection.

### Corrections

**Identifier canonicalisation was collapsing whitespace instead of parsing
structure**, so `'21CFR 312.32'` and `'21 CFR 312.32'` counted as two distinct
citations. Since `distinct` sizes the eval slice, this inflated the very number
the slice is planned from. Canonicalisation is now per-class and structural:
each citation is re-assembled from its parsed parts, so `C.F.R.` vs `CFR`, a
missing space, `Part`, `§`, en-dashes, line breaks, and case all collapse. Eight
spellings of one citation now yield one value.

**CFR citations were being detected as section headings.** `21 CFR 820.30,
Subpart C - Design Controls` matches the numbered-heading shape but is a
reference. Lines opening with a legal citation are now rejected, and numbered
headings must begin with a capitalised word.

### Verification gate — results

Two independent agents. **Zero false positives from either, for the second phase
running.** Every finding was reproduced before being acted on.

#### The clean result that mattered most

The code reviewer's first priority was **character-offset correctness**, because
`pages[].start/end` and `sections[].start/end` index into the text blob and Phase
5's citation verifier will assert cited spans against them. An off-by-one would
have corrupted citation verification four phases downstream, presenting as
"span not found" rather than an error. Verified clean across all 155 documents:

- `"\n\n".join(text[p.start:p.end] for p in pages) == text` — 0 failures
- `p.end - p.start == p.chars` — 0 mismatches; `len(text) == n_chars` everywhere
- Sections: 0 overlaps, 0 gaps, 0 negative spans, `sections[-1].end == len(text)`
  in all 147 documents with sections
- All 3,167 identifier offsets satisfy `text[i.start:i.end] == i.raw`
- Cross-checked against a *fresh* pypdf extraction, not merely internal consistency

#### HIGH — outline sections anchored to page top

`_sections_from_outline` set every bookmark's start to the top of its page.
Because `_close_sections` sets each end to the next start and bookmarks routinely
share a page, **54.1% of outline sections were zero-length** — and worse than
empty, each page's whole text was attributed to the *last* heading on it:

```
p6  len=     0  'PART II - IMPLEMENTATION'
p6  len=     0  '1. Objective'
p6  len=  2345  'C. Remote Regulatory Assessment'   <- contains all of the above
```

31 documents had over half their sections empty; 27.4% of all sections corpus-wide.
Structure-aware chunking would have emitted mostly-empty chunks and mislabelled
the rest — and Phase 3's ablation would have blamed the chunking strategy.

Fixed by locating each bookmark title within its page text, with a three-tier
fallback (full title, prefix, whitespace-insensitive regex) for titles that do not
match extracted text byte for byte. Outline entries are also now sorted by page,
since bookmark-tree order is not guaranteed to be page order and an out-of-order
entry would have produced a silently-empty negative span.

**Result: zero-length sections 27.4% → 0.0%.**

#### MEDIUM

| Finding | Reproduced | Fix |
|---|---|---|
| **Plural citation forms matched nothing at all.** `§?` and singular `Part` meant `21 CFR §§ 211.42(d)` and `21 CFR parts 210 and 211` failed the optional group *and* took the following digits with them, so the whole match was abandoned | two documents reported zero CFR citations while visibly containing several | `§{0,2}` and `[Pp]arts?` |
| **ToC detection discarded body pages.** `\s{2,}\d{1,3}$` matches any line ending in whitespace and a number, which occurs on ordinary pages of line-numbered guidances and in data tables | `fda-119789`: 11 of 25 pages flagged as ToC, ~13 headings dropped — about half its structure | Dot-leaders classify pages; the trailing-number form only rejects individual lines. `fda-119789`: 14 → 24 sections |
| **Running-header suppression never did what its comment claimed.** It tracked the last three accepted headings with no page component | a page-footer date became **26** sections in one protocol (`'11 May 2015'` parsed as number `11` + text `May 2015`) | Keyed on (heading, page) with a 2-page window. 26 → 2 |
| **CFR/USC canonical granularity disagreed.** CFR discarded subsections, USC kept them, so distinct counts sat in one column at different resolutions | CFR distinct 521 at section level vs 1,098 with subsection | Both levels computed and reported separately for every class |
| **Line-number contamination.** PDF extraction interleaves the marginal line number with the citation, so `21 CFR 812.2(c)` on line 71 extracts as `21 CFR 71 \n812.2(c)` | 63 occurrences; `73 CFR 1271.3` survived visibly — there is no CFR title 73 | Title bounds (CFR ≤ 50, U.S.C. ≤ 54) plus repair when a decimal-less captured section precedes a decimal-bearing number on the next line |

#### LOW, all fixed

`needs_extraction` crashed instead of recovering on a mid-UTF-8-truncated output
(`UnicodeDecodeError` is a `ValueError`, not a `JSONDecodeError`, so it escaped the
handler and aborted the whole run) — now caught, and writes are atomic via
temp+replace. A usable-looking outline yielding zero in-range sections reported
`structure_source: "outline"` with no sections instead of falling through to the
heuristic detector. The `chars_per_page` "total" was a sum of 155 ratios with no
referent, printed beside two real corpus totals — dropped, with the corpus-wide
rate stated separately. A stale docstring claimed 223 numbered headings for a
document whose detector ceiling is 190. And `dense[:10]`/`thin[:10]` were truncated
*before* `len()` was used in the prose — the same defect already fixed once for
`docs_with_no_sections`, reintroduced in two more places.

#### Also disclosed rather than hidden

The identifier coverage column is computed corpus-wide, which concealed an almost
total source split. Now reported: `cfr` is 91/115 FDA documents versus 18/40
protocols (2,531 occurrences against 56), `docket` and `fed_register` are 100% FDA,
`registry` is 100% protocols. **The exact-identifier slice is therefore effectively
FDA-only** — acceptable, since it still exercises the behaviour under test, but a
property to state rather than bury.

The ~0.075% character-total gap against the Phase 0 report is now explained in the
report itself: exactly two characters per page boundary, from the `\n\n` separators
added when pages are concatenated into one addressable blob.

### Final numbers (post-fix)

| Metric | Value |
|---|---|
| Extracted | 155 / 155, zero failures |
| Structure source | 88 outline, 59 heuristic, 8 none |
| Sections | 7,306 total, **0 zero-length** (was 27.4%) |
| CFR — docs / occurrences | 109 (70%) / 2,587 |
| CFR — distinct section / +subsection | **521** / 1,098 |
| Identifier occurrences repaired | 63 |
| Tests | 95 passing (was 51 at end of Phase 0) |

### Known limitations, recorded not hidden

- Heuristic heading detection still produces false positives on protocol synopsis
  tables (`1 BioThrax given at Days 1, 15, and 29`). Not chased: the report flags
  dense-section documents for inspection, and Phase 3's ablation is the arbiter of
  whether structure-aware chunking is worth using at all.
- 8 documents produced no sections; structure-aware chunking must fall back to
  fixed-size for those, which is itself worth reporting as a comparison.
- The 20% / 80% text-layer thresholds from Phase 0 are still unrevisited against
  real extracted text.

---

## Phase 0 — Corpus acquisition and characterisation

**Date:** 2026-08-13
**Status:** **shipped** — both verification gates passed, all findings fixed
**Artifact:** `reports/corpus_report.md`, `reports/corpus_stats.json`, `corpus/manifest.jsonl`

### What shipped

`src/ragpipe/` with three stages behind `make corpus`:

| Module | Responsibility |
|---|---|
| `sources/fda.py` | FDA guidance catalogue → normalised records |
| `sources/ctgov.py` | ClinicalTrials.gov v2 API (`aggFilters=docs:prot`) → protocol records |
| `sample.py` | Deterministic stratified sampling (seed `20260813`) |
| `net.py` | Per-host throttling, backoff honouring `Retry-After`, hash-verified resumable downloads |
| `pdfcheck.py` | Per-page text-layer classification, table and identifier proxies |
| `stats.py` | Report generation |
| `cli.py` | `manifest` / `fetch` / `stats` |

### Numbers

| Metric | Value |
|---|---|
| Candidate pool | 2,232 FDA guidance docs with PDFs; 583 protocol docs across 600 studies |
| Sampled | 160 (120 FDA + 40 protocols) |
| Downloaded | 160 / 160, zero failures, 164 MB |
| Indexable | 158 (98.8%) |
| Text layer | 157 digital_native, 1 mixed, 2 image_only |
| Pages | min 1 / median 17 / p90 73 / max 228 — 5,123 total |
| Extracted characters | 12,718,891 total |
| Estimated tokens | ~3.18 M |
| Estimated chunks | ~14.6k @ 256 tok, ~7.3k @ 512 tok, ~3.7k @ 1024 tok |

Corpus size lands where it needed to: ~7.3k chunks at 512 tokens is a real
retrieval problem, and small enough that local re-embedding stays free — which is
the precondition for the chunking and embedding sweeps in Phases 1 and 4
happening at all rather than being skipped.

### Eval-slice viability (measured, not assumed)

| Slice | Precondition | Result |
|---|---|---|
| Version currency | Draft + Final of same guidance | **8 complete pairs** — viable |
| Table lookup | Tabular content | **676 table-like lines** — viable |
| ACL-filtered | Real document partition | **16 partitions**, largest 23 docs — viable |
| Temporal | Undated / spread dates | 4 undated docs; dates span decades — viable |
| Exact identifier | Identifiers **in body text** | **139 docs carry a docket/registry ID in metadata, but only 33 docket-format tokens appear in body text** — see open question below |

### Findings

**FDA metadata defects, handled explicitly:** the catalogue bottoms out at a
placeholder year of 1900 (4 documents have no usable date); the docket field
contains the literal string `"None found"` (21 documents have no docket); issuing
offices are `<br>`-delimited HTML, and stripping tags before splitting silently
concatenates them into strings like
`"Center for Drug Evaluation and ResearchCenter for Biologics Evaluation and Research"`
(46 documents have multiple offices). All surfaced in the report rather than
quietly patched.

**ClinicalTrials.gov fingerprints the User-Agent.** It returns 403 for any UA
that does not begin with the true client token. Verified:

```
python-httpx/0.28.1                                    200
python-httpx/0.28.1 ragpipe/0.1 (contact@example.com)  200
ragpipe/0.1 (contact@example.com)                      403
curl/8.7.1          (sent from httpx)                  403
Mozilla/5.0                                            403
python-requests/2.32                                   403
```

A conventionally "polite" custom UA is exactly what gets blocked. `net.py`
prefixes the honest client token and appends our identifier; the matrix is in a
comment so the prefix is not later "cleaned up" into an outage that looks like
downtime.

**`pypdf` 6.x renamed its error base class** to `PyPdfError`; `PdfError` no
longer exists.

### Corrections

**The scan-suspect planting logic does not work, and was described as if it did.**
`sample.py` force-included two protocols with declared size ≥ 10 MB, on the
theory that large files are scans, so the text-layer detector and quarantine path
would be exercised by construction. Both planted documents (29.5 MB and 33.0 MB)
classified as `digital_native` — they are large because of embedded figures, not
because they are scanned. The two genuinely image-only documents in the corpus
turned up **unplanted**, in the ordinary stratified sample.

So the detector works, but byte size is a poor scan proxy and the "deliberate
test case" claim was unearned. One planted document sits at 16 blank pages of 93
(17.2%), just under the 20% `mixed` threshold, which shows the threshold is doing
real work and is worth revisiting with evidence.

Action: replace corpus-roulette with a real unit test for `pdfcheck` using a
synthetic no-text PDF, and relabel the sampling reason to describe what it
actually selects (large documents — still useful for extraction stress-testing).
Deferred until the code-review gate reports, to avoid editing files under review.

**Two lint findings were real defects, not style.** A `rng.random()` loop that
claimed to stabilise the seed but only burned entropy (deleted), and unflushed
progress output that made a multi-minute download look hung when piped to a log
(fixed with `flush=True`).

### Verification gate — numerical audit

An independent subagent recomputed every figure in `reports/corpus_report.md`
from `corpus/manifest.jsonl` and `data/fetched.jsonl`, forbidden from importing
`stats.py`. Result: **no figure is wrong.** All 160 manifest and fetched records
matched on `doc_id`, no duplicate or null hashes, and every count, distribution,
and derived estimate reproduced exactly.

It found seven *presentational* defects instead. All seven were independently
reproduced before being accepted — no false positives.

| # | Defect | Reproduced | Fix |
|---|---|---|---|
| 1 | **Size table silently mixes subsets.** Pages and Bytes cover all 160 downloaded docs; Extracted characters covers only the 158 indexable, while the surrounding prose implies an indexable framing throughout | bytes **+5.0%**, pages +1.5% vs indexable-only; the 7.3 MB quarantined scan is counted in bytes but excluded from chars | Compute all distributions over the indexable set; report downloaded footprint as its own explicit line |
| 2 | **Docket row bundles two incompatible identifier types.** `docs_with_docket_id` = 99 FDA dockets + 40 NCT IDs, but the body-text detector regex structurally cannot match an NCT ID, so the adjacent "33 found in text" row is FDA-only | regex fails on `NCT07027878`, matches `FDA-1992-N-0007`; all 33 tokens fall in 28 FDA docs, zero in the 40 protocols | Split into three rows by identifier class; add an NCT pattern to the detector |
| 3 | **Eval-slice table mixes units under one "Count" column.** Rows 1/2/5 count documents; rows 3/4 count occurrences | 676 table-like lines are spread across **60** documents, not 676 | Add a unit column; report document counts alongside occurrence counts |
| 4 | **"unreadable pdfs: 0" reads as "every PDF yielded usable text."** The flag only fires when a PDF fails to *open* | Both quarantined docs opened fine: 180 chars from 68 pages, 709 from 8 | Rename the metric; add a negligible-extracted-text count |
| 5 | **Partial scans badly understated.** Prose says "a protocol with a scanned appendix is `mixed`", but the 20% threshold means most are not | **23** indexable docs contain ≥1 image-only page (108 pages); only **1** is labelled `mixed`, 22 are `digital_native` | Report the ≥1-blank-page count; correct the prose; revisit the threshold with this evidence |
| 6 | ACL partition computed over 160 rather than the 158 that will be indexed | two rows inflated by one each | Compute over indexable |
| 7 | **Two data-quality labels are not enforced by their logic — correct only by luck.** Multi-office count has no source filter; date count checks only the `date_invalid` flag, never a missing date | 46 multi-office docs, 0 non-FDA (so no filter needed *yet*); `date_invalid`=4 coincides exactly with missing=4 | Add the source filter; count missing dates explicitly. Both would drift silently on new data |

Methodology notes accepted without change: the p90 is a truncating order
statistic (understates the 158-row by 1.3%, defensible); chunk estimates treat
the corpus as one continuous token stream and so ignore ~158 partial chunks at
document boundaries (~1–4%); the three chunk rows are exactly 4:2:1 by
construction and therefore carry no information beyond the first.

The audit also independently surfaced the scan-planting failure recorded above,
having reached it from the data rather than from the code.

**Assessment:** the pipeline's arithmetic is sound; its *labelling* was not. Every
defect is a case of a number being computed over one population and presented
next to prose implying another. Worth noting for later phases — this is precisely
the failure mode that would make a retrieval ablation table quietly wrong while
looking entirely plausible.

### Verification gate — code review

A second independent subagent read all of `src/ragpipe/`, exercised the pure
functions with edge-case inputs, and checked whether the stated rationale in each
comment is actually true of the implementation. It reported **3 high, 8 medium,
and 8 latent** findings. Every one was reproduced before being acted on. **No
false positives from either agent.**

The dominant theme was not broken arithmetic — it was *claims the code did not
deliver*. Four separate files asserted sha256 pinning and verification that did
not exist.

#### High

| Finding | Reproduced | Fix |
|---|---|---|
| **`--refetch` never re-downloaded anything.** It put every document in the work list but still passed the stored hash to `download()`, which short-circuits on a match — so the flag re-ran inspection and issued zero requests | stubbed client: 0 requests with the skip enabled, 1 with it disabled | Added `skip_if_present`; `--refetch` now forces the request. Also added the `--reinspect` flag that the old behaviour was accidentally providing |
| **No hash pinning existed, and four places claimed it did.** `SourceDoc` had no `sha256` field; the manifest had no hash column; `download()` computed a digest and returned it but never *compared* it to anything. A changed upstream PDF would be accepted silently and its new hash recorded as truth. Claimed in `README.md` (twice), the generated report prose, the `Makefile`, and `net.py`'s own docstring | manifest key union confirmed to have no hash field | `SourceDoc.sha256` added; first fetch establishes the pin and writes it back to the committed manifest; every later fetch verifies and raises `HashMismatch`, leaving the existing file intact. `--allow-drift` re-pins deliberately. The four claims are now true |
| **The scan-suspect safeguard selected zero scanned documents** — and actively harmed the sample by excluding every ≥10 MB document from the remainder pool | both planted documents classified `digital_native`; all three documents lacking a full text layer were *below* the threshold | Heuristic removed and honestly relabelled; large documents no longer excluded from the remainder. Deterministic coverage moved to `tests/test_pdfcheck.py` |

#### Medium

| Finding | Reproduced | Fix |
|---|---|---|
| **`_round_robin` starved strata alphabetically.** With `target < len(strata)` it broke mid-pass over a *sorted* key list, so the alphabetically last strata always got zero. `--fda-n 40` over 58 strata would yield a CBER/CDER/CDRH-only corpus with nothing from Human Foods Program or ORA | 20 strata, target 12 → strata 12–19 received nothing | Keys shuffled with the seeded RNG; start point rotates each round |
| **Pair budget was double its documented size, and samplers overshot `target`.** The budget counted *titles* while the comment claimed a document share (~17% vs "~8%"); a `max(4, …)` floor meant `--fda-n 4` returned 8 documents | `sample_fda(pool, 0\|1\|3\|5)` all returned 8 | Budget expressed in documents and bounded by `target`; both samplers now return `<= target` |
| **Docket regex missed line-wrapped and en-dash identifiers**, understating the metric that decides whether the exact-identifier slice is viable | 3 of 3 sampled documents contained `'FDA-1996-\nD-0012'`-style wrapping invisible to the strict pattern | Pattern accepts Unicode dashes and absorbs line breaks; separate `REGISTRY_TOKEN_RE` added for NCT IDs |
| **One bad PDF or a Ctrl-C discarded the whole run.** `pdfcheck.inspect` sat outside the `try`, results were written only after the loop, and `pypdf`'s base class does not cover every failure mode — so an escape at document 150 lost all 150 records. Combined with the missing pins, the next run would then re-download everything | code path confirmed; `pypdf` raises `DependencyError`/`RecursionError` outside `PyPdfError` | Inspection moved inside the guard, broad exception catch, checkpoint every 10 documents, and a `finally` that always persists |
| **`_split_offices` shredded office names containing commas.** `fda-75334` produced three fake offices from "Office of Policy, Legislation, and International Affairs", also inflating the multi-office data-quality count | exact fragmentation reproduced from the committed manifest | Separator is now `,(?!\s)`. Verified against the live index: offices are separated by comma-**without**-space, while names contain comma-**with**-space |
| `_dist` p90 fell below the median for small n — at n=2 it returned the *minimum* | `_dist([1, 500])` → p90 = 1 | Nearest-rank p90 |
| `Retry-After` branch slept after the final attempt and ignored the RFC 9110 date form | 5 sleeps for 5 attempts vs 4 on the backoff path; date form fell through to hammering the server | Sleep guarded on a further attempt existing; both header forms parsed |
| **`download()` had no retry at all** — it bypassed `get()`, so the ~160 PDF fetches (essentially all the traffic) got one attempt each, and a single transient 503 failed a document permanently. `.part` files leaked on exception | code path confirmed | Retry/backoff loop added around the streaming download; `.part` cleaned up in `finally` |

#### Latent, all fixed

Dead `decrypt()` branch that could never fire; `unreadable_reason` silently
dropped instead of recorded; `_era` raising `ValueError` on a non-numeric date;
`render_markdown` crashing on a `None` table count; a docstring example (30 blank
pages of 200 = 15%) that would classify as `digital_native`, not `mixed`, under
its own threshold; an off-by-one between a comment saying "at or before" and code
using `<`; `_split_list` not applying the `<br>` rule the module docstring
mandates; `is_indexable` defined but never called while two call sites duplicated
its literal.

#### Determinism

The core claim held — byte-identical selections across runs, no dict or set
iteration order leaking into results. One real weakness: both samplers shared a
single `Random`, so the protocol sample depended on how much entropy the FDA stage
happened to consume. Holding the seed and changing only `--fda-n` changed the
protocol sample. Each sampler now seeds its own RNG from `seed` plus a per-source
salt, and a regression test pins the independence.

### Response: 51 regression tests

`tests/` now has one test per defect, each documenting the failure it prevents.
The threshold logic was extracted into `pdfcheck.classify_pages()` so it is
testable without constructing PDFs — previously it was exercised only by whatever
happened to be in the corpus, which is how the scan-detector claim went unchecked.

`make test` runs them; `make lint` covers `src` and `tests`.

### Post-fix corpus (the shipped numbers)

The sampling fixes changed which documents are selected, so these supersede the
pre-review figures above rather than being comparable to them.

| Metric | Value |
|---|---|
| Downloaded | 160 / 160, zero failures |
| **sha256 pinned in committed manifest** | **160 / 160** |
| On-disk hash matches pin | 160 / 160 (verified independently) |
| Indexable | 155 (96.9%) |
| Text layer | 154 digital_native, 1 mixed, 5 image_only |
| Docs with ≥1 image-only page | 18 (40 pages) — only 1 clears the `mixed` bar |
| Estimated tokens | ~2.99 M |
| Estimated chunks | ~6,880 @ 512 tokens |

Integrity behaviour verified end to end:

- Re-running `fetch` with all pins satisfied issues **zero** downloads.
- Deliberately corrupting one pin produced `HASH MISMATCH`, marked that document
  failed, left the existing file untouched, and let the other 159 proceed.

### Eval-slice viability, post-fix

| Slice | Unit | Count |
|---|---|---|
| Version currency | documents | 16 (8 complete pairs) |
| Exact identifier — FDA, in metadata | documents | 93 |
| Exact identifier — FDA, in body text | documents | 26 (41 occurrences) |
| Exact identifier — protocols, in body text | documents | **20 of 40** |
| Table lookup | documents | 64 (528 table-like lines) |
| Temporal | documents | 3 undated |

The protocol row is new information, and it is the most consequential change: it
previously read as zero, because the FDA docket pattern structurally cannot match
an NCT ID. Half the protocols do carry their registry ID in body text. Combined
with the 26 FDA documents, **46 documents have a retrievable exact identifier**
rather than the 28 reported before the fix.

That does not close the open question — roughly 70% of documents still carry an
identifier only in metadata — but it makes the exact-identifier slice considerably
more viable than the first report implied, and it is a direct consequence of a
review finding rather than of new data.

### Open questions carried into Phase 1

1. **The exact-identifier slice needs a different probe than dockets.** Only 33
   docket-format tokens appear in body text across the corpus, so BM25 over chunk
   text will not find a document by its docket. Either the identifier has to be
   indexed from metadata into searchable text, or the slice should be built on
   identifiers that *are* abundant in body text — CFR citations (`21 CFR 314.50`),
   ICH codes (`Q3C`, `E6(R2)`), statute sections, drug names. Needs measuring in
   Phase 1 once text is extracted and persisted.
2. **`MIN_CHARS_PER_TEXT_PAGE = 100` and the 20% / 80% thresholds are guesses.**
   Revisit against the extracted text once it exists.
3. **Table detection is a whitespace-column proxy**, adequate for ranking
   documents to hand-inspect, not for a metric. The 676 figure should not be
   quoted as a table count.
