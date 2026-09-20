# Demo video — shot list and script

**Target: 3 minutes.** A recruiter or engineer watches the first 30 seconds and decides
whether to keep watching, so the verification diff — the one thing here that a stock RAG
demo cannot show — has to be on screen inside the first minute.

**There is now a build script: `bin/build-demo`.** It renders the frames from the live
service, narrates with macOS `say`, generates the subtitles from the same text, and
composites with ffmpeg — so `docs/demo.mp4` and `docs/demo.srt` are reproducible rather
than recorded. The synthetic voice is a **placeholder**; `--no-voice` gives a silent cut
with burned-in subtitles as a base for recording your own audio over.

It is blocked on nothing but a fresh free-tier quota (20 requests/day), and it probes
generation before rendering so an exhausted day cannot produce a video whose narration and
frames disagree. That probe now also stops the build when the day is spent, when a
per-minute rate limit is mistaken for one, when the model answers without citing, and when
not one quote could be located — each of which would have produced a frame contradicting
its own narration.

**The build script's narration is generated, not transcribed.** The corpus size comes from
the live `/health` and the identifier margin from `reports/failure_modes.json`, keyed by
the chunking the service reports it is actually serving. Two figures were wrong while they
were literals: "a hundred and sixty documents" (152 are indexable) and a bare "seventy
five" (that is `fixed`; the served `structural` chunking is 43×). So the shot list below is
the manual alternative and reads very close to the audio, but the spoken figures come from
the machine.

The spoken text lives in `narrate.py` inside `bin/build-demo` — edit it there, not here,
or the subtitles and the voice will drift apart.

**Captions have their own band.** Every frame is scaled into the top 624 px and the bottom
96 px is reserved, so a caption can never cover the picture. It used to: at 720p the
subtitles sat across the architecture diagram's EVALUATE lane, hiding `192 configurations`,
`the finding` and the `43–75×` result — the most important number in the video, obscured in
the frame whose job is to show it. The cause was that ffmpeg renders an SRT through a
384×288 ASS canvas, so `MarginV=42` meant 105 real pixels; the build now pins the canvas to
the video's size and fails rather than guessing if it cannot.

**Scene 3 is cropped, not scrolled.** Headless Chrome's `--screenshot` renders from the
document origin and ignores the scroll that the `#citations` deep link performs, so that
frame came out 87% empty with the verification panes just entering at the bottom edge. The
page now publishes the card's document offset on `body[data-cite-top]` and the build crops
to it — screenshot and DOM dump from a single page load, so it still costs one generation
request. If you restyle the citations card, that attribute is load-bearing.

**Recording it yourself is still worth considering.** I can build and verify the flow but not capture a screen or
a voice. Everything below has been run end to end against the live container, and every
figure quoted is in a committed artifact.

---

## Before you record

```bash
make chunk              # only if data/chunks/structural.jsonl is missing
export GEMINI_API_KEY="$(sed -n 's/^GEMINI_API_KEY=//p' .env.gemini)"
agentrag start          # colima + compose + health wait, ~40s
agentrag status         # confirm: generator available, 13,423 passages
```

Then **prime the daily quota check**: the free tier allows 20 generation requests per day,
and the demo needs 2–3. Run `agentrag stats` afterwards to confirm you have headroom.

Browser setup: one tab at `http://localhost:8000`, zoom ~125% so the citation panes are
legible at 1080p; a **second tab at `docs/architecture.svg`** for Shot 5 (open it with
`open docs/architecture.svg` — it is 1280×720, so it fills the frame without cropping);
and a terminal window sized to show ~15 lines.

**Two things about this tier that will bite a take, both measured rather than guessed:**

**Latency is a lottery.** `reports/serving_bench.md` records 2.2–27.1 s over 16 calls; a
verification run for this script observed **2.1 s, 7.2 s, 18.3 s and 39.3 s** — so the real
spread is wider than the artifact captures, and 39 s of dead air will ruin a take. Have the
retrieval-only query ready to cut to.

**Answerable queries refuse, and that is expected.** Of the three answerable example chips,
a live run had **one answer and two refuse**. That is failure mode 3 in
`reports/failure_modes.md`, not a bug: the model is instructed to answer only from
retrieved text, and it is conservative. Use the **904(a)(3) query** as the "it answers"
shot — it is the one that reliably produces citations — and treat any other chip as a
bonus if it lands.

---

## Shot 1 — the claim (0:00–0:20)

*Screen: the dashboard, empty state visible.*

> "This answers questions over 152 FDA guidance documents and clinical trial protocols.
> What makes it different from every other RAG demo is the second half: every quote the
> model produces gets **located in the source document** before you see it. If it can't be
> found, this page says so instead of hiding it."

Point at the "What happens when you ask" card — search, answer, verify — and note that
step 3 is the part most systems skip.

---

## Shot 2 — a verified answer, and the diff (0:20–1:10)

*Click the **"How do I submit a 904(a)(3) report?"** chip.* Verified working: it returns
5 citations, 4 of which verify, and — see Shot 3 — one that does not.

While it runs (a few seconds), say what is happening:

> "Keyword search over 13,423 passages first — that takes a fraction of a millisecond.
> Then the top 5 passages, and nothing else, go to the model, which is required to quote
> them."

If it refuses instead, say so and move on — "the model is conservative, and I'd rather it
refuse than guess" is a better line than a retake. Then click it again; it answers most of
the time.

When it lands, go straight to the **Citations** card:

> "Here's the model's quote on the left, and what the document actually says on the right.
> Both panes, always — showing the source only on a mismatch would make 'verified' the one
> state where you can't check the work."

Scroll to the retrieval table:

> "And this is what the search returned — five passages at the default setting, with a
> badge marking the ones the answer actually quoted. You can expand any of them and read
> the text, so the ranking is something you can judge rather than take on trust."

---

## Shot 3 — the failure, not the success (1:10–1:50)

This is the shot that distinguishes the project. **Do not skip it.**

*Click "Ask something the documents don't cover" — the helium question.*

> "A refusal is a feature here. The model was told to answer only from the retrieved text,
> and it reported that it couldn't. The alternative is a confident answer with no support."

Then the honest part — and you do not need to reach for an anecdote, because the
904(a)(3) query from Shot 2 **produces one on screen**. A verification run for this script
returned `4 of 5 located`: four `normalized` and one **`unverified`**.

*Scroll back to the citation panel with the red left border.*

> "Four of those five quotes were found in the source. This one wasn't — so it's marked
> unverified, with the source pane saying 'not found in the passage the AI cited'. Most
> systems would have shown you all five as citations and you'd never know."

Worth naming the middle verdict too, if it appears:

> "`normalized` means it matched after collapsing whitespace — which is a *real* match in
> PDF text, where line breaks land mid-sentence. There's a third tier,
> `line_number_ambiguous`, that matches only after stripping the legislative line numbers
> FDA drafts interleave — and that one is deliberately **not** counted as verified,
> because I caught it scoring a model's '40 CFR' as a citation of a document's '21 CFR'."

---

## Shot 4 — the measurement (1:50–2:35)

*Cut to the terminal.*

```bash
agentrag bench          # per-stage latency, no quota spent
```

> "Retrieval is under two tenths of a millisecond. Verification is six microseconds.
> Generation is 12.5 seconds — five orders of magnitude more than everything I control. That's why the stages are timed
> separately: a single end-to-end number would only ever move when somebody else's API
> moved."

Then open `reports/retrieval_eval.md` (or show the README table):

> "This is the finding I didn't expect. On exact identifiers — CFR citations, NCT numbers —
> dense embedding retrieval loses to keyword search by a factor of **43 to 75**,
> depending on how the documents are split — 43× on the `structural` chunking this
> service actually serves. And naive reciprocal-rank fusion is *worse than keyword search
> alone*, because fusion rewards agreement, so a retriever with no signal doesn't abstain.
> It votes, and it outvotes a correct top hit."

> "So the shipped service uses BM25. Not as a simplification for the demo — that's what the
> table supports."

---

## Shot 5 — the architecture (2:35–3:00)

*Open `docs/architecture.svg` full-screen in a browser.* It is 1280×720 and dark-themed to
match the dashboard, so it fills a 1080p frame cleanly with no cropping.

Read it top to bottom — the three lanes are the whole system, and the middle one is the
part that matters:

> "Three stages. **Build** is offline and runs once: 160 sha256-pinned PDFs, 152 of them
> with a usable text layer, become 13,423 passages and a keyword index."

> "**Serve** is per request. Retrieval in 0.17 milliseconds. The model gets five passages
> and nothing else, and is required to quote them — that's the slow part, 12.5 seconds of
> somebody else's API. Then verification, in six microseconds."

*Point at the dashed arrow feeding VERIFY from `retrieve`, not from `generate`.*

> "And this arrow is the design decision. The verifier checks the model's quotes against
> **the retrieved passages** — not against the model's own claim about where its quote came
> from. It doesn't trust returned offsets. It searches. That's what makes it work with any
> provider, and it's what catches a real sentence attributed to the wrong passage."

*Then drop to the bottom lane.*

> "And this is the part I'd argue is the actual engineering. The evaluation harness was
> built **first** — 240 probe queries whose ground truth is character spans rather than
> chunk ids, so the same queries stay valid when I re-chunk and every comparison is a
> measurement instead of a circular one."

> "192 configurations later: dense embedding retrieval loses to keyword search by **43–75×**
> on exact identifiers. So BM25 alone ships. Not a simplification for the demo — that's
> what the table supports."

Close on the box in the top-right, because saying no to something is the more interesting
claim:

> "Qdrant is in the stack and the API doesn't use it. Measured at roughly seven times
> slower than the in-process matrix product it would replace, at this corpus size. It's
> there because containerising the serving path was a deliverable — but as a measured
> trade, not an assumed win."

**Note on the `agentrag bench` shot.** If you keep Shot 4's terminal run, this lane's
latency figures are the same numbers, so don't read them twice — point at the diagram and
say "these are the p50s from that run".

## Cut these if you run long

In priority order — the citation diff and the identifier finding are the two that cannot go:

1. **Shot 4's `agentrag bench` terminal run.** The architecture diagram carries the same
   per-stage numbers, so this is the cheapest thing to drop.
2. Shot 2's retrieval table.
3. Shot 5's closing Qdrant point — the "we measured it and said no" line is good, but the
   three lanes are the substance.

## Do not say on camera

- **No dollar figures.** Cost is reported in tokens; the free tier bills nothing and no
  confirmed paid rate was obtained. A currency number would be invented.
- **Never quote the 75× on its own.** It is the `fixed` chunking's identifier margin.
  The service serves `structural`, which is 43×. Quoting the best configuration's number
  while demonstrating a different one is precisely what the evaluation harness exists to
  catch, and both the narration and the architecture diagram were doing it. Say "43 to
  75, depending on how the documents are split".
- **No single latency figure for generation** without its range. 12.5 s is a median of a
  distribution measured at 2.2–27.1 s and *observed* as wide as 39.3 s; a single
  observation is an anecdote.
- **Nothing about `wrong_chunk` rates.** It is zero in every artifact and has been seen
  live once. Describe it as something the verifier caught, not as a measured frequency. The
  same applies to the `unverified` citation in Shot 3 — it is a live demonstration, not a
  published rate.
- Don't claim the golden answer set is fully curated in the sense of "every answer
  verified" — every flagged pair was read and judged, 8 rejected; the rest are
  machine-validated and spot-checked.

## Afterwards

```bash
agentrag stop --all     # containers and the Docker VM
```
