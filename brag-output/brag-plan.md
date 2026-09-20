# Brag Plan: agentrag

## What is this app?
A retrieval system over 152 public FDA and ClinicalTrials.gov documents that answers
questions **and then locates every quote the model produced in the source document** before
the reader sees it — showing the model's quote and the document's text side by side, and
marking anything it cannot find.

## The angle
Every RAG demo shows you an answer. This one shows you the receipt — and shows you the one
that failed. The video's whole job is the two-pane diff and the red `unverified` badge next
to it. A product that publishes its own misses is the claim; nothing else here needs saying.

Specific to this project and to no other: the quoted text on screen is from a real run of
this service on 2026-09-20, including the `th e eSubmitter` mid-token split that PDF
extraction actually produced.

## Hook (first 2-3 seconds)
One line, full scale, on near-black: **"An AI answer you can check."** Then the wordmark
`agentrag` in mono underneath, dim. No motion flourish — the restraint is the hook.

## Key moments (the middle)
- The real query typed into the real search box: *"How should a 904(a)(3) report be
  submitted to FDA?"*, with `13,423 passages · under 0.2 ms` ticking in beneath it.
- The citation card: `✓ normalized`, the model's quote and the document's text side by side,
  identical. The proof, not a description of the proof.
- A second card arrives: `✗ unverified` — source pane reads *"not found in the passage the
  AI cited."* The system reporting its own miss, on camera.
- A real recorded refusal, and the dashboard's own line about it: *"A refusal is a feature
  here."* Not a staged failure — it is in `reports/generation_eval.json`.
- The evaluation harness was built **first**, on 240 span-grounded probe queries.

## Outro / punchline
The finding the harness produced, then silence: **"Across 192 configurations, keyword
search beats dense retrieval by 43–75× on exact identifiers."** Then the wordmark and the
one-line claim.

## User flow worth showing
Entry → key action → result, and all three are real:
1. **Ask** — a regulatory question typed into the box.
2. **Retrieve + answer** — 13,423 passages searched; the model must quote what it is given.
3. **Verify** — every quote located in the source, each labelled `exact` / `normalized` /
   `unverified`. The result screen *is* the product.

## Tone
- Preset: `polished`
- Creative direction: quiet, evidence-first product film — a lab instrument, not a launch.
- Interpretation: four scenes, long holds, soft crossfades, mixed-case light type. No
  exclamation, no zooms, no count-up theatrics. The one moment allowed to feel like an
  event is the red `unverified` card arriving.

## Format: landscape — 1920x1080
## Duration: 38.0s

## Visual identity (from the project)
Taken verbatim from `src/ragpipe/dashboard.py` `:root`:
- Background: `#0f1117` (panels `#171a23`, raised `#1d212c`, borders `#262b38`)
- Accent: `#58a6ff` (blue) · verified `#3fb950` · failed `#f85149` · caution `#d29922`
- Text: `#e6e8ee`, dim `#9aa3b8`
- Display font: system UI stack (`-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica`)
- Body/mono font: `ui-monospace, SFMono-Regular, Menlo, Consolas` — the interface is
  mono-heavy by design; identifiers and verdicts are always mono.
- Strongest visual element: the citation card — a coloured left border (green/red), a mono
  verdict chip, and two text panes labelled `THE MODEL'S QUOTE` and `WHAT THE DOCUMENT SAYS`.

## Share copy (draft)
Most RAG demos show you an answer. I built one that shows you where every quote came from —
and flags the ones it couldn't find. 152 FDA documents, 13,423 passages, every citation
located in the source before you see it.

## Audio direction
- Role: sparse professional accents over a low bed.
- Music: `happy-beats-business-moves-vol-9-by-ende-dot-app.mp3` (114.84 BPM), held well
  under the visuals and faded out under the final wordmark.
- Music treatment: start at 0, low gain throughout, no swell, full fade across the last 2s.
  The track is upbeat; `polished` needs it as a floor, not a driver.
- Music cue guidance: preset cue file read. Target strong cues at **4.23s** (query settles),
  **10.54s** (first citation card), **12.65s** (the `unverified` card). Sequential reveals
  sit on the beat grid at 0.53s spacing — accents only; every text line holds past its
  reading floor regardless of where the beat falls.
- Audio-reactive treatment: none. Reactive glow would read as decoration on an instrument.
- SFX posture: sparse, motion-matched. Roughly four cues in the whole video.
- Audio-coupled moments: key ticks under the typed query; one card-place per citation card;
  a single dry accent on the `unverified` badge; nothing on the outro.
- Restraint rule: no whooshes, no risers, no impact on the logo. If a cue is not matching a
  visible movement, it does not exist.

## Storyboard

### Scene 1 — The claim — 4.0s
Near-black `#0f1117`. Centred: **"An AI answer you can check."** in light-weight mixed case,
large. Beat, then `agentrag` in mono `#9aa3b8` below it, small, letter-spaced.
Sequential/interaction: two elements, headline then wordmark, ~0.6s apart. Headline holds
≥1.8s settled.
Audio intent: the bed enters quietly; nothing punctuates the line.
Audio-coupled idea: none.
Music: low bed from 0.
Transition mood: soft crossfade (0.7s) → Scene 2

### Scene 2 — Ask — 5.0s
The search box from the real interface on the dark ground: rounded panel, `#262b38` border,
the placeholder replaced by the query typing in —
*"How should a 904(a)(3) report be submitted to FDA?"* — then the mono meta line beneath:
`13,423 passages · searched by keyword (BM25) · under 0.2 ms`.
Sequential/interaction: **yes** — the query types character by character (~1.6s), then the
meta line fades in on the 4.23s strong cue and holds ≥1.2s.
Audio intent: quiet competence; the machine is working, not straining.
Audio-coupled idea: subtle key ticks under the typing, low and sparse — not one per glyph.
Music: bed continues.
Transition mood: soft crossfade (0.7s) → Scene 3

### Scene 3 — The proof — 8.0s
**The centerpiece.** The citation card, recreated from real verifier output.
Card 1 arrives: green left border, mono chip `✓ normalized`, identifier
`fda-83375::structural::00038  chars 443–578`. Below it two panes, labelled
`THE MODEL'S QUOTE` and `WHAT THE DOCUMENT SAYS`, both containing the same real sentence.
Card 2 then arrives beneath: red left border, chip `✗ unverified`, `chars not located`, and
the right pane reading *"— not found in the passage the AI cited —"*.
Sequential/interaction: **yes** — card 1 on the 10.54s cue, card 2 on the 12.65s cue. Each
chip is the required reading (≥0.8s settled); the quote text is texture, truncated with an
ellipsis so it stays legible on a phone. Both cards hold together to 17.0s — that pair is
the image the whole video exists to deliver.
Audio intent: the one moment with weight. The red card should land.
Audio-coupled idea: a soft card slide as each card arrives; one dry accent on `unverified`.
Music: bed unchanged — do not swell into the red card. Restraint makes it worse, correctly.
Transition mood: soft crossfade (0.8s) → Scene 4

### Scene 4 — The refusal — 6.5s
A real recorded outcome, not a staged one: query and answer come from
`reports/generation_eval.json`, `unanswerable` slice, `refusal_source: model`.
Small mono query at the top: `What are the requirements of 7 CFR 1609.61?` Then the answer
at display size: *"The provided documents do not address the requirements of 7 CFR
1609.61."* Then, in amber, a verbatim clause from the dashboard's own refusal banner:
**"A refusal is a feature here."**
Sequential/interaction: query at 17.3s (texture), answer on the 17.91s beat holding **4.4s
settled** (12 words at the 0.3s/word floor is 3.6s), tag line at 20.54s.
Audio intent: a single dry mark on the answer. Nothing triumphant — it is a non-event, and
that is the point.
Audio-coupled idea: one low accent as the answer lands.
Transition mood: soft crossfade (0.8s) → Scene 5

### Scene 5 — The harness, built first — 7.5s
Headline: **"The evaluation harness was built first."** Then two mono lines, dim:
`240 probe queries · ground truth is character spans, not chunk ids` and
`so the same queries stay valid when the documents are re-split`.
Sequential/interaction: yes — headline at 24.22s, line A at 26.31s, line B at 28.4s, each
held past its reading floor and **left on screen** so the three read as one statement
rather than replacing each other.
Audio intent: steady. No accent — this scene is an argument, not a reveal.
Audio-coupled idea: none.
Transition mood: soft crossfade (0.8s) → Scene 6

### Scene 6 — The finding, and out — 7.0s
Dark ground. The line, centred, light-weight: **"Across 192 configurations, keyword search
beats dense retrieval by 43–75× on exact identifiers."** Hold. It clears, and `agentrag`
arrives at full scale with *every quote, located in the source* beneath.
Sequential/interaction: finding lands at 31.54s and holds **3.2s settled**; the wordmark
takes the last ~1.7s. Scene runs 31.0–38.0s.
Audio intent: settle and stop.
Audio-coupled idea: none. The outro carries no SFX by design.
Music: full fade 35.5–38.0s, ending in silence under the wordmark.
Transition mood: end.

### Why this is 38s and not 22s
`/brag`'s creative law is 15–25s, "not one second more without a reason." The reason: at
22s the clip omitted the two strongest things the project has — that it **refuses**, and
that the **evaluation harness was built first** and decided the design. Those are the
differentiators for a hiring audience; the cut without them was a product tour.

**Music mood for this video:** restrained upbeat bed, used as a floor.
**Audio summary:** a quiet bed enters on the claim, carries the query and the proof with
four motion-matched accents, and fades to silence under the wordmark — the only emphasis in
the whole track is a single dry hit on the citation that failed.

## Figure discipline (non-negotiable, carried from docs/demo-script.md)

Every number on screen must exist in a committed artifact, and two are traps:

- **Never the bare 75×.** That is the `fixed` chunking's identifier margin; this service
  serves `structural` at 43×. The range `43–75×` with "depending on how the documents are
  split" is the only honest form. Quoting the best configuration's number while showing a
  different one is precisely what this project's evaluation harness exists to catch.
- **No generation latency.** 12.5s is a median of a distribution measured at 2.2–27.1s and
  observed at 39.3s; a single figure without its range is an anecdote. Omit it entirely.
- Retrieval is published as a **bound** (`under 0.2 ms`), not a decimal — the stored samples
  do not exist to reproduce a three-decimal figure.
- No dollar figures anywhere. Cost is reported in tokens; no confirmed paid rate exists.
