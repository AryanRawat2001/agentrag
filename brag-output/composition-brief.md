# Hyperframes Composition Brief: agentrag

## Objective
Create a short, restrained launch video for **agentrag** — a retrieval system that locates
every quote a model produces in the source document before the reader sees it.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/agentrag.mp4`
- Format: landscape — 1920x1080
- Duration: 38.0 seconds

## Source Material
- Project root: `/Users/aryanrawat2001/RAG Pipeline with Hybrid Search`
- Primary files read: `src/ragpipe/dashboard.py` (the UI: markup, `:root` palette, copy),
  `README.md`, `docs/architecture.svg`, `reports/failure_modes.json`
- Product name: **agentrag**
- Tagline / strongest claim: *"every quote the AI uses is checked against the original
  document before you see it. If a quote cannot be found, this page says so instead of
  hiding it."*
- Key UI moment to recreate: **the citation card** — coloured left border, mono verdict
  chip, and two labelled panes (`THE MODEL'S QUOTE` / `WHAT THE DOCUMENT SAYS`).
- Copy that must appear verbatim:
  - `An AI answer you can check.`
  - `How should a 904(a)(3) report be submitted to FDA?`
  - `13,423 passages · searched by keyword (BM25) · under 0.2 ms`
  - `THE MODEL'S QUOTE` / `WHAT THE DOCUMENT SAYS`
  - `✓ normalized` / `✗ unverified`
  - `not found in the passage the AI cited`
  - `What are the requirements of 7 CFR 1609.61?`
  - `"The provided documents do not address the requirements of 7 CFR 1609.61."`
  - `A refusal is a feature here.`
  - `240 probe queries · ground truth is character spans, not chunk ids`
  - `Across 192 configurations, keyword search beats dense retrieval by 43–75× on exact
    identifiers.`
  - `every quote, located in the source`

### The quoted text is real — and each card's provenance differs
**Do not invent quote text, identifiers or verdicts.** Both verdicts on screen are output
from this repo's own `ragpipe.citations.verify_citation`, re-run to confirm, not written
from memory.

- **Card 1** is the real 2026-09-20 run: the quote as served, chunk
  `fda-83375::structural::00038`, span `chars 443–578`. Its verdict is **`normalized`**,
  not `exact`. An earlier draft said `exact` from memory of a screenshot; re-checking the
  stored text through the verifier returned `normalized`, so that is what the card says.
- **Scene 4's refusal** is a real recorded outcome from `reports/generation_eval.json` —
  query, answer text and `refusal_source: model` as logged, from the `unanswerable` slice.
  The tag line is a verbatim clause from the dashboard's own refusal banner.
- **Card 2** is a real verifier result over a *constructed* attribution, and this is a
  deliberate compromise. The committed artifacts contain **zero** recorded `unverified`
  citations (`generation_eval.json`: 22 claimed, 22 located), and the one produced live on
  2026-09-20 was not saved. So the quote is a real sentence from a real ClinicalTrials.gov
  protocol (`ctgov-NCT00567567`) attributed to this FDA guidance chunk — exactly the
  "real sentence, wrong passage" case the verifier exists to catch. `verify_citation`
  returns `unverified` with no span, and the card renders precisely what the dashboard
  renders in that state, down to the legend.
- An earlier draft of card 2 cited `fda-83375::structural::00041` — a chunk that **does not
  exist**, the document ends at `::00040` — with a quote no model ever produced. It was
  caught by checking against the corpus rather than by reading it back.

## Creative Direction
- Tone preset: `polished`
- Creative direction: quiet, evidence-first product film — a lab instrument, not a launch.
- Interpretation: four scenes, long holds, soft crossfades (0.7–0.8s), mixed-case
  light-weight display type over mono detail. No zooms, no count-ups, no risers. The only
  moment permitted any weight is the red `unverified` card arriving.
- Angle: every RAG demo shows you an answer; this one shows you the receipt, *and* shows
  you the one that failed. A product that publishes its own misses is the entire claim.
- Hook: `An AI answer you can check.` at full scale on near-black, wordmark beneath.
- Outro / punchline: the evaluation finding, then the wordmark, then silence.
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals
  - Any implication that this is a live screen capture — it is a designed recreation of a
    real result, and the real walkthrough is `docs/demo.mp4`

## Visual Identity
Verbatim from `src/ragpipe/dashboard.py` `:root`:
- Background: `#0f1117` · panel `#171a23` · raised `#1d212c` · border `#262b38`
- Text: `#e6e8ee` · dim `#9aa3b8`
- Accent: `#58a6ff` · verified `#3fb950` · failed `#f85149` · caution `#d29922`
- Display font: system UI stack (no webfont — the interface ships none)
- Body font: `ui-monospace, SFMono-Regular, Menlo, Consolas` — identifiers and verdicts are
  always mono in this product, and that is a real characteristic of it
- Visual references: the citation card's coloured left border; the mono verdict chip; the
  two-column pane layout; the rounded `#262b38` search box

## Storyboard
Use `brag-output/brag-plan.md` as the creative contract.

1. **The claim** — 4.0s — the hook line, then the wordmark.
2. **Ask** — 5.0s — the real query types into the real search box; the mono meta line lands.
3. **The proof** — 8.0s — card 1 (`✓ normalized`, panes identical), then card 2
   (`✗ unverified`). Both hold together. **This is the centerpiece.**
4. **The refusal** — 6.5s — a real recorded refusal, and why that is the right outcome.
5. **The harness** — 7.5s — the evaluation harness was built first; span-grounded queries.
6. **The finding, and out** — 7.0s — the 43–75× result, then the wordmark and the claim.

Scenes 4 and 5 were added after the first cut: at 22s the clip was leaving out the two
strongest things the project has — that it refuses, and that the eval harness came first.
`/brag`'s creative law caps at 25s "without a reason"; this is the reason, recorded here.

## Audio
- Audio role: sparse professional accents over a low bed.
- Audio arc: bed enters under the claim, carries the query and the proof, fades to silence
  under the wordmark.
- Music: `assets/music/bed.mp3` (Happy Beats / Business Moves vol 9, 114.84 BPM).
- Music treatment: fade in 0→0.8s to **0.16** static level, hold, full fade 19.5→22.0s.
  Volume via the `data-automation` volume lane, never a timeline tween.
- Music cue guidance: bundled preset read. Strong cues targeted — **6.34s** (meta line),
  **10.54s** (card 1), **12.65s** (card 2). Beat grid is 0.53s; accents may sit on it,
  readable text may not.
- Audio-reactive treatment: **none.** Reactive glow reads as decoration on an instrument.
- Audio-coupled moments:
  - typed query — three sparse key ticks, not one per glyph
  - card 1 arrival — one soft card slide
  - card 2 arrival — one soft card slide plus a single dry accent on the `unverified` chip
  - outro — nothing
- SFX selection guidance: prefer the softer `card-slide` family over `card-place`, and
  low-frequency-risk impacts; everything sits under the bed.
- Exact SFX choice: `assets/sfx/card-1.ogg`, `card-2.ogg`, `accent.ogg`,
  `key-1.wav`/`key-2.wav`/`key-3.wav`, all copied into the composition.
- Restraint rule: **four SFX moments in the whole video.** If a cue is not matching a
  visible movement, it does not exist. No whoosh, no riser, no impact on the logo.

## Legibility (the first cut failed this)

Type is sized for a phone feed, not for the authoring screen. The first pass used 17px
pane labels and 22px quote text, which is about **7px** once LinkedIn renders 1080p into a
~420px mobile column — verified by downscaling a frame to 420px and looking at it, not by
assuming. Nothing a viewer must read is now below ~22px, and the lines they must read are
31px and up. Re-run that check after any copy change: `ffmpeg -i agentrag.jpg -vf scale=420:-1`.

## Figure discipline (hard constraint, not a preference)
Every number on screen exists in a committed artifact of this repo.

- **Never the bare `75×`.** That is the `fixed` chunking's identifier margin; the service
  serves `structural` at 43×. Only the range `43–75×` may appear.
- **No generation latency.** 12.5s is a median of a 2.2–27.1s distribution observed as wide
  as 39.3s. Omitted entirely.
- Retrieval appears as a **bound** (`under 0.2 ms`), never a decimal.
- No dollar figures. Cost is reported in tokens; no confirmed paid rate exists.
