# brag-output

A 38-second launch clip for **agentrag**, generated with
[`/brag`](https://github.com/latent-spaces/brag) (`--tone polished`) and rendered locally
by Hyperframes.

| File | What it is |
|---|---|
| `agentrag.mp4` | 1920x1080, 38.0s, H.264 + AAC. Poster baked as frame 0. |
| `agentrag-preview.gif` | 7s loop of the citation-card reveal, cropped to the cards. **This is what the repo README embeds** — GitHub strips `<video>` for repo-hosted files (verified against its markdown API), so a GIF is the only thing that plays inline. |
| `agentrag.jpg` | Poster still — the two-card frame at 14.5s. Baked as frame 0 of the mp4, and the custom-thumbnail upload for platforms that accept one. |
| `share-copy.txt` | The caption. |
| `brag-plan.md` | Creative plan and storyboard. |
| `composition-brief.md` | The brief handed to Hyperframes, including the provenance of every figure and quote on screen. |
| `composition/` | The Hyperframes project. **Gitignored** — it carries bundled third-party audio. |

## This is the hook, not the demo

`docs/demo.mp4` is the real artifact: a 3-minute walkthrough whose frames are a live run of
the service. This clip is a **designed recreation** of a real result — it exists to be
postable on LinkedIn and X, where three minutes is too long, and to link to the real one.
Do not present it as a screen capture.

## Everything on screen is checked

Both citation verdicts are output from this repo's own
`ragpipe.citations.verify_citation`, re-run against the corpus rather than written from
memory. Card 1 is the real 2026-09-20 run. Card 2 is a real verifier result over a
constructed attribution, because the committed artifacts contain **zero** recorded
`unverified` citations — `composition-brief.md` explains exactly why and how.

Every figure exists in a committed artifact. The margin is always the range **43–75×**,
never the bare 75× (that is the `fixed` chunking; the service serves `structural` at 43×).
Generation latency is omitted entirely — 12.5s is a median of a 2.2–27.1s distribution.

## Credits

Music: [ende.app](https://ende.app/en) "Happy Beats / Business Moves" · SFX:
[Kenney](https://kenney.nl/) · Both bundled with the `/brag` skill and licensed for use in
the rendered video. Rendering: [Hyperframes](https://hyperframes.heygen.com/), run locally
with telemetry disabled.
