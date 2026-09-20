"""A single-page query dashboard, served from the API itself.

The user-facing name is **agentrag** (what you type to run it); `ragpipe` remains the
Python package and import namespace. This module is the only place the two differ on
purpose rather than by mistake, and a test asserts the page says `agentrag`.

## What it is for

Not "a UI for the demo". The one thing this project does that a stock RAG pipeline does
not is *verify* citations against the source text, and a JSON response makes that hard to
see: `"method": "exact"` is a string, and the interesting cases are the ones where the
model's quote and the document differ by a single token.

So the centre of the page is a **word-level diff** of the model's quote against the
document, LCS over whitespace-split tokens. Two panes of monospace text technically
contain the same information, but the differences that matter here are one character
wide — `40 CFR` against `21 CFR`, `60 days` against `30 days`, `may` against `shall` —
and a reader has to hunt for them. Those are the same single-token meaning changes Phase 6
found lexical coverage scoring 1.000 and accepting; the diff puts them on screen.

`tests/test_dashboard_diff.py` lifts `diffTokens`/`renderDiff` out of this string and runs
them under node, so the algorithm is executed rather than asserted to exist.

## Components

| element | what it does |
|---|---|
| `#health` | passage count, that search is BM25, and which model answers — from `/health` on load |
| `#keynote` | shown only when no API key is configured: says search still works, so a missing key does not read as a broken app |
| `#f` `#q` `#k` `#ro` | the ask bar. The controls are labelled **"passages to read"** and **"search only / skip the AI answer"** — never `k` or `retrieve_only`, which are names from the code |
| `#examples` | four one-click questions, phrased as questions, **including one the corpus cannot answer** so a visitor can watch it refuse |
| `#status` | live elapsed timer during a request, because generation is seconds of somebody else's network and a static line reads as a hang |
| `#emptyState` | "What happens when you ask" — search, answer, verify — plus a glossary. The first paint teaches instead of showing an empty page; hidden after the first query |
| `#noanswer` | the search-only / no-key explanation, in prose. Replaces an `ANSWER` card that contained no answer |
| `#answerCard` | the answer, a `#banner` naming the outcome in words, and `#verdicts` pills |
| `#citeCard` | one panel per citation: verdict glyph, char span, the diff, and a collapsed "what each verdict means" |
| `#retrCard` | **"Passages found"** — the ranking with match bars, a `quoted` badge marking what the answer actually used, and expandable passage text |
| `#timeCard` | per-stage ms and share of total for this request |
| `#cumulative` | latency across everything the process has served, from `/stats`, in words |

## Writing for someone who has never seen this project

A screenshot review found the first version written for a reader who already knew the
system, and every item below was a real complaint rather than a hypothetical one:

- **`k=5`** as a dropdown label. `k` is a variable from the retrieval code. It is now
  "passages to read", with a tooltip on the trade.
- **"retrieve only"**, unexplained, and silently *checked* when no key was configured so it
  looked like a broken app rather than a deliberate mode. Now "search only / skip the AI
  answer", with a banner saying what still works.
- **`retrieve p50 0.373ms / p95 0.373ms (n=1)`** in the footer: three pieces of jargon
  wrapped around a single number and dressed up as a distribution. The strip now says
  "Your request took: search 0.64 ms", and only earns the word "typical" at
  `MIN_TYPICAL` samples and a tail figure at `MIN_TAIL`. Same defect the project's own
  reports were corrected for.
- **A raw API token printed mid-sentence**: "No answer requested. retrieve_only Retrieval
  ran and is shown below."
- **Two words for one thing** — the explainer said "passages", the results table said
  "chunks". A test now enforces one.
- **A route list and a paragraph about per-token rates** on first paint, before the visitor
  has asked anything. Both folded into "For developers".

Jargon that remains — BM25, passage, verified — is defined in a glossary on the page, and
a test asserts each has a `<dt>` and not merely a mention.

## Decisions worth not undoing

**Both citation panes are always shown**, including when they match. Showing the source
only on a mismatch would make `verified` the one state in which the reader cannot check
the work — and `verified` is the claim most worth checking. When the two are equal after
whitespace collapsing the panes render *plain*, because marking the whitespace tokens
would put highlights directly under a legend saying there is nothing to mark.

**Every verdict carries a glyph as well as a colour** (`✓` / `!` / `✗`), so it survives a
colour-blind reader and a greyscale screenshot.

**The three refusal sources are described as what actually happened.** Only
`refusal_source: "model"` means a model was asked and declined; `no_context` and
`score_gate` refuse *before* generating, cost no quota, and involve no model. An earlier
version of the banner described all three as the model declining, which is a false
statement about two of them — caught by cross-checking the page's field reads against
every response branch.

**Failure modes the citation panel exists to display:**

- `exact` / `normalized` — verified. Normalized matched after whitespace collapsing, which
  is the common case in PDF text and is still a real match.
- `wrong_chunk` — the quote exists in the retrieved context but not in the chunk the model
  attributed it to. A real sentence, a wrong citation. One of the four citations in the
  first live request through this service came back this way.
- `line_number_ambiguous` — deliberately **not** counted as verified, and never green.
  Phase 5's gate found this tier verifying a meaning change: a model's "40 CFR" scored as
  a verified citation of a document's "21 CFR", because 53.2% of chunks in this corpus lose
  digits to line-number stripping.
- `unverified` / `too_short` — not found, or too short to be evidence of anything.

**Cost is never computed here.** The page may display `cost_usd` from `/stats` — including
its `null by design` — but must never multiply a token count by a rate of its own. A test
enforces exactly that distinction: reading the server's figure is honest, inventing one is
the defect Phase 4's audit found four times.

## Why it is one string with no build step

No npm, no bundler, no framework. A single HTML document with inline CSS and vanilla JS,
served from memory by the same process that answers the queries. A portfolio project that
needs a Node toolchain to show its own output has added a moving part that demonstrates
nothing about retrieval. (node is used to *test* the diff, and is not required to run it.)

No JS template literals anywhere, deliberately: a test asserts every `$` on the page is
the DOM helper, which is how it guarantees no currency amount is ever rendered.
"""

from __future__ import annotations

#: Verdicts that count as verified. Mirrors `citations.VERIFIED_METHODS` -- imported
#: rather than restated so the page cannot drift from the verifier.
from ragpipe.citations import VERIFIED_METHODS

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agentrag — answers with citations you can check</title>
<style>
  :root {
    --bg: #0f1117; --panel: #171a23; --panel2: #1d212c; --line: #262b38; --text: #e6e8ee;
    --dim: #9aa3b8; --ok: #3fb950; --bad: #f85149; --warn: #d29922; --accent: #58a6ff;
    --ins: #2ea04326; --del: #f8514926;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }
  .wrap { max-width: 1140px; margin: 0 auto; padding: 30px 20px 90px; }
  a { color: var(--accent); }

  header { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  h1 { font-size: 21px; margin: 0; letter-spacing: -0.01em; }
  .tag {
    font-family: var(--mono); font-size: 11.5px; color: var(--dim);
    border: 1px solid var(--line); border-radius: 999px; padding: 3px 9px;
  }
  .lede { color: var(--dim); font-size: 14px; margin: 10px 0 4px; max-width: 74ch; }
  .lede b { color: var(--text); font-weight: 600; }
  .meta { color: var(--dim); font-size: 12.5px; margin: 0 0 22px; font-family: var(--mono); }

  /* ---- ask bar ---- */
  /* Bottom-aligned, not centre-aligned: the "passages to read" label sits *above* its
     select, so centring the row left the select visibly lower than the input and the
     button beside it. */
  form { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; }
  .field { position: relative; flex: 1 1 400px; }
  input[type=text] {
    width: 100%; padding: 12px 13px; border-radius: 9px; color: var(--text);
    background: var(--panel); border: 1px solid var(--line); font-size: 15px;
  }
  input[type=text]:focus-visible, select:focus-visible, button:focus-visible,
  summary:focus-visible, .chip:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 2px;
  }
  .kbd {
    position: absolute; right: 10px; top: 50%; transform: translateY(-50%);
    font-family: var(--mono); font-size: 11px; color: var(--dim);
    border: 1px solid var(--line); border-radius: 5px; padding: 1px 5px; pointer-events: none;
  }
  select, button {
    padding: 12px 14px; border-radius: 9px; background: var(--panel);
    color: var(--text); border: 1px solid var(--line); font-size: 14px; cursor: pointer;
  }
  button.primary {
    background: var(--accent); color: #06121f; border-color: var(--accent); font-weight: 650;
    min-width: 96px;
  }
  button:disabled { opacity: 0.55; cursor: progress; }
  label.check {
    color: var(--dim); font-size: 13.5px; display: flex; gap: 6px; align-items: center;
    cursor: pointer; user-select: none;
  }

  .examples { display: flex; gap: 7px; flex-wrap: wrap; margin: 12px 0 0; align-items: center; }
  .examples > span { color: var(--dim); font-size: 12.5px; margin-right: 2px; }
  .chip {
    font-size: 12.5px; color: var(--dim); background: var(--panel);
    border: 1px solid var(--line); border-radius: 999px; padding: 5px 11px; cursor: pointer;
  }
  .chip:hover { color: var(--text); border-color: var(--accent); }
  .chip { text-align: left; line-height: 1.35; }
  .chip i {
    display: block; font-style: normal; color: var(--dim); font-size: 11px; opacity: 0.85;
  }
  .chip:hover i { color: var(--accent); }

  .status {
    margin: 14px 0 0; color: var(--dim); font-size: 13.5px; font-family: var(--mono);
    min-height: 0;
  }
  .status:empty { margin: 0; }
  .status.err { color: var(--bad); }
  .spin {
    display: inline-block; width: 9px; height: 9px; margin-right: 7px; border-radius: 50%;
    background: var(--accent); animation: pulse 1s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 0.25; } 50% { opacity: 1; } }
  @media (prefers-reduced-motion: reduce) {
    html { scroll-behavior: auto; }
    .spin { animation: none; opacity: 0.8; }
  }

  /* ---- cards ---- */
  .card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 11px;
    padding: 17px 19px; margin-top: 15px;
  }
  .card > h2 {
    font-size: 12px; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--dim); margin: 0 0 13px; font-weight: 650;
    display: flex; gap: 9px; align-items: center; flex-wrap: wrap;
  }
  .card > h2 .hint { text-transform: none; letter-spacing: 0; font-weight: 400; opacity: 0.8; }
  .answer { font-size: 15.5px; line-height: 1.68; white-space: pre-wrap; }
  .answer.refused { color: var(--text); }

  .banner {
    display: flex; gap: 11px; align-items: flex-start; padding: 12px 14px;
    border-radius: 9px; margin-bottom: 13px; font-size: 14px; line-height: 1.5;
  }
  .banner .ico { font-family: var(--mono); font-weight: 700; }
  .banner.good { background: #2ea04319; border: 1px solid var(--ok); }
  .banner.good .ico { color: var(--ok); }
  .banner.info { background: #58a6ff14; border: 1px solid var(--accent); }
  .banner.info .ico { color: var(--accent); }
  .banner.warnb { background: #d2992219; border: 1px solid var(--warn); }
  .banner.warnb .ico { color: var(--warn); }
  .banner b { font-weight: 650; }
  .banner .dimnote { color: var(--dim); font-family: var(--mono); font-size: 12px; }

  .pills { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
  .pill {
    font-family: var(--mono); font-size: 11.5px; padding: 4px 9px; border-radius: 999px;
    border: 1px solid var(--line); color: var(--dim); background: #0d1017;
  }
  .pill b { color: var(--text); font-weight: 650; }
  .pill.ok { border-color: var(--ok); color: var(--ok); }
  .pill.bad { border-color: var(--bad); color: var(--bad); }
  .pill.warn { border-color: var(--warn); color: var(--warn); }

  /* ---- citations ---- */
  .cite {
    border: 1px solid var(--line); border-radius: 10px; padding: 0; margin-bottom: 12px;
    overflow: hidden; background: var(--panel2);
  }
  .cite.ok { border-left: 3px solid var(--ok); }
  .cite.bad { border-left: 3px solid var(--bad); }
  .cite.warn { border-left: 3px solid var(--warn); }
  .cite-head {
    display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 11px 14px;
    font-family: var(--mono); font-size: 11.5px; color: var(--dim);
    border-bottom: 1px solid var(--line); background: #0d1017;
  }
  /* The verdict carries a glyph as well as a colour, so it survives a colour-blind
     reader and a greyscale screenshot. */
  .verdict { font-weight: 700; display: flex; gap: 6px; align-items: center; }
  .verdict.ok { color: var(--ok); } .verdict.bad { color: var(--bad); }
  .verdict.warn { color: var(--warn); }
  .cite-body { padding: 13px 14px; }
  .side { display: grid; grid-template-columns: 1fr 1fr; gap: 13px; }
  @media (max-width: 760px) { .side { grid-template-columns: 1fr; } }
  .side > div > span.lbl {
    display: block; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--dim); margin-bottom: 6px;
  }
  .side pre, .chunktext {
    margin: 0; padding: 10px 11px; background: #0d1017; border: 1px solid var(--line);
    border-radius: 7px; font-family: var(--mono); font-size: 12.5px; line-height: 1.55;
    white-space: pre-wrap; word-break: break-word; max-height: 210px; overflow: auto;
  }
  ins, del { text-decoration: none; border-radius: 3px; padding: 0 1px; }
  ins { background: var(--ins); color: var(--ok); }
  del { background: var(--del); color: var(--bad); text-decoration: line-through; }
  .difflegend { color: var(--dim); font-size: 11.5px; margin-top: 9px; font-family: var(--mono); }
  .difflegend ins, .difflegend del { margin: 0 2px; }
  .identical { color: var(--dim); font-size: 12px; margin-top: 9px; }
  .note { color: var(--dim); font-size: 12.5px; margin-top: 11px; line-height: 1.55; }
  .note code { font-family: var(--mono); color: var(--accent); font-size: 11.5px; }

  /* ---- tables ---- */
  table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
  th, td { text-align: left; padding: 8px 9px; border-bottom: 1px solid var(--line); }
  th {
    color: var(--dim); font-weight: 650; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  td.num, th.num { text-align: right; font-family: var(--mono); }
  tr.cited td { background: #2ea04310; }
  .badge {
    font-family: var(--mono); font-size: 10.5px; border-radius: 4px; padding: 1px 6px;
    border: 1px solid var(--ok); color: var(--ok); white-space: nowrap;
  }
  details.chunk > summary {
    cursor: pointer; color: var(--dim); font-size: 12.5px; padding: 4px 0;
    list-style: none;
  }
  details.chunk > summary::-webkit-details-marker { display: none; }
  details.chunk > summary::before { content: "▸ "; color: var(--accent); }
  details.chunk[open] > summary::before { content: "▾ "; }
  .chunktext { margin-top: 7px; max-height: 260px; }

  /* ---- labelled controls ---- */
  .ctl { display: flex; flex-direction: column; gap: 3px; }
  .ctl-lbl { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--dim); }
  .ctl select { padding: 9px 11px; }
  label.check { gap: 8px; align-items: center; line-height: 1.25; }
  label.check small { display: block; font-size: 11px; opacity: 0.75; }
  label.check input {
    margin: 0; appearance: none; width: 16px; height: 16px; flex: 0 0 16px;
    border: 1px solid var(--line); border-radius: 4px; background: var(--panel);
    cursor: pointer; position: relative;
  }
  label.check input:checked { background: var(--accent); border-color: var(--accent); }
  label.check input:checked::after {
    content: "\u2713"; position: absolute; inset: 0; color: #06121f; font-size: 12px;
    font-weight: 700; display: flex; align-items: center; justify-content: center;
  }
  label.check input:disabled { opacity: 0.55; cursor: not-allowed; }
  label.check:has(input:disabled) { opacity: 0.7; cursor: not-allowed; }

  /* ---- the teaching card ---- */
  .steps { margin: 0; padding-left: 22px; }
  .steps li { margin-bottom: 9px; line-height: 1.6; }
  .steps b { font-weight: 650; }
  .ok-t { color: var(--ok); font-weight: 600; }
  .bad-t { color: var(--bad); font-weight: 600; }

  details.glossary, details.devbox { margin-top: 14px; }
  details.glossary > summary, details.devbox > summary {
    cursor: pointer; color: var(--accent); font-size: 13px; list-style: none;
    display: inline-flex; gap: 6px; align-items: center;
  }
  details.glossary > summary::-webkit-details-marker,
  details.devbox > summary::-webkit-details-marker { display: none; }
  details.glossary > summary::before, details.devbox > summary::before { content: "\u25b8"; }
  details.glossary[open] > summary::before, details.devbox[open] > summary::before { content: "\u25be"; }
  details.glossary dl { margin: 12px 0 0; font-size: 13.5px; }
  details.glossary dt {
    font-weight: 650; color: var(--text); margin-top: 11px; font-family: var(--mono);
    font-size: 12.5px;
  }
  details.glossary dd { margin: 3px 0 0; color: var(--dim); line-height: 1.6; }
  details.glossary code, details.devbox code {
    font-family: var(--mono); font-size: 12px; color: var(--accent);
  }
  details.devbox p { color: var(--dim); line-height: 1.7; margin: 10px 0 0; }

  .bar { height: 5px; border-radius: 3px; background: var(--accent); min-width: 2px; }
  .hidden { display: none; }
  footer {
    margin-top: 36px; padding-top: 18px; border-top: 1px solid var(--line);
    color: var(--dim); font-size: 12.5px; line-height: 1.75;
  }
  footer code { font-family: var(--mono); color: var(--accent); }
  #cumulative { font-family: var(--mono); font-size: 12px; color: var(--dim); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>agentrag</h1>
    <span class="tag">retrieval-augmented answers, every quote checked</span>
  </header>
  <p class="lede">
    Ask a question about public FDA and ClinicalTrials.gov documents. You get an answer
    <b>and proof</b>: every quote the AI uses is checked against the original document
    before you see it. If a quote cannot be found, this page says so instead of hiding it.
  </p>
  <p class="meta" id="health" aria-live="polite"></p>
  <div id="keynote"></div>

  <form id="f" autocomplete="off">
    <span class="field">
      <label class="lbl" for="q" hidden>Your question</label>
      <input type="text" id="q" placeholder="Ask about reporting deadlines, submission routes, CFR citations…">
      <span class="kbd" id="kbdhint">/</span>
    </span>
    <span class="ctl">
      <label class="ctl-lbl" for="k">passages to read</label>
      <select id="k" title="How many of the best-matching passages get sent to the AI. More passages means more context and a slower, more expensive answer.">
        <option value="3">3</option>
        <option value="5" selected>5</option>
        <option value="10">10</option>
      </select>
    </span>
    <label class="check" for="ro" title="Search the documents and stop there. No AI, no waiting, nothing generated.">
      <input type="checkbox" id="ro"> <span>search only<br><small>skip the AI answer</small></span>
    </label>
    <button class="primary" id="go" type="submit">Ask</button>
  </form>

  <div class="examples" id="examples"><span>Try one:</span></div>
  <p class="status" id="status" role="status" aria-live="polite"></p>

  <div id="noanswer"></div>

  <div class="card" id="emptyState">
    <h2>What happens when you ask</h2>
    <ol class="steps">
      <li><b>Search.</b> Your question is matched against 13,423 passages by keyword
        (BM25). No AI involved, and it takes under a millisecond.</li>
      <li><b>Answer.</b> The best-matching passages — and nothing else — are sent to the
        AI, which must quote them. This is the slow part: a few seconds.</li>
      <li><b>Verify.</b> Every quote is searched for in the original document. Found means
        <span class="ok-t">verified</span>. Not found is reported as
        <span class="bad-t">unverified</span> rather than quietly dropped — which is the
        part most systems skip.</li>
    </ol>
    <details class="glossary">
      <summary>What do the words mean?</summary>
      <dl>
        <dt>passage</dt><dd>A slice of one document, split at its section boundaries —
          roughly a paragraph or two. The searchable unit.</dd>
        <dt>BM25 / keyword search</dt><dd>Classic word-overlap ranking. It beats AI
          embeddings badly on exact identifiers like <code>21 CFR 314.50</code>, which is
          why this project measured both and ships this one.</dd>
        <dt>passages to read</dt><dd>How many top matches get sent to the AI. Fewer is
          cheaper and more focused; more gives the AI a better chance of finding the
          answer, and more chance to wander.</dd>
        <dt>search only</dt><dd>Runs the search and stops. You see what was found without
          waiting for, or paying for, a generated answer.</dd>
        <dt>verified</dt><dd>The exact words the AI quoted were located in the source
          document. Not "looks plausible" — actually found, at a character position this
          page shows you.</dd>
      </dl>
    </details>
  </div>

  <div class="card hidden" id="answerCard">
    <h2>Answer <span class="hint" id="answerHint"></span></h2>
    <div id="banner"></div>
    <div class="answer" id="answer"></div>
    <div class="pills" id="verdicts"></div>
  </div>

  <div class="card hidden" id="citeCard">
    <h2>Citations <span class="hint">the model&rsquo;s quote, diffed against the document</span></h2>
    <div id="cites"></div>
    <p class="note">
      Both panes are shown even when they match. Showing the source only on a mismatch
      would make &ldquo;verified&rdquo; the one state where you cannot check the work.
    </p>
    <details class="glossary">
      <summary>What each verdict means</summary>
      <dl>
        <dt>exact</dt><dd>The quoted words were found in the cited passage, character for
          character. Verified.</dd>
        <dt>normalized</dt><dd>Found after collapsing runs of whitespace. Still verified —
          this is the ordinary case in text extracted from a PDF, where line breaks land
          mid-sentence.</dd>
        <dt>line_number_ambiguous</dt><dd><b>Not</b> counted as verified, deliberately.
          It matched only after stripping leading integers, and this project caught that
          transform scoring a model&rsquo;s &ldquo;40 CFR&rdquo; as a verified citation of
          a document&rsquo;s &ldquo;21 CFR&rdquo;. Better to under-claim.</dd>
        <dt>wrong_chunk</dt><dd>The sentence is real and appears in the retrieved text, but
          not in the passage the AI said it came from. A true quote with a false
          address.</dd>
        <dt>unverified</dt><dd>Not found at all. The AI produced words that are not in the
          document it cited.</dd>
        <dt>too_short</dt><dd>The quote was too short to be evidence of anything, so it is
          not credited.</dd>
      </dl>
    </details>
  </div>

  <div class="card hidden" id="retrCard">
    <h2>Passages found <span class="hint" id="retrHint"></span></h2>
    <table>
      <thead><tr>
        <th>#</th><th>document</th><th>section</th><th class="num" title="BM25 keyword-overlap score. Higher is a closer keyword match; it is not a probability.">match</th><th></th>
      </tr></thead>
      <tbody id="retr"></tbody>
    </table>
    <div id="chunks"></div>
  </div>

  <div class="card hidden" id="timeCard">
    <h2>This request, by stage <span class="hint">retrieval and verification are local; generation is not</span></h2>
    <table>
      <thead><tr><th>stage</th><th class="num">ms</th><th>share of total</th></tr></thead>
      <tbody id="times"></tbody>
    </table>
    <p class="note" id="timeNote"></p>
  </div>

  <footer>
    <div id="cumulative"></div>
    <details class="devbox">
      <summary>For developers &mdash; API and cost</summary>
      <p>
        This page is a client of the same HTTP API you can call directly:
        <code>GET /health</code> readiness and index size &middot;
        <code>POST /query</code> search, answer and verify (the call this page makes)
        &middot; <code>GET /stats</code> latency percentiles and token totals for this
        process &middot; <code>GET /docs</code> interactive OpenAPI.
      </p>
      <p>
        Usage is reported in <b>tokens, not dollars</b>. The free tier bills nothing and no
        confirmed paid per-token rate for these models was ever obtained, so a currency
        figure here would be one this project invented — and an earlier phase of it was
        caught doing exactly that four times. Set <code>RAGPIPE_PRICE_IN</code> and
        <code>RAGPIPE_PRICE_OUT</code> to have the server compute real costs.
      </p>
    </details>
  </footer>
</div>

<script>
const VERIFIED = ["exact", "normalized"];
const AMBIGUOUS = ["line_number_ambiguous"];
const GLYPH = { ok: "\\u2713", warn: "!", bad: "\\u2717" };

const EXAMPLES = [
  { label: "When is an annual report due?",
    q: "What is the deadline for submitting an annual report?", k: 5 },
  { label: "How do I submit a 904(a)(3) report?",
    q: "How should a 904(a)(3) report be submitted to FDA?", k: 5 },
  { label: "What goes in a device change plan?",
    q: "What must a Predetermined Change Control Plan include for a device?", k: 5 },
  { label: "Ask something the documents don't cover",
    q: "What is the boiling point of helium at 3 atmospheres?", k: 5,
    why: "watch it refuse instead of guessing" }
];

const $ = (id) => document.getElementById(id);
const esc = (s) => (s == null ? "" : String(s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])));

function klass(method) {
  if (VERIFIED.includes(method)) return "ok";
  if (AMBIGUOUS.includes(method)) return "warn";
  return "bad";
}

/* ---------------------------------------------------------------------------
   Word-level diff, model quote vs document text.

   The centrepiece. Two panes of monospace text next to each other technically
   contain the answer, but a reader has to hunt for it -- and the differences that
   matter here are one character wide: "40 CFR" against "21 CFR", or a doubled space
   left by stripping a PDF line number. Highlighting them is the difference between
   a demo you can read and a demo you have to study.

   Longest-common-subsequence over whitespace-split tokens. Quotes are a sentence or
   two, so the O(n*m) table is a few thousand cells.
--------------------------------------------------------------------------- */
function diffTokens(a, b) {
  const n = a.length, m = b.length;
  const lcs = [];
  for (let i = 0; i <= n; i++) lcs.push(new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1
                                : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const out = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { out.push(["same", a[i]]); i++; j++; }
    else if (lcs[i + 1][j] >= lcs[i][j + 1]) { out.push(["del", a[i]]); i++; }
    else { out.push(["ins", b[j]]); j++; }
  }
  while (i < n) { out.push(["del", a[i]]); i++; }
  while (j < m) { out.push(["ins", b[j]]); j++; }
  return out;
}

function renderDiff(quote, source, side) {
  const qa = (quote || "").split(/(\\s+)/).filter((t) => t.length);
  const sa = (source || "").split(/(\\s+)/).filter((t) => t.length);
  const ops = diffTokens(qa, sa);
  let html = "";
  for (const op of ops) {
    const kind = op[0], tok = esc(op[1]);
    if (kind === "same") html += tok;
    else if (kind === "del" && side === "model") html += "<del>" + tok + "</del>";
    else if (kind === "ins" && side === "source") html += "<ins>" + tok + "</ins>";
    else if (kind === "del" && side === "source") continue;
    else if (kind === "ins" && side === "model") continue;
  }
  return html;
}

function diffIsClean(quote, source) {
  const norm = (t) => (t || "").replace(/\\s+/g, " ").trim();
  return norm(quote) === norm(source);
}

/* --------------------------------------------------------------------------- */
async function health() {
  try {
    const h = await (await fetch("/health")).json();
    $("health").textContent = h.chunks.toLocaleString()
      + " passages, searched by keyword (BM25)"
      + (h.generator_available ? " \\u00b7 answers by " + h.generator : "");
    if (!h.generator_available) {
      // Not an error, and it must not read like one: the search half is fully local and
      // works. Say what still works, then what does not, then why.
      $("keynote").innerHTML = '<div class="banner info"><span class="ico" aria-hidden="true">i</span>'
        + "<span><b>Search works; AI answers are switched off.</b> No API key is configured, "
        + "so this demo will find and show you the relevant passages but will not generate "
        + "an answer from them. The verification step needs an answer to check, so it is "
        + "idle too. Everything below is real search output.</span></div>";
      $("ro").checked = true;
      $("ro").disabled = true;
      $("go").textContent = "Search";
      $("go").title = "No API key configured, so this searches without generating an answer.";
    }
  } catch (e) {
    $("health").textContent = "the service is not responding";
  }
}

function renderExamples() {
  const host = $("examples");
  for (const ex of EXAMPLES) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.innerHTML = esc(ex.label) + (ex.why ? " <i>" + esc(ex.why) + "</i>" : "");
    b.addEventListener("click", () => {
      $("q").value = ex.q;
      $("k").value = String(ex.k);
      ask();
    });
    host.appendChild(b);
  }
}

function renderBanner(d) {
  const host = $("banner");
  host.innerHTML = "";
  const v = d.verification;
  let cls, ico, text;
  if (d.refused) {
    cls = "good"; ico = GLYPH.ok;
    // Three different refusals, and conflating them would be a false claim about what
    // happened. `refusal_source` exists in the API for exactly this reason: only "model"
    // means a model was asked and declined. The other two refuse *before* generating, so
    // they cost no quota and involved no model at all -- describing them as the model
    // declining would be inventing an event.
    if (d.refusal_source === "no_context") {
      text = "<b>Refused before generating.</b> Retrieval returned nothing usable, so there "
        + "was no evidence to ground an answer in and no request was sent to the model. "
        + "No quota spent, and nothing was invented to fill the gap.";
    } else if (d.refusal_source === "score_gate") {
      text = "<b>Refused before generating.</b> The top retrieval score fell below the "
        + "refusal threshold, so the question was judged unanswerable from this corpus "
        + "without asking the model. No quota spent.";
    } else {
      text = "<b>The model refused, and that is the correct outcome.</b> It was told to "
        + "answer only from the retrieved text and reported that it could not. A refusal "
        + "is a feature here: the alternative is a confident answer with no support.";
    }
    if (d.reason) text += ' <span class="dimnote">' + esc(d.reason) + "</span>";
  } else if (!v || v.claimed === 0) {
    cls = "warnb"; ico = GLYPH.warn;
    text = "<b>Answered without citing anything.</b> Nothing to verify, so nothing here is "
      + "grounded — treat it as unsupported.";
  } else if (v.verified === v.claimed) {
    cls = "good"; ico = GLYPH.ok;
    text = "<b>Every quote was located in the source.</b> " + v.verified + " of "
      + v.claimed + " citations verified against the document text below.";
  } else {
    cls = "warnb"; ico = GLYPH.warn;
    text = "<b>" + (v.claimed - v.verified) + " of " + v.claimed + " citations did not verify.</b> "
      + "The diff below shows exactly where the model&rsquo;s quote and the document part ways.";
  }
  host.innerHTML = '<div class="banner ' + cls + '"><span class="ico" aria-hidden="true">'
    + ico + '</span><span>' + text + "</span></div>";
}

function renderVerdicts(d) {
  const v = d.verification, out = [];
  if (v) {
    const all = v.claimed > 0 && v.verified === v.claimed;
    out.push('<span class="pill ' + (all ? "ok" : (v.claimed ? "bad" : "warn")) + '">located <b>'
      + v.verified + "/" + v.claimed + "</b></span>");
    if (v.precision != null) {
      out.push('<span class="pill">precision <b>' + v.precision.toFixed(2) + "</b></span>");
    }
    for (const name of Object.keys(v.buckets || {})) {
      out.push('<span class="pill ' + klass(name) + '">' + esc(name) + " <b>"
        + v.buckets[name] + "</b></span>");
    }
  }
  if (d.tokens) {
    out.push('<span class="pill">tokens <b>' + d.tokens.prompt + " in / "
      + d.tokens.output + " out"
      + (d.tokens.thinking ? " / " + d.tokens.thinking + " think" : "") + "</b></span>");
  }
  $("verdicts").innerHTML = out.join("");
}

function renderCites(cites) {
  if (!cites || !cites.length) { $("citeCard").classList.add("hidden"); return; }
  const parts = [];
  for (let i = 0; i < cites.length; i++) {
    const c = cites[i];
    const cls = klass(c.method);
    const span = (c.start != null && c.end != null) ? c.start + "\\u2013" + c.end : "not located";
    const located = c.source_text ? true : false;
    const clean = located && diffIsClean(c.quote, c.source_text);

    let modelPane, sourcePane, legend;
    if (located && clean) {
      // Equal once whitespace is collapsed. Marking the whitespace tokens would put
      // highlights on screen directly under a legend saying there is nothing to diff,
      // so the panes are rendered plain and the legend says why.
      modelPane = esc(c.quote);
      sourcePane = esc(c.source_text);
      legend = '<p class="identical">Identical once whitespace is collapsed &mdash; the '
        + "only differences are line breaks and spacing from the PDF, so there is nothing "
        + "to mark. Both panes are shown anyway.</p>";
    } else if (located) {
      modelPane = renderDiff(c.quote, c.source_text, "model");
      sourcePane = renderDiff(c.quote, c.source_text, "source");
      legend = '<p class="difflegend">'
        + "<del>struck</del> = in the model&rsquo;s quote only &nbsp; "
        + "<ins>highlighted</ins> = in the document only</p>";
    } else {
      modelPane = esc(c.quote);
      sourcePane = "\\u2014 not found in the passage the AI cited \\u2014";
      legend = '<p class="difflegend">No span to diff: the quote was not found, so there is '
        + "nothing in the document to line it up against.</p>";
    }

    parts.push(
      '<div class="cite ' + cls + '">'
      + '<div class="cite-head">'
      + '<span class="verdict ' + cls + '"><span aria-hidden="true">' + GLYPH[cls]
      + "</span>" + esc(c.method) + "</span>"
      + "<span>" + esc(c.doc_id || "?") + "</span>"
      + "<span>" + esc(c.chunk_id) + "</span>"
      + "<span>chars " + span + "</span>"
      + "</div>"
      + '<div class="cite-body"><div class="side">'
      + '<div><span class="lbl">the model&rsquo;s quote</span><pre>' + modelPane + "</pre></div>"
      + '<div><span class="lbl">what the document says</span><pre>' + sourcePane + "</pre></div>"
      + "</div>" + legend + "</div></div>"
    );
  }
  $("cites").innerHTML = parts.join("");
  $("citeCard").classList.remove("hidden");
}

function renderRetrieved(rows) {
  const top = rows.length ? Math.max.apply(null, rows.map((r) => r.score)) : 1;
  const body = [];
  const panels = [];
  let nCited = 0;
  for (const r of rows) {
    if (r.cited) nCited++;
    const width = Math.max(2, Math.round(100 * (r.score / (top || 1))));
    body.push(
      "<tr" + (r.cited ? ' class="cited"' : "") + ">"
      + '<td class="num">' + r.rank + "</td>"
      + "<td>" + esc(r.doc_id || "\\u2014") + "</td>"
      + "<td>" + esc(r.section_heading || "\\u2014") + "</td>"
      + '<td class="num">' + r.score.toFixed(3)
      + '<div class="bar" style="width:' + width + '%"></div></td>'
      + "<td>" + (r.cited ? '<span class="badge">quoted</span>' : "") + "</td>"
      + "</tr>"
    );
    if (r.text) {
      panels.push(
        '<details class="chunk"><summary>' + r.rank + ". " + esc(r.doc_id || "")
        + " \\u00b7 " + esc((r.section_heading || "no heading")) + " \\u00b7 "
        + r.n_chars.toLocaleString() + " chars"
        + (r.cited ? " \\u00b7 quoted in the answer" : "") + "</summary>"
        + '<div class="chunktext">' + esc(r.text)
        + (r.truncated ? "\\n\\n\\u2026 truncated" : "") + "</div></details>"
      );
    }
  }
  $("retr").innerHTML = body.join("");
  $("chunks").innerHTML = panels.length
    ? '<p class="note">Expand any of these to read the passage the search actually '
      + "returned, so you can judge the ranking for yourself.</p>"
      + panels.join("")
    : "";
  $("retrHint").textContent = rows.length
    ? rows.length + " passages ranked"
      + (nCited ? ", " + nCited + " quoted in the answer" : "")
    : "";
  $("retrCard").classList.remove("hidden");
}

function renderTimings(t) {
  const total = t.total || 0;
  const order = ["retrieve", "generate", "verify", "total"];
  const rows = [];
  for (const s of order) {
    if (t[s] == null) continue;
    const share = (s === "total" || !total) ? "" : (100 * t[s] / total).toFixed(2) + "%";
    const width = (s === "total" || !total) ? 0 : Math.max(1, Math.round(100 * t[s] / total));
    rows.push("<tr><td>" + s + '</td><td class="num">' + t[s].toFixed(3) + "</td><td>" + share
      + (width ? '<div class="bar" style="width:' + width + '%"></div>' : "") + "</td></tr>");
  }
  $("times").innerHTML = rows.join("");
  $("timeNote").textContent = (t.generate != null && t.verify != null)
    ? "Verification is deterministic and local, so it is timed rather than assumed free."
    : "Retrieval only \\u2014 no generation was requested, so no generation time is reported "
      + "(a zero would read as instant).";
  $("timeCard").classList.remove("hidden");
}

// A percentile over one observation is just the observation. The first version of this
// strip printed "retrieve p50 0.373ms / p95 0.373ms (n=1)": three pieces of jargon
// wrapped around a single number and dressed up as a distribution -- the same defect this
// project's own reports were corrected for. Below MIN_TYPICAL samples it states the plain
// measurement; the word "typical" and the tail appear only once there is a distribution.
const MIN_TYPICAL = 5;
const MIN_TAIL = 20;
const STAGE_WORDS = { retrieve: "search", generate: "AI answer", verify: "quote check" };

function human(ms) {
  if (ms == null) return "";
  return ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : ms.toFixed(ms < 10 ? 2 : 0) + " ms";
}

async function refreshCumulative() {
  try {
    const s = await (await fetch("/stats")).json();
    const l = s.latency, u = s.usage;
    const bits = [];
    for (const k of ["retrieve", "generate", "verify"]) {
      const row = l[k];
      if (!row || !row.n) continue;
      let piece = STAGE_WORDS[k] + " " + human(row.p50_ms);
      if (row.n >= MIN_TYPICAL) piece += " typical";
      if (row.n >= MIN_TAIL) piece += ", slowest 5% over " + human(row.p95_ms);
      bits.push(piece);
    }
    if (!bits.length) { $("cumulative").textContent = ""; return; }
    const n = Math.max.apply(null,
      ["retrieve", "generate", "verify"].map((k) => (l[k] && l[k].n) || 0));
    let text = (n === 1 ? "Your request took: " : "Across " + n + " requests on this page: ")
      + bits.join(" \\u00b7 ") + ".";
    if (u.n_generated && u.tokens_per_query != null) {
      text += "  " + u.tokens_per_query + " tokens per answer on average"
        + (u.cost_usd === null ? ", billed as tokens rather than dollars" : "") + ".";
    }
    $("cumulative").textContent = text;
  } catch (e) { /* the strip is decoration; never let it break the page */ }
}

function permalink(q, k, ro) {
  const p = new URLSearchParams();
  p.set("q", q);
  p.set("k", String(k));
  if (ro) p.set("r", "1");
  // The fragment is carried through deliberately. `replaceState` with a query-only
  // relative reference *drops* it -- so `#citations` was cleared here, before the fetch
  // resolved, and the scroll that reads `location.hash` could never fire. The link looked
  // implemented, had a test asserting its code was present, and did nothing.
  history.replaceState(null, "", "?" + p.toString() + location.hash);
}

let ticker = null;
function startTicker(retrieveOnly) {
  const t0 = performance.now();
  const label = retrieveOnly
    ? "retrieving\\u2026"
    : "retrieving \\u00b7 generating \\u00b7 verifying \\u2014 generation is a network call";
  const paint = () => {
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    $("status").innerHTML = '<span class="spin" aria-hidden="true"></span>' + label
      + "  " + secs + "s";
  };
  paint();
  ticker = setInterval(paint, 100);
}
function stopTicker() { if (ticker) { clearInterval(ticker); ticker = null; } }

async function ask() {
  // Captured before `permalink()` rewrites the URL: whether the caller asked to land on
  // the citations is a property of the *request*, not of whatever the address bar says
  // several seconds later once a network round trip has completed.
  const wantCitations = location.hash === "#citations";
  const query = $("q").value.trim();
  if (!query) { $("q").focus(); return; }
  const k = parseInt($("k").value, 10);
  const ro = $("ro").checked;

  $("go").disabled = true;
  $("status").className = "status";
  for (const id of ["answerCard", "citeCard", "retrCard", "timeCard"]) {
    $(id).classList.add("hidden");
  }
  $("emptyState").classList.add("hidden");
  $("noanswer").innerHTML = "";
  permalink(query, k, ro);
  startTicker(ro);
  const started = performance.now();

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: query, k: k, retrieve_only: ro })
    });
    const d = await res.json();
    stopTicker();
    if (!res.ok) {
      $("status").className = "status err";
      $("status").textContent = "error " + res.status + ": "
        + (d.detail ? JSON.stringify(d.detail) : "request failed");
      return;
    }
    const wall = ((performance.now() - started) / 1000).toFixed(2);

    if (d.answer || d.refused) {
      $("answer").textContent = d.answer || "";
      $("answerHint").textContent = d.refused ? "refused" : "";
      renderBanner(d);
      renderVerdicts(d);
      $("answerCard").classList.remove("hidden");
    } else {
      // The raw `reason` is an API token ("retrieve_only", "no generator configured"), and
      // an earlier version printed it mid-sentence: "No answer requested. retrieve_only
      // Retrieval ran and is shown below." The card is also hidden rather than shown
      // empty -- a panel headed ANSWER containing no answer is worse than no panel.
      $("answer").textContent = "";
      $("answerHint").textContent = "";
      $("answerCard").classList.add("hidden");
      // Five distinct causes. The first version collapsed three of them into "running
      // without a key" — a confident false statement when the real cause is a spent daily
      // quota, and the two are fixed completely differently. The daily and per-minute
      // cases were then collapsed in turn, because both bodies contain the word "quota";
      // those need opposite advice, so the daily one is decided upstream off the
      // structured quotaId and arrives here as a field. Order matters below: every
      // specific test precedes the substring fallback.
      const reason = d.reason || "";
      let why;
      if (reason === "retrieve_only") {
        why = "<b>Search only.</b> You asked for the matching passages without an AI answer, "
          + "so nothing was generated and there was nothing to verify. The passages are below.";
      } else if (d.refusal_source === "daily_quota") {
        why = "<b>The daily AI quota is used up.</b> This runs on a free tier capped at 20 "
          + "requests per day, and search is unaffected — the passages below are the real "
          + "ranking. Verification needs an answer to check, so it is idle until the quota "
          + "resets.";
      } else if (/quota/i.test(reason)) {
        // A *per-minute* 429's body also reads "you exceeded your current quota", and the
        // advice is the opposite: wait seconds, not a day. This branch used to test the
        // substring first and told people to come back tomorrow over a rate limit that
        // would have cleared before they finished reading it. The daily case is decided
        // upstream off the structured `quotaId` and arrives as `refusal_source`.
        why = "<b>Rate limited, briefly.</b> The AI provider refused this request for asking "
          + "too quickly, not because the day's allowance is gone. Search is unaffected — the "
          + "passages below are the real ranking. Ask again in a few seconds.";
      } else if (/no generator/i.test(reason)) {
        why = "<b>No AI answer available.</b> This service is running without an API key, so "
          + "it searched the documents but could not generate an answer. The passages are "
          + "below.";
      } else {
        why = "<b>No AI answer available.</b> Search ran and the passages are below. Reason: "
          + '<span class="dimnote">' + esc(reason.slice(0, 160)) + "</span>";
      }
      $("noanswer").innerHTML = '<div class="banner info"><span class="ico" aria-hidden="true">i</span>'
        + "<span>" + why + "</span></div>";
    }

    renderCites(d.citations);
    // A `#citations` fragment scrolls to the proof once it exists. The card is built
    // after the fetch resolves, so a plain anchor cannot work on load -- and without
    // this, deep-linking to the verification panes is impossible and any tool capturing
    // them has to guess a pixel offset against a page whose height depends on how many
    // citations the model produced. `bin/build-demo` was doing exactly that, and cropped
    // the wrong region the first time the answer got longer.
    //
    // `behavior: "instant"` because this is a deep link, not a reading gesture: the
    // reader asked for the proof, so animating past everything above it is motion with
    // no information in it. It also removes a timing variable for anything capturing
    // the page -- though note that headless Chrome's `--screenshot` ignores the
    // resulting scroll entirely and captures from the document origin, which is why
    // `bin/build-demo` reads `data-cite-top` below instead of relying on this.
    if (wantCitations && !$("citeCard").classList.contains("hidden")) {
      $("citeCard").scrollIntoView({ block: "start", behavior: "instant" });
    }
    // The card's document offset, published for anything that has to frame it without a
    // viewport -- a screenshot tool cannot ask the layout engine where a card is, and
    // the whole reason the `#citations` link exists is that guessing a pixel offset
    // against a page whose height depends on the answer is not a crop, it is a guess.
    // Reported by the page, which is the only thing that knows.
    document.body.dataset.citeTop = String(
      Math.round($("citeCard").getBoundingClientRect().top + window.scrollY)
    );
    renderRetrieved(d.retrieved || []);
    if (d.timings_ms) renderTimings(d.timings_ms);
    $("status").textContent = wall + "s round trip.";
    refreshCumulative();
  } catch (e) {
    stopTicker();
    $("status").className = "status err";
    $("status").textContent = "request failed: " + e;
  } finally {
    $("go").disabled = false;
  }
}

$("f").addEventListener("submit", (ev) => { ev.preventDefault(); ask(); });

// The "/" badge promises a shortcut. On a touch device there is no key to press, so the
// badge would be advertising something the visitor cannot do.
if (!window.matchMedia || !window.matchMedia("(hover: hover)").matches) {
  $("kbdhint").classList.add("hidden");
}

// "/" focuses the box, the way search-first tools behave. Escape gives the page back.
document.addEventListener("keydown", (ev) => {
  if (ev.key === "/" && document.activeElement !== $("q")) {
    ev.preventDefault();
    $("q").focus();
  } else if (ev.key === "Escape" && document.activeElement === $("q")) {
    $("q").blur();
  }
});

// A shared link reproduces the exact query, which is what makes a demo linkable.
(function boot() {
  renderExamples();
  health();
  refreshCumulative();
  const p = new URLSearchParams(location.search);
  const q = p.get("q");
  if (q) {
    $("q").value = q;
    if (p.get("k")) $("k").value = p.get("k");
    if (p.get("r")) $("ro").checked = true;
    ask();
  }
})();
</script>
</body>
</html>
"""


def verified_methods_match_page() -> bool:
    """True when the page's `VERIFIED` list still matches the verifier's.

    The page hardcodes the two verified methods in JavaScript, which is a duplicated
    constant and therefore something that can drift. This is checked by a test rather
    than trusted: if `citations.VERIFIED_METHODS` ever gains or loses a member, a green
    badge in the browser would silently start disagreeing with the report.
    """
    marker = 'const VERIFIED = ["'
    start = DASHBOARD_HTML.index(marker) + len(marker)
    end = DASHBOARD_HTML.index("]", start)
    listed = {part.strip().strip('"') for part in DASHBOARD_HTML[start:end].split('", "')}
    return listed == set(VERIFIED_METHODS)
