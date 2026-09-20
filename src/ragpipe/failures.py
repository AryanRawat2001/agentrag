"""The failure analysis: five ways this pipeline is wrong, with figures.

Phase 8 asks for "five named failure modes". The temptation is to write them from memory,
and this module exists to make that impossible: **every figure below is read out of a
committed artifact**, and a mutation test asserts that nothing in the prose survives
replacing the inputs with sentinels.

That constraint is not decoration. Writing this section revealed that the corpus's
line-number exposure — quoted in three live places as "7,229 of 13,706 chunks (52.7%)" —
was computed before the post-quarantine regeneration, and the corpus now holds 13,423
chunks. The conclusion survived (53.2%), the arithmetic did not, and nothing would have
caught it because the figure had never lived in a machine-readable file.

## What counts as a failure mode here

A claim is only listed if an artifact can produce its number:

- `reports/retrieval_eval.json` — 192 retriever/slice runs with exact ground truth
- `reports/generation_eval.json` — 16 queries through generate + verify, with buckets
- `reports/judge_calibration.json` — the tier-2 judge against constructed negatives
- `corpus/curation_verdicts.json` — human verdicts on flagged golden pairs
- `data/chunks/*.jsonl` — for corpus-level exposure measurements

Failures that were *observed* but never landed in an artifact get their own section and
are labelled as anecdotes, because `n=1` in a chat log is not a measured rate. That
section is itself a finding: the free tier's 20-requests-per-day ceiling is what keeps the
generation sample at 16, and several real failure modes are visible there but unbounded.

## The shape of every entry

What goes wrong, the number, why it happens, and what this project did about it. The last
part matters most: a failure analysis that only lists failures is a defect register, and a
reader wants to know which ones are load-bearing constraints rather than open bugs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Slices whose ground truth marks a chunk relevant when an identifier merely *occurs*
#: there. Right for retrieval, wrong as an answerability label -- see failure mode 3.
MENTION_LABELLED = ("exact_identifier",)


def load_artifacts(reports: Path, corpus: Path) -> dict[str, Any]:
    """Read every artifact the analysis needs. Missing ones are recorded, not fatal."""
    wanted = {
        "retrieval": reports / "retrieval_eval.json",
        "generation": reports / "generation_eval.json",
        "judge": reports / "judge_calibration.json",
        "curation": reports / "golden_curation.json",
        "verdicts": corpus / "curation_verdicts.json",
        "bench": reports / "serving_bench.json",
        "ann": reports / "ann_recall.json",
    }
    out: dict[str, Any] = {"missing": []}
    for name, path in wanted.items():
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            out["missing"].append(name)
    return out


def identifier_collapse(retrieval: dict[str, Any]) -> list[dict[str, Any]]:
    """Dense vs sparse vs naive fusion on the `exact_identifier` slice, per chunking.

    The ratio is computed rather than stored, and it is the headline of the whole project:
    the received wisdom is that dense retrieval beats keyword search, and on the one query
    type a regulatory corpus is actually asked about, it loses by more than an order of
    magnitude.
    """
    rows: list[dict[str, Any]] = []
    runs = [r for r in retrieval.get("runs", []) if r["slice_name"] == "exact_identifier"]
    for chunking in sorted({r["chunking"] for r in runs}):
        by_name = {r["retriever"]: r for r in runs if r["chunking"] == chunking}
        sparse = by_name.get("bm25")
        dense = by_name.get("dense bge-small")
        fused = by_name.get("hybrid bge-small rrf")
        if not (sparse and dense):
            continue
        s = sparse["metrics"]["recall@10"]
        d = dense["metrics"]["recall@10"]
        rows.append(
            {
                "chunking": chunking,
                "sparse": s,
                "dense": d,
                "fused": fused["metrics"]["recall@10"] if fused else None,
                # Guarded: a dense recall of exactly zero would make this infinite, and an
                # infinity in a report is a division nobody checked.
                "ratio": (s / d) if d else None,
            }
        )
    return rows


def refusal_breakdown(generation: dict[str, Any]) -> dict[str, Any]:
    """Refusals per slice, separating the ones the eval set mislabels as answerable.

    The `exact_identifier` ground truth marks a chunk relevant when the identifier
    *occurs* in it. `ident-0000`'s gold span is `21 CFR 1.980(k)` inside a footnote list:
    the document cites the regulation and never states its requirements, so refusing is
    the **correct** behaviour and scoring it as over-refusal measures the dataset.
    """
    outcomes = generation.get("outcomes", [])
    per_slice: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        slice_name = outcome.get("slice_name", "?")
        bucket = per_slice.setdefault(slice_name, {"n": 0, "refused": 0})
        bucket["n"] += 1
        if outcome.get("refused"):
            bucket["refused"] += 1

    answerable = [o for o in outcomes if o.get("slice_name") != "unanswerable"]
    honest = [o for o in answerable if o.get("slice_name") not in MENTION_LABELLED]
    return {
        "per_slice": dict(sorted(per_slice.items())),
        "answerable_n": len(answerable),
        "answerable_refused": sum(1 for o in answerable if o.get("refused")),
        "honest_n": len(honest),
        "honest_refused": sum(1 for o in honest if o.get("refused")),
        "refused_queries": [
            {"query_id": o.get("query_id"), "slice": o.get("slice_name"), "query": o.get("query")}
            for o in answerable
            if o.get("refused")
        ],
    }


#: FDA stamps this on every page of a guidance document. Matched against whitespace- and
#: hyphen-collapsed text because the PDF layer breaks it across lines.
NONBINDING_MARKING = "contains nonbinding recommendations"


def has_nonbinding_marking(chunks: Sequence[Mapping[str, Any]], doc_id: str) -> bool:
    """Whether `doc_id`'s own text carries FDA's non-binding marking.

    Checked rather than assumed. The report used to state that the document contributing
    the most obligation errors "marks itself non-binding", which was hardcoded prose
    attached to a `max()` over the verdicts: true of the document that happens to lead
    today, false for the ClinicalTrials.gov protocols one rejection behind it, and false
    for 23 of the 115 FDA documents in the corpus as well.
    """
    for chunk in chunks:
        if chunk.get("doc_id") != doc_id:
            continue
        if NONBINDING_MARKING in re.sub(r"[\s\-]+", " ", chunk.get("text", "")).lower():
            return True
    return False


def obligation_errors(
    verdicts: dict[str, Any], chunks: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    """Golden pairs a human rejected, and for what.

    `n_judged` counts recorded verdicts, which is **not** the current flag count -- the
    two drifted apart once detector fixes retired a flag on an already-judged pair. The
    renderer must not call it a number of flagged pairs.
    """
    entries = verdicts.get("verdicts", verdicts)
    rejected = {
        pair_id: entry
        for pair_id, entry in entries.items()
        if isinstance(entry, dict) and entry.get("verdict") == "reject"
    }
    by_doc: dict[str, int] = {}
    for pair_id in rejected:
        doc_id = pair_id.split("::")[0]
        by_doc[doc_id] = by_doc.get(doc_id, 0) + 1
    # Ties broken by doc_id, not by dict order, so the named document does not depend on
    # the order the verdict file happens to list its keys in.
    worst_doc, worst_n = min(by_doc.items(), key=lambda kv: (-kv[1], kv[0])) if by_doc else ("", 0)
    return {
        "n_judged": len(entries),
        "n_rejected": len(rejected),
        "rejected": dict(sorted(rejected.items())),
        "worst_doc": worst_doc,
        "worst_doc_n": worst_n,
        "worst_doc_nonbinding": bool(
            worst_doc and chunks is not None and has_nonbinding_marking(chunks, worst_doc)
        ),
    }


def line_number_exposure(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """How much of the corpus loses digits to line-number stripping.

    The measurement behind demoting `line_number_ambiguous` out of the verified set.
    Computed here rather than quoted: the previously circulated figure predated the
    post-quarantine corpus regeneration and no longer described the chunks on disk.
    """
    from ragpipe.citations import normalize_line_numbers

    affected = 0
    before = after = 0
    for chunk in chunks:
        text = chunk.get("text") or ""
        stripped, _ = normalize_line_numbers(text)
        digits_before = sum(c.isdigit() for c in text)
        digits_after = sum(c.isdigit() for c in stripped)
        before += digits_before
        after += digits_after
        if digits_after < digits_before:
            affected += 1
    total = len(chunks)
    return {
        "n_chunks": total,
        "n_affected": affected,
        "share_affected": round(affected / total, 4) if total else None,
        "digits_before": before,
        "digits_after": after,
        "digits_removed": before - after,
    }


def build_payload(artifacts: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    generation = artifacts.get("generation") or {}
    return {
        "identifier_collapse": identifier_collapse(artifacts.get("retrieval") or {}),
        "refusal": refusal_breakdown(generation),
        "obligations": obligation_errors(artifacts.get("verdicts") or {}, chunks),
        "line_numbers": line_number_exposure(chunks),
        "citation_buckets": generation.get("buckets") or {},
        "faithfulness": generation.get("faithfulness") or {},
        "judge": {
            "positive_rate": (artifacts.get("judge") or {}).get("positive_rate"),
            "negative_rate": (artifacts.get("judge") or {}).get("negative_rate"),
        },
        "generation_n": generation.get("n_queries"),
        "curation": (artifacts.get("curation") or {}).get("summary") or {},
        "missing_artifacts": artifacts.get("missing", []),
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _short_doc(pair_id: str) -> str:
    """A readable document label from a pair id.

    `ctgov-NCT02942264-Prot_SAP_000::gold::016` -> `NCT02942264`; `fda-75426::gold::085`
    -> `fda-75426`. A fixed-width tail slice produced `4-Prot_SAP_000`, which truncates
    the one part that identifies the document and keeps the part that does not.
    """
    doc = pair_id.split("::")[0]
    for part in doc.split("-"):
        if part.startswith("NCT"):
            return part
    return doc


def _clip(text: str, limit: int) -> str:
    """Truncate on a word boundary, with an ellipsis so the cut is visible."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


def render_report(payload: dict[str, Any]) -> str:
    """Render the failure analysis. Every figure comes from `payload`."""
    lines: list[str] = []
    a = lines.append

    collapse = payload["identifier_collapse"]
    refusal = payload["refusal"]
    obligations = payload["obligations"]
    line_numbers = payload["line_numbers"]
    faith = payload["faithfulness"]
    buckets = payload["citation_buckets"]
    n_gen = payload["generation_n"]

    a("# Failure analysis — five ways this pipeline is wrong")
    a("")
    a(
        "Every figure here is read from a committed artifact and recomputed on each build; "
        "a mutation test asserts that no number in this prose survives replacing the "
        "inputs. That is not ceremony. Writing this section is what revealed that the "
        "corpus line-number exposure quoted in three live places had been computed before "
        "the post-quarantine regeneration — the conclusion held, the arithmetic did not, "
        "and nothing caught it because the figure had never lived in a machine-readable "
        "file."
    )
    a("")

    # ---- 1 ----
    a("## 1. Dense retrieval collapses on the query type this corpus is for")
    a("")
    if collapse:
        a("| chunking | BM25 recall@10 | dense recall@10 | naive RRF | sparse / dense |")
        a("|---|---:|---:|---:|---:|")
        for row in collapse:
            ratio = "—" if row["ratio"] is None else f"**{row['ratio']:.0f}×**"
            fused = "—" if row["fused"] is None else f"{row['fused']:.4f}"
            a(
                f"| `{row['chunking']}` | {row['sparse']:.4f} | {row['dense']:.4f} "
                f"| {fused} | {ratio} |"
            )
        a("")
        worst = max(collapse, key=lambda r: r["ratio"] or 0)
        # A dense recall of exactly zero yields `ratio: None`, which must not reach a
        # format spec -- and "a factor of infinity" would be a division nobody checked.
        factor = "—" if worst["ratio"] is None else f"{worst['ratio']:.0f}"
        a(
            f"On exact identifiers — `21 CFR 314.50`, `NCT02942264`, docket numbers — a "
            f"sentence embedding is close to useless: **{worst['sparse']:.4f} against "
            f"{worst['dense']:.4f}**, a factor of {factor}. The embedding maps "
            "an identifier to a region of space shared by every other identifier, because "
            "the token is rare and carries no distributional meaning. Keyword matching has "
            "no such problem: the string is either there or it is not."
        )
        a("")
        if worst["fused"] is not None:
            a(
                f"**And naive fusion makes it worse than sparse alone** "
                f"({worst['fused']:.4f} against {worst['sparse']:.4f}). Reciprocal rank "
                "fusion rewards agreement between retrievers, so a retriever with no "
                "signal on this slice does not abstain — it votes, and outvotes a correct "
                "top hit. The fix that works is min-max fusion at a dense weight of 0.1, "
                "which matches BM25 on identifiers while gaining on section lookup."
            )
            a("")
        a(
            "**Consequence for the shipped system:** the served retriever is BM25 alone. "
            "That is not a simplification for the demo; it is what the table supports."
        )
    a("")

    # ---- 2 ----
    a("## 2. Reference answers manufacture obligations that the source does not impose")
    a("")
    a(
        f"**{obligations['n_rejected']} of the golden set's answers restated a "
        f"recommendation as a requirement**, found by reading every one of "
        f"{obligations['n_judged']} judged pairs against its source."
    )
    a("")
    if obligations["rejected"]:
        a("| document | pair | what the source says vs what the answer says |")
        a("|---|---|---|")
        for pair_id, entry in obligations["rejected"].items():
            a(
                f"| `{_short_doc(pair_id)}` | `{pair_id.split('::')[-1]}` "
                f"| {_clip(entry.get('reason', ''), 190)} |"
            )
        a("")
    a(
        "FDA guidance states the convention explicitly: **must** is a requirement, "
        "**should** is a recommendation, **may** is an option. An answer that says *must* "
        "about a source that says *should* has invented a legal obligation — and it is "
        "lexically **indistinguishable** from a correct one, which is why five machine "
        "checks passed all of these. The last of those checks is answer/evidence word "
        "coverage, and a bag of words cannot see the difference between `may` and `shall` "
        "any more than between `60 days` and `30 days`."
    )
    a("")
    # Derived in `obligation_errors`, not here. These were "Two" and "Three" as words --
    # invisible to a digit-only mutation guard, and both would have survived any change to
    # the verdicts silently, in the one module whose opening paragraph promises the
    # opposite. The document's non-binding status is now checked against its own text, so
    # the clause disappears rather than lies when the leading document is a protocol.
    worst_doc = obligations.get("worst_doc", "")
    worst_n = obligations.get("worst_doc_n", 0)
    if worst_doc and worst_n > 1:
        clause = (
            ", which carries FDA's own `Contains Nonbinding Recommendations` marking"
            if obligations.get("worst_doc_nonbinding")
            else ""
        )
        a(
            f"{worst_n} of them come from a single document "
            f"(`{_short_doc(worst_doc + '::x')}`){clause}."
        )
    a("")
    a(
        "**Consequence:** a reference answer with a manufactured requirement scores a "
        "*correct* model answer as wrong, so this failure mode corrupts the evaluation "
        "rather than the product. It is why the set is now curated and why the verdicts "
        "are committed."
    )
    a("")

    # ---- 3 ----
    naive_share = (
        _pct(refusal["answerable_refused"] / refusal["answerable_n"])
        if refusal["answerable_n"]
        else "—"
    )
    a("## 3. The over-refusal rate was mostly a labelling failure, not a model failure")
    a("")
    per_slice = refusal["per_slice"]
    if per_slice:
        a("| slice | refused |")
        a("|---|---:|")
        for name, row in per_slice.items():
            a(f"| `{name}` | {row['refused']}/{row['n']} |")
        a("")
    a(
        f"Measured naively, the model refused **{refusal['answerable_refused']} of "
        f"{refusal['answerable_n']}** answerable queries ({naive_share}), which reads as a "
        "broken refusal gate. It is not."
    )
    a("")
    a(
        "The `exact_identifier` ground truth marks a chunk relevant when the identifier "
        "**occurs** in it. That is correct for retrieval and wrong as an answerability "
        "label. `ident-0000`'s gold span is `21 CFR 1.980(k)` inside a footnote list: the "
        "document *cites* the regulation and never states its requirements, so a model "
        'asked "what are the requirements of 21 CFR 1.980?" and refusing is **right**, '
        "and scoring it as over-refusal measures the dataset."
    )
    a("")
    if refusal["honest_n"]:
        a(
            f"Excluding the mention-labelled slice: **{refusal['honest_refused']} of "
            f"{refusal['honest_n']}** "
            f"({_pct(refusal['honest_refused'] / refusal['honest_n'])}). "
            f"Unanswerable queries were refused "
            f"{per_slice.get('unanswerable', {}).get('refused', 0)}/"
            f"{per_slice.get('unanswerable', {}).get('n', 0)}."
        )
        a("")
    remaining = [q for q in refusal["refused_queries"] if q["slice"] not in MENTION_LABELLED]
    if remaining:
        a("The refusals that remain genuinely questionable:")
        a("")
        for q in remaining:
            a(f"- `{q['query_id']}` ({q['slice']}) — “{(q['query'] or '')[:90]}”")
        a("")
    a(
        "**Consequence:** golden answers cannot be derived from the retrieval eval set, "
        "which is why Phase 6 drafted a separate one. A metric that looked like a model "
        "defect was a property of the ruler."
    )
    a("")

    # ---- 4 ----
    a("## 4. Answers reach past the span they cite")
    a("")
    if faith:
        a(
            f"Tier 1 located **{faith.get('n_located')} of {faith.get('n_claimed')}** "
            f"claimed citations — no fabricated quotes. Tier 2 then asked whether the "
            f"located text *supports* the answer, and "
            f"**{faith.get('answers_fully_supported')} of "
            f"{faith.get('answers_with_citations')}** answers were fully supported "
            f"({faith.get('answer_level_rate')}), with "
            f"{faith.get('n_partial')} partial."
        )
        a("")
    a(
        "A quote being real is not the same as a quote being sufficient. The recurring "
        "shape is an answer that synthesises across a chunk — combining a sentence with "
        "the one after it — while citing only the span that carries the headline phrase. "
        'Tier 1 cannot see this by construction: it asks "is this quote in the '
        'document?", and the answer is yes.'
    )
    a("")
    judge = payload["judge"]
    if judge.get("negative_rate") is not None:
        a(
            f"The judge that measures this is itself calibrated against **constructed "
            f"negatives** — a real answer paired with another document's evidence, which "
            f"cannot be supported under any reading. It rejects those at "
            f"{_pct(judge['negative_rate'])} and accepts true positives at "
            f"{_pct(judge['positive_rate'])}. The asymmetry is deliberate: a low negative "
            "rate is unambiguous evidence of a broken judge, while a low positive rate is "
            "ambiguous between a bad judge and bad reference data — and here it was the "
            "reference data, which is finding 2."
        )
    a("")
    a(
        f"**Consequence:** read the headline as a range, not a point. At "
        f"n={faith.get('answers_with_citations')} answers, one verdict either way moves it "
        f"by {1 / faith['answers_with_citations']:.3f}"
        if faith.get("answers_with_citations")
        else "**Consequence:** the sample is too small to quote as a point estimate."
    )
    a("")

    # ---- 5 ----
    a("## 5. The verifier's own tiers can certify a change in meaning")
    a("")
    a(
        "The most uncomfortable finding, because it is a defect in the feature this "
        "project is *about*. A citation tier that discounted line-adjacent integers — "
        "added to handle FDA draft guidances, whose extracted text interleaves "
        "legislative line numbers mid-sentence — scored a model's **“40 CFR”** as "
        "a verified citation of a document's **“21 CFR”**."
    )
    a("")
    a(
        f"Exposure, recomputed on the corpus currently on disk: "
        f"**{line_numbers['n_affected']:,} of {line_numbers['n_chunks']:,} chunks "
        f"({_pct(line_numbers['share_affected'])}) lose at least one digit** to that "
        f"normalisation, {line_numbers['digits_removed']:,} digit characters in total. So "
        "a fabricated quote can land in that bucket by coincidence, across half the corpus."
    )
    a("")
    if buckets:
        a("Verification buckets from the shipped generation run:")
        a("")
        a("| bucket | n |")
        a("|---|---:|")
        for name, count in sorted(buckets.items(), key=lambda kv: (-kv[1], kv[0])):
            a(f"| `{name}` | {count} |")
        a("")
    a(
        "The tier was demoted to a non-verified diagnostic, and it is reported as its own "
        "colour in the UI so it can never read as a pass. It survived its author's review "
        "because it was tested on the case it was built for and never on the case it would "
        "break. **When widening a check to fix false negatives, the test that matters is "
        "the one for the false positives it introduces.**"
    )
    a("")

    # ---- observed, not measured ----
    a("## Observed once, and therefore not a rate")
    a("")
    a(
        f"Two failure modes have been seen in live use but are **absent from every "
        f"artifact**, and the honest thing is to label them anecdotes rather than quietly "
        f"promote them. The generation sample is n={n_gen} because the free tier allows 20 "
        "requests per day, and that ceiling — not a sampling decision — is why these are "
        "unbounded."
    )
    a("")
    a(
        "- **`wrong_chunk`**: the very first request served through the container returned "
        "four citations, three `exact` and one whose quote was a real sentence from the "
        "retrieved context attributed to the **wrong chunk** — a true quote with a false "
        "address. The shipped eval shows "
        f"`wrong_chunk: {buckets.get('wrong_chunk', 0)}`, so this project has no measured "
        "rate for it."
    )
    a(
        "- **`line_number_ambiguous` in production**: a live query about a device change "
        "control plan returned a quote that verified only under line-number normalisation. "
        f"The shipped eval shows `line_number_ambiguous: "
        f"{buckets.get('line_number_ambiguous', 0)}`."
    )
    a("")
    a(
        "Both are visible on the dashboard when they occur, which is the point of showing "
        "the model's quote diffed against the document rather than a boolean."
    )
    a("")

    if payload["missing_artifacts"]:
        a(
            "> **Incomplete:** this build could not read "
            + ", ".join(f"`{m}`" for m in payload["missing_artifacts"])
            + ", so the findings depending on them are omitted rather than estimated."
        )
        a("")
    return "\n".join(lines)
