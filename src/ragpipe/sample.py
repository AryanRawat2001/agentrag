"""Deterministic stratified sampling.

We choose ~160 documents out of ~2,800 candidates, and the choice decides which
failure modes the project can demonstrate at all. Sampling is therefore
deliberate rather than uniform:

  1. Draft/final pairs are force-included first. The "which version is
     authoritative?" question needs both revisions of the same guidance present;
     a uniform sample would essentially never produce a matched pair.
  2. The largest available protocols are force-included, to stress-test text
     extraction on documents in the hundreds of pages.
  3. Everything else is filled by round-robin across strata.

Each sampler seeds its own RNG from `seed` plus a per-source salt, so the two
samples are independent: changing --fda-n does not perturb the protocol sample.
A single shared Random would couple them, which quietly breaks reproducibility
whenever either parameter changes.

**Reproducibility caveat, stated plainly:** the seed pins *our* choices, not the
upstream indexes. Both sources are live, so a rerun months later can see a
different candidate pool. `corpus/manifest.jsonl` plus its sha256 pins is what
actually reproduces a corpus; the seed only reproduces the selection given an
unchanged pool.
"""

from __future__ import annotations

import random
from collections import defaultdict

from ragpipe.models import SourceDoc
from ragpipe.sources.fda import normalise_title

# Share of the FDA sample reserved for matched draft/final pairs. Expressed in
# documents, not titles — each pair contributes two documents, and conflating the
# two is how this ended up at double its documented size once already.
PAIR_SHARE = 0.15

# Force-include this many of the largest protocols, purely to exercise
# extraction on very long documents.
#
# This was previously labelled a "scan detector test case" on the theory that
# large files are scans. That was wrong: the two largest protocols (29.5 MB and
# 33.0 MB) are digital-native documents full of figures, and both classified as
# `digital_native`. Byte size is not a scan proxy. Deterministic coverage of the
# image_only and mixed classification paths belongs in a unit test over synthetic
# PDFs, not in corpus roulette.
N_LARGE_DOCS = 2


def _era(iso_date: str | None) -> str:
    """Decade bucket for stratification. Tolerates malformed dates."""
    if not iso_date or len(iso_date) < 4 or not iso_date[:4].isdigit():
        return "undated"
    year = int(iso_date[:4])
    if year < 2000:
        return "pre-2000"
    if year < 2010:
        return "2000s"
    if year < 2020:
        return "2010s"
    return "2020s"


def _round_robin(
    strata: dict[object, list[SourceDoc]], target: int, rng: random.Random
) -> list[SourceDoc]:
    """Take one document from each stratum in turn until `target` is reached.

    Two details that a naive implementation gets wrong:

      * **Stratum order is shuffled, not sorted.** When `target < len(strata)` not
        every stratum can be represented, so somebody gets nothing. Iterating a
        sorted key list means the *alphabetically last* strata are always the ones
        starved — with `--fda-n 40` over 58 strata that silently yielded a
        CBER/CDER/CDRH-only corpus and nothing from Human Foods Program or ORA.
        Shuffling makes the choice unbiased; the RNG is seeded, so it stays
        reproducible.
      * **The start point rotates each round.** The final round is usually partial,
        and without rotation the same strata always win that partial round,
        skewing counts by one.

    Over-samples small strata relative to their share of the corpus, which is
    intended: a center holding 60 documents still needs eval representation, and
    proportional sampling would give it almost none.
    """
    picked: list[SourceDoc] = []
    pools = {k: list(v) for k, v in strata.items()}
    keys = sorted(pools, key=str)
    rng.shuffle(keys)
    if not keys:
        return picked

    round_no = 0
    while len(picked) < target and any(pools.values()):
        offset = round_no % len(keys)
        for key in keys[offset:] + keys[:offset]:
            if len(picked) >= target:
                break
            if pools[key]:
                picked.append(pools[key].pop())
        round_no += 1
    return picked


def sample_fda(docs: list[SourceDoc], target: int, seed: int) -> list[SourceDoc]:
    """Sample FDA guidance, prioritising draft/final pairs of the same document.

    Never returns more than `target` documents.
    """
    rng = random.Random(f"{seed}:fda")
    pool = [d for d in docs if d.url]
    if target <= 0:
        return []

    # --- 1. draft/final pairs -------------------------------------------------
    by_title: dict[str, dict[str, list[SourceDoc]]] = defaultdict(lambda: defaultdict(list))
    for d in pool:
        if d.status in {"Draft", "Final"}:
            by_title[normalise_title(d.title)][d.status].append(d)

    paired_titles = [
        t for t, groups in by_title.items() if groups.get("Draft") and groups.get("Final")
    ]
    rng.shuffle(paired_titles)

    # Budget in documents, converted to titles. Bounded by target so a small
    # --fda-n cannot overshoot.
    pair_doc_budget = min(target, round(target * PAIR_SHARE))
    n_titles = min(len(paired_titles), pair_doc_budget // 2)

    selected: list[SourceDoc] = []
    selected_ids: set[str] = set()

    for title in paired_titles[:n_titles]:
        for status in ("Draft", "Final"):
            doc = by_title[title][status][0]
            if doc.doc_id not in selected_ids and len(selected) < target:
                doc.extra["sample_reason"] = "draft_final_pair"
                doc.extra["pair_key"] = title
                selected.append(doc)
                selected_ids.add(doc.doc_id)

    # --- 2. stratified remainder ---------------------------------------------
    strata: dict[object, list[SourceDoc]] = defaultdict(list)
    for d in pool:
        if d.doc_id in selected_ids:
            continue
        strata[(d.center or "unknown", d.status or "unknown", _era(d.issue_date))].append(d)
    for bucket in strata.values():
        rng.shuffle(bucket)

    for doc in _round_robin(strata, target - len(selected), rng):
        doc.extra["sample_reason"] = "stratified_center_status_era"
        selected.append(doc)

    return selected[:target]


def sample_ctgov(docs: list[SourceDoc], target: int, seed: int) -> list[SourceDoc]:
    """Sample protocols, spread across sponsor class.

    Never returns more than `target` documents.
    """
    rng = random.Random(f"{seed}:ctgov")
    pool = [d for d in docs if d.url]
    if target <= 0:
        return []

    selected: list[SourceDoc] = []
    selected_ids: set[str] = set()

    # --- 1. largest documents, for extraction stress-testing -----------------
    by_size = sorted(pool, key=lambda d: -(d.extra.get("declared_bytes") or 0))
    for doc in by_size[: min(N_LARGE_DOCS, target)]:
        doc.extra["sample_reason"] = "largest_available_document"
        selected.append(doc)
        selected_ids.add(doc.doc_id)

    # --- 2. stratified by sponsor class --------------------------------------
    # Industry, academic, and NIH sponsors write structurally different
    # protocols; that variance is why this source is in the corpus at all.
    #
    # Note: large documents are NOT excluded here. A previous version filtered
    # them out of the remainder pool, which removed candidates while the
    # force-include above contributed nothing it claimed to.
    strata: dict[object, list[SourceDoc]] = defaultdict(list)
    for d in pool:
        if d.doc_id not in selected_ids:
            strata[d.extra.get("sponsor_class") or "UNKNOWN"].append(d)
    for bucket in strata.values():
        rng.shuffle(bucket)

    for doc in _round_robin(strata, target - len(selected), rng):
        doc.extra["sample_reason"] = "stratified_sponsor_class"
        selected.append(doc)

    return selected[:target]
