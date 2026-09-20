"""Phase 0 command line: manifest -> fetch -> stats.

Three separate commands rather than one pipeline, because each stage has a
different failure mode and a different cost. Discovery is two cheap index calls;
fetching is ~160 PDF downloads against public infrastructure; inspection is
CPU-bound. Keeping them separate means a failure in one doesn't force redoing the
others, and the intermediate artifacts stay inspectable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragpipe import (
    CHUNKS_DIR,
    CORPUS_DIR,
    CURATION_VERDICTS_PATH,
    DATA_DIR,
    EVALSET_PATH,
    EXTRACT_INDEX_PATH,
    EXTRACTED_DIR,
    FETCHED_PATH,
    GOLDEN_PATH,
    MANIFEST_PATH,
    RAW_DIR,
    REPORTS_DIR,
    VECTORS_DIR,
    bench,
    chunk_report,
    chunking,
    dedup,
    embedding,
    eval_report,
    evalset,
    extract_report,
    fusion,
    gen_eval,
    generation,
    metrics,
    pdfcheck,
    rerank,
    retrieval,
    vectorstore,
)
from ragpipe import dense as dense_mod
from ragpipe import extract as extract_mod
from ragpipe import golden as golden_mod
from ragpipe import judge as judge_mod
from ragpipe import stats as stats_mod
from ragpipe.models import (
    FetchedDoc,
    load_fetched,
    load_manifest,
    read_jsonl,
    write_jsonl,
    write_jsonl_dicts,
)
from ragpipe.net import HashMismatch, PoliteClient
from ragpipe.sample import sample_ctgov, sample_fda
from ragpipe.sources import ctgov, fda

# Flush partial results to disk this often, so an interruption at document 150
# does not discard the first 149 downloads.
CHECKPOINT_EVERY = 10


def cmd_manifest(args: argparse.Namespace) -> int:
    with PoliteClient(min_interval=args.min_interval) as client:
        print("Fetching FDA guidance index ...", flush=True)
        fda_pool = fda.discover(client)
        print(f"  {len(fda_pool)} guidance documents with a PDF link", flush=True)

        print("Fetching ClinicalTrials.gov studies with attached protocols ...", flush=True)
        ctgov_pool = ctgov.discover(client, max_studies=args.ctgov_pool)
        print(f"  {len(ctgov_pool)} protocol documents across {args.ctgov_pool} studies scanned")

    # Each sampler seeds its own RNG, so --fda-n cannot perturb the protocol sample.
    fda_sample = sample_fda(fda_pool, args.fda_n, args.seed)
    ctgov_sample = sample_ctgov(ctgov_pool, args.ctgov_n, args.seed)
    selected = fda_sample + ctgov_sample

    pairs = len({d.extra["pair_key"] for d in fda_sample if d.extra.get("pair_key")})

    print()
    print(f"Sampled {len(fda_sample)} FDA + {len(ctgov_sample)} protocols = {len(selected)} docs")
    print(f"  draft/final pairs included:  {pairs} ({2 * pairs} documents)")
    print(f"  seed:                        {args.seed}")

    if args.dry_run:
        print("\n--dry-run: manifest not written")
        return 0

    # Preserve sha256 pins already established for documents that survive
    # re-sampling, so re-running `manifest` does not silently discard integrity
    # pins for documents we have already fetched.
    existing = {d.doc_id: d.sha256 for d in load_manifest(MANIFEST_PATH) if d.sha256}
    carried = 0
    for doc in selected:
        if doc.sha256 is None and doc.doc_id in existing:
            doc.sha256 = existing[doc.doc_id]
            carried += 1

    write_jsonl(MANIFEST_PATH, selected)
    print(f"\nWrote {MANIFEST_PATH.relative_to(CORPUS_DIR.parent)}")
    if carried:
        print(f"  carried forward {carried} existing sha256 pins")
    unpinned = sum(1 for d in selected if not d.sha256)
    if unpinned:
        print(f"  {unpinned} documents not yet pinned — `make fetch` will pin them")
    return 0


def _inspect_into(doc_id: str, dest: Path, sha: str | None, size: int | None) -> FetchedDoc:
    """Run PDF inspection and package the result as a FetchedDoc."""
    summary = pdfcheck.inspect(dest)
    fields = {k: v for k, v in summary.items() if k in FetchedDoc.__dataclass_fields__}
    return FetchedDoc(
        doc_id=doc_id,
        ok=True,
        path=str(dest.relative_to(DATA_DIR.parent)),
        sha256=sha,
        bytes=size,
        **fields,
    )


def cmd_fetch(args: argparse.Namespace) -> int:
    manifest = load_manifest(MANIFEST_PATH)
    if not manifest:
        print(f"No manifest at {MANIFEST_PATH}. Run `make manifest` first.", file=sys.stderr)
        return 1

    results: dict[str, FetchedDoc] = load_fetched(FETCHED_PATH)
    by_id = {d.doc_id: d for d in manifest}
    manifest_dirty = False

    if args.reinspect:
        todo = [d for d in manifest if (RAW_DIR / d.source / f"{d.doc_id}.pdf").exists()]
        print(f"--reinspect: re-running PDF inspection on {len(todo)} local files, no downloads")
    elif args.refetch:
        todo = list(manifest)
        print(f"--refetch: re-downloading all {len(todo)} documents")
    else:
        todo = [d for d in manifest if d.doc_id not in results]
        print(
            f"{len(manifest)} in manifest, {len(todo)} to fetch "
            f"({len(manifest) - len(todo)} already done)"
        )

    def flush() -> None:
        write_jsonl(FETCHED_PATH, list(results.values()))
        if manifest_dirty:
            write_jsonl(MANIFEST_PATH, list(by_id.values()))

    try:
        with PoliteClient(min_interval=args.min_interval) as client:
            for i, doc in enumerate(todo, 1):
                dest = RAW_DIR / doc.source / f"{doc.doc_id}.pdf"
                label = f"[{i}/{len(todo)}] {doc.doc_id}"

                try:
                    if args.reinspect:
                        prior = results.get(doc.doc_id)
                        sha = doc.sha256 or (prior.sha256 if prior else None)
                        size = dest.stat().st_size
                    else:
                        sha, size = client.download(
                            doc.url,
                            dest,
                            pinned_sha256=doc.sha256,
                            # --refetch must actually hit the network. Leaving the
                            # skip in place made the flag a silent no-op that only
                            # re-ran inspection.
                            skip_if_present=not args.refetch,
                        )
                        if doc.sha256 is None:
                            doc.sha256 = sha  # first fetch establishes the pin
                            manifest_dirty = True

                    results[doc.doc_id] = _inspect_into(doc.doc_id, dest, sha, size)
                    r = results[doc.doc_id]
                    print(
                        f"{label}  {r.text_layer:14s} "
                        f"pages={r.pages if r.pages is not None else '?':>4} "
                        f"chars={r.text_chars or 0:>8,} "
                        f"{(size or 0) / 1e6:5.1f}MB",
                        flush=True,
                    )

                except HashMismatch as exc:
                    if args.allow_drift:
                        # Re-pin deliberately: fetch again with no expectation.
                        sha, size = client.download(
                            doc.url, dest, pinned_sha256=None, skip_if_present=False
                        )
                        doc.sha256 = sha
                        manifest_dirty = True
                        results[doc.doc_id] = _inspect_into(doc.doc_id, dest, sha, size)
                        print(f"{label}  RE-PINNED (content changed upstream)", flush=True)
                    else:
                        results[doc.doc_id] = FetchedDoc(
                            doc_id=doc.doc_id, ok=False, error=str(exc)
                        )
                        print(f"{label}  HASH MISMATCH — upstream document changed", flush=True)

                except Exception as exc:  # noqa: BLE001 - one bad doc shouldn't stop the run
                    results[doc.doc_id] = FetchedDoc(
                        doc_id=doc.doc_id, ok=False, error=f"{type(exc).__name__}: {exc}"
                    )
                    print(f"{label}  FAILED  {type(exc).__name__}: {exc}", flush=True)

                if i % CHECKPOINT_EVERY == 0:
                    flush()
    finally:
        # Always persist. Losing this file means the next run has no sha256 pins
        # from fetch results and no record of what already succeeded.
        flush()

    ok = sum(1 for r in results.values() if r.ok)
    failed = [r for r in results.values() if not r.ok]
    print(f"\n{ok}/{len(results)} OK -> {FETCHED_PATH.name}")
    if manifest_dirty:
        print(f"Updated sha256 pins in {MANIFEST_PATH.relative_to(CORPUS_DIR.parent)}")
    for r in failed:
        print(f"  FAILED {r.doc_id}: {r.error}", file=sys.stderr)
    return 1 if failed else 0


def cmd_extract(args: argparse.Namespace) -> int:
    """PDF -> normalised text, structure, and identifiers.

    Incremental by content hash: a document is re-extracted only when its
    manifest sha256 differs from the hash recorded in its extraction output.
    """
    manifest = load_manifest(MANIFEST_PATH)
    if not manifest:
        print(f"No manifest at {MANIFEST_PATH}. Run `make manifest` first.", file=sys.stderr)
        return 1

    fetched = load_fetched(FETCHED_PATH)
    # Only indexable documents are extracted. Quarantined scans would contribute
    # OCR noise, which is the whole reason Phase 0 detects them.
    todo = [
        d
        for d in manifest
        if (r := fetched.get(d.doc_id)) and r.ok and pdfcheck.is_indexable(r.text_layer)
    ]
    skipped_unindexable = len(manifest) - len(todo)

    summaries: list[dict] = []
    failures: list[dict[str, str]] = []
    reused = 0

    for i, doc in enumerate(todo, 1):
        pdf_path = RAW_DIR / doc.source / f"{doc.doc_id}.pdf"
        out_path = extract_mod.output_path(EXTRACTED_DIR, doc)
        label = f"[{i}/{len(todo)}] {doc.doc_id}"

        if not args.force and not extract_mod.needs_extraction(out_path, doc.sha256):
            with out_path.open(encoding="utf-8") as fh:
                cached = json.load(fh)
            cached.pop("text", None)
            cached.pop("identifiers", None)
            cached["n_sections"] = len(cached.get("sections") or [])
            cached["pages"] = len(cached.get("pages") or [])
            summaries.append(cached)
            reused += 1
            continue

        if not pdf_path.exists():
            failures.append({"doc_id": doc.doc_id, "error": "PDF missing on disk"})
            print(f"{label}  MISSING PDF", flush=True)
            continue

        try:
            extracted = extract_mod.extract_document(doc, pdf_path)
        except Exception as exc:  # noqa: BLE001 - one bad PDF shouldn't stop the run
            failures.append({"doc_id": doc.doc_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{label}  FAILED  {type(exc).__name__}: {exc}", flush=True)
            continue

        extract_mod.write_document(extracted, out_path)
        summaries.append(extracted.summary())
        print(
            f"{label}  {extracted.structure_source:9s} "
            f"sections={len(extracted.sections):>4} "
            f"chars={extracted.n_chars:>8,} "
            f"cfr={extracted.identifier_counts.get('cfr', 0):>4}",
            flush=True,
        )

    write_jsonl_dicts(EXTRACT_INDEX_PATH, summaries)
    payload = extract_report.build(summaries, failures)
    report = REPORTS_DIR / "extraction_report.md"
    js = REPORTS_DIR / "extraction_stats.json"
    extract_report.write(payload, report, js)

    print()
    print(
        f"{len(summaries)} documents extracted ({reused} reused from cache), {len(failures)} failed"
    )
    print(f"{skipped_unindexable} skipped as not indexable")
    ids = payload["identifiers"][payload["primary_kind"]]
    print(
        f"primary identifier `{payload['primary_kind']}`: "
        f"{ids['docs_containing']} docs, "
        f"{ids['distinct_section_level']:,} distinct sections / "
        f"{ids['distinct_with_subsection']:,} with subsection"
    )
    print(f"\nWrote {report}\n      {js}")
    return 1 if failures else 0


def _is_quarantined(extracted: dict) -> bool:
    """Whether an extracted document is excluded from every index.

    One predicate, used by both `cmd_chunk` and `cmd_sweep`. It reads the stored
    verdict **and** re-applies the threshold to the stored ratio, because the two can
    disagree: `text_quality` is computed once at extraction and extraction is cached by
    content hash, so lowering `CHARACTER_SPACING_THRESHOLD` without `extract --force`
    leaves stale verdicts on disk. Re-checking the ratio makes the live constant
    authoritative; keeping the verdict check means a future quality signal that is not
    ratio-based still quarantines.

    The verdict check tests membership in `pdfcheck.QUARANTINE_QUALITIES` rather than a
    single hardcoded class. It previously compared against `"character_spaced"` alone,
    so `broken_encoding` documents — the two with Caesar-shifted glyph-code text layers
    — passed straight into the index while the constant declared them unindexable.

    These were two separate expressions in `cmd_chunk` and `cmd_sweep`, and they could
    disagree — putting different corpora behind two reports that claim to describe the
    same one.
    """
    if extracted.get("text_quality") in pdfcheck.QUARANTINE_QUALITIES:
        return True
    # Re-apply every live threshold to the stored ratios, so lowering a constant takes
    # effect without `extract --force`. `space_collapsed` is intentionally absent from
    # `QUARANTINE_QUALITIES` and so is not re-checked here: its text is degraded but
    # readable and its citations verify at exact offsets.
    if extracted.get("single_char_token_share", 0.0) >= pdfcheck.CHARACTER_SPACING_THRESHOLD:
        return True
    return bool(extracted.get("control_char_share", 0.0) >= pdfcheck.CONTROL_CHAR_THRESHOLD)


def cmd_chunk(args: argparse.Namespace) -> int:
    """Chunk every extracted document with each strategy, then cluster duplicates."""
    index_rows = list(read_jsonl(EXTRACT_INDEX_PATH))
    if not index_rows:
        print(
            f"No extraction index at {EXTRACT_INDEX_PATH}. Run `make extract` first.",
            file=sys.stderr,
        )
        return 1

    strategies = args.strategies or sorted(chunking.STRATEGIES)
    unknown = [s for s in strategies if s not in chunking.STRATEGIES]
    if unknown:
        deferred = [s for s in unknown if s in chunking.DEFERRED_STRATEGIES]
        for s in deferred:
            print(f"{s!r} is not implemented: {chunking.DEFERRED_STRATEGIES[s]}", file=sys.stderr)
        print(f"unknown strategies: {[s for s in unknown if s not in deferred]}", file=sys.stderr)
        return 1

    per_strategy: list[dict] = []
    # Real section offsets, for measuring boundary agreement against the documents
    # rather than against the chunk output being measured.
    section_spans: dict[str, set[int]] = {}

    quarantined: list[tuple[str, dict]] = []

    for strategy in strategies:
        chunks: list[dict] = []
        # Built once per strategy and only when needed, so `fixed` and `structural`
        # never pay for a model load they do not use.
        embedder = (
            embedding.LocalEmbedder(args.embed_model)
            if strategy in chunking.EMBEDDING_STRATEGIES
            else None
        )
        quarantined = []
        for row in index_rows:
            path = EXTRACTED_DIR / row["source"] / f"{row['doc_id']}.json"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as fh:
                extracted = json.load(fh)

            # Quarantine a document whose text layer cannot be read, on the same
            # reasoning as a scanned PDF: text that cannot be read cannot be retrieved,
            # and indexing it silently puts a floor under every retrieval metric that is
            # indistinguishable from the retriever underperforming. Three documents in
            # this corpus qualify across two defect classes — see `pdfcheck`.
            if _is_quarantined(extracted):
                quarantined.append((extracted["doc_id"], extracted))
                continue
            section_spans.setdefault(
                extracted["doc_id"],
                {
                    b
                    for sec in (extracted.get("sections") or [])
                    for b in (sec["start"], sec["end"])
                },
            )
            chunks.extend(
                c.as_dict()
                for c in chunking.build_chunks(
                    extracted,
                    strategy,
                    target=args.target_chars,
                    overlap=args.overlap_chars,
                    embedder=embedder,
                )
            )

        for doc_id, extracted in quarantined:
            # Name the defect class and print the ratio that triggered it. Reporting a
            # bare "text quality flagged at extraction" for anything that was not
            # character-spaced hid *which* defect fired -- and once `broken_encoding`
            # existed, that was most of them. A document flagged by a stored verdict
            # alone still must not print a ratio that contradicts its own reason.
            single = extracted.get("single_char_token_share", 0.0)
            control = extracted.get("control_char_share", 0.0)
            if control >= pdfcheck.CONTROL_CHAR_THRESHOLD:
                reason = (
                    f"text layer is raw glyph codes, no usable ToUnicode map "
                    f"({control:.1%} control characters)"
                )
            elif single >= pdfcheck.CHARACTER_SPACING_THRESHOLD:
                reason = f"text layer is character-spaced ({single:.1%} single-character tokens)"
            else:
                reason = (
                    f"text quality flagged at extraction as "
                    f"{extracted.get('text_quality', 'unknown')!r}"
                )
            print(
                f"[{strategy}] quarantined {doc_id}: {reason} — not indexable",
                file=sys.stderr,
            )
        print(f"[{strategy}] {len(chunks):,} chunks — clustering duplicates ...", flush=True)
        duplicate_of, raw_clusters = dedup.find_duplicates(
            [(c["chunk_id"], c["text"]) for c in chunks], threshold=args.dup_threshold
        )
        for c in chunks:
            c["duplicate_of"] = duplicate_of.get(c["chunk_id"])

        chunk_docs = {c["chunk_id"]: c["doc_id"] for c in chunks}
        chunk_text = {c["chunk_id"]: c["text"] for c in chunks}
        clusters = chunk_report.cluster_metadata(raw_clusters, chunk_docs, chunk_text)

        out_path = CHUNKS_DIR / f"{strategy}.jsonl"
        write_jsonl_dicts(out_path, chunks, sort_key="chunk_id")
        (CHUNKS_DIR / f"{strategy}.clusters.json").write_text(
            json.dumps(clusters, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        stats = chunk_report.build_strategy_stats(
            strategy, chunks, clusters, duplicate_of, section_spans
        )
        per_strategy.append(stats)
        print(
            f"[{strategy}] {stats['n_chunks']:,} chunks, "
            f"{stats['duplicates']['chunks_marked_duplicate']:,} duplicates in "
            f"{stats['duplicates']['clusters']} clusters "
            f"({stats['duplicates']['cross_document_clusters']} cross-document)",
            flush=True,
        )

    # The report is a whole-corpus artifact, so it is assembled from every strategy
    # with chunks on disk — not only the ones re-chunked in this invocation.
    #
    # `--strategies semantic` previously rewrote the committed report to cover
    # `semantic` alone, silently dropping `fixed` and `structural` from the
    # comparison table. The chunk files survived, so nothing looked broken; the
    # artifact just quietly stopped being a comparison. Re-deriving the missing
    # strategies from disk makes a partial re-chunk safe.
    done = {s["strategy"] for s in per_strategy}
    for other in sorted(set(chunking.STRATEGIES) - done):
        path = CHUNKS_DIR / f"{other}.jsonl"
        clusters_path = CHUNKS_DIR / f"{other}.clusters.json"
        if not (path.exists() and clusters_path.exists()):
            continue
        print(f"[{other}] reusing chunks on disk for the report", flush=True)
        existing = retrieval.load_chunks(path)
        clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
        duplicate_of = {c["chunk_id"]: c["duplicate_of"] for c in existing if c.get("duplicate_of")}
        per_strategy.append(
            chunk_report.build_strategy_stats(
                other, existing, clusters, duplicate_of, section_spans
            )
        )

    per_strategy.sort(key=lambda s: s["strategy"])
    payload = chunk_report.build(per_strategy, chunking.DEFERRED_STRATEGIES)
    report = REPORTS_DIR / "chunking_report.md"
    js = REPORTS_DIR / "chunking_stats.json"
    chunk_report.write(payload, report, js)
    print(f"\nWrote {report}\n      {js}")
    return 0


def _load_extracted_docs() -> list[dict]:
    """Load extracted documents without their full text.

    The eval set needs identifiers, sections, titles, and lengths — not the text
    itself, which is ~12 MB and irrelevant to ground truth.
    """
    docs: list[dict] = []
    for row in read_jsonl(EXTRACT_INDEX_PATH):
        path = EXTRACTED_DIR / row["source"] / f"{row['doc_id']}.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            doc = json.load(fh)
        doc.pop("text", None)
        docs.append(doc)
    return docs


def cmd_evalset(args: argparse.Namespace) -> int:
    """Build the golden retrieval set from extracted documents, with exact ground truth.

    Built from documents, never from chunks: ground truth is character spans, so it
    must not depend on how the corpus happened to be chunked.

    **Quarantined documents are excluded from the pool.** Ground truth being
    chunking-independent does not make it quarantine-independent: a span in a document
    that no strategy indexes resolves to no chunk under every retriever, so the query
    is unscorable by construction and silently shrinks the slice it belongs to. Before
    this filter, 7 of 180 answerable queries pointed at the three unreadable documents
    — 6 `section_lookup` and 1 `title_lookup` — and regenerating the set made that
    worse rather than better, because quarantining more documents means more queries
    aimed at them. The same predicate the chunker uses decides it here, so the eval set
    and the index can never disagree about which corpus exists.
    """
    all_docs = _load_extracted_docs()
    if not all_docs:
        print("No extracted documents. Run `make extract` first.", file=sys.stderr)
        return 1
    docs = [d for d in all_docs if not _is_quarantined(d)]
    excluded = len(all_docs) - len(docs)

    queries = evalset.build_all(docs, seed=args.seed, per_slice=args.per_slice)
    write_jsonl_dicts(EVALSET_PATH, [q.as_dict() for q in queries], sort_key="query_id")

    by_slice = Counter(q.slice_name for q in queries)
    print(
        f"Built {len(queries)} queries from {len(docs)} indexable documents "
        f"(seed {args.seed}); {excluded} quarantined document(s) excluded from the pool"
    )
    for name, count in sorted(by_slice.items()):
        print(f"  {name:20s} {count:>4}")
    print(f"\nWrote {EVALSET_PATH}")
    return 0


def _retrievers_for(
    chunks: list[dict], args: argparse.Namespace, truncation: dict[str, dict]
) -> Iterator[Any]:
    """Yield each retriever to score, building it only when it is reached.

    A generator rather than a list because dense indexes are expensive to build and
    hold: constructing every variant up front would embed the corpus several times
    over before scoring a single query, and keep every matrix resident. Yielding
    them one at a time means a run that is interrupted has still done real work, and
    peak memory is one index rather than all of them.
    """
    # Built once and reused as both the baseline row and the sparse leg of every
    # hybrid. Constructing it twice was ~0.8s of pure waste, and — more to the point —
    # left two call sites that a future argument change could silently desynchronise,
    # which would make the fusion rows incomparable to the baseline they are read
    # against. One object cannot drift from itself.
    sparse = retrieval.BM25Retriever(chunks, name="bm25", use_identifier_atoms=True)
    yield sparse

    if args.with_ablations:
        # Each variant isolates exactly one design decision, so a row difference is
        # attributable rather than a mix of causes.
        #
        # The tokenizer decision: identifier atoms on/off.
        yield retrieval.BM25Retriever(chunks, name="bm25 -atoms", use_identifier_atoms=False)
        # Heading modes. Three, not two: choosing `text` over `embed_text` leaves
        # the heading indexed in 41.8% of chunks, because a section's text begins
        # with its own heading. Only `strip` is heading-free.
        yield retrieval.BM25Retriever(chunks, name="bm25 heading=source", heading_mode="source")
        yield retrieval.BM25Retriever(chunks, name="bm25 heading=strip", heading_mode="strip")

    if not args.dense_models:
        return

    strategy = chunks[0]["strategy"] if chunks else "unknown"

    for model in args.dense_models:
        embedder = embedding.LocalEmbedder(model, batch_size=args.batch_size)
        cache = embedding.VectorCache.open(VECTORS_DIR / f"{model}-{strategy}", embedder.dim)
        dense = dense_mod.DenseRetriever(chunks, embedder, cache=cache, count_truncation=True)
        # Flushed on success only. An earlier `try/finally` here claimed to preserve
        # vectors "already computed" across a later failure, but there is no later
        # step that can fail: truncation counting runs before encoding, and a crash
        # inside `encode_prepared` is all-or-nothing, so the cache stays clean and
        # empty. The claim was false and the construct bought nothing. Resumability
        # is real, but at strategy/model granularity, not within one strategy.
        cache.flush()
        if dense.truncation:
            truncation[f"{strategy}/{model}"] = dense.truncation
        print(
            f"  [{model}] {len(cache):,} vectors cached "
            f"({cache.hits:,} hits, {cache.misses:,} encoded, {cache.deduped:,} deduped "
            f"= {cache.hits + cache.misses + cache.deduped:,} requested), "
            f"{dense.truncation['pct_truncated']:.1f}% of chunks truncated at "
            f"{embedder.window} tokens",
            flush=True,
        )

        yield dense

        # Fusion rows. All three methods, because the source guide conflates them:
        # vanilla RRF has no weights at all, weighted RRF weights *ranks*, and the
        # guide's "0.7/0.3" describes weighted *score* fusion. Same inputs, so a
        # row difference is the fusion method and nothing else.
        rrf_hybrid = fusion.HybridRetriever(
            [sparse, dense], method="rrf", name=f"hybrid {model} rrf", fetch_k=args.fetch_k
        )
        yield rrf_hybrid
        for w_dense in args.fusion_weights:
            weights = [1.0 - w_dense, w_dense]  # [sparse, dense]
            yield fusion.HybridRetriever(
                [sparse, dense],
                method="weighted_rrf",
                weights=weights,
                name=f"hybrid {model} wrrf(s{1 - w_dense:g}/d{w_dense:g})",
                fetch_k=args.fetch_k,
            )
            yield fusion.HybridRetriever(
                [sparse, dense],
                method="minmax",
                weights=weights,
                name=f"hybrid {model} minmax(s{1 - w_dense:g}/d{w_dense:g})",
                fetch_k=args.fetch_k,
            )

        if not args.rerank_model:
            continue

        # Reranked on top of *both* the sparse baseline and the fused hybrid.
        # Reranking only the best first stage would leave its contribution
        # entangled with fusion's: two rows over different candidate lists show
        # whether the cross-encoder adds anything fusion did not already find.
        for base in (sparse, rrf_hybrid):
            yield rerank.CrossEncoderReranker(
                base,
                chunks,
                model=args.rerank_model,
                candidate_k=args.candidate_k,
                batch_size=args.batch_size,
            )


def cmd_eval(args: argparse.Namespace) -> int:
    """Score retrievers against the golden set and write the ablation table."""
    queries = [evalset.EvalQuery(**q) for q in read_jsonl(EVALSET_PATH)]
    if not queries:
        print(f"No eval set at {EVALSET_PATH}. Run `make evalset` first.", file=sys.stderr)
        return 1

    strategies = args.chunkings or sorted(chunking.STRATEGIES)
    runs: list[dict] = []
    truncation: dict[str, dict] = {}

    for strategy in strategies:
        chunk_path = CHUNKS_DIR / f"{strategy}.jsonl"
        if not chunk_path.exists():
            print(f"skipping {strategy}: no chunks at {chunk_path}", file=sys.stderr)
            continue
        chunks = retrieval.load_chunks(chunk_path)

        # Ground truth is character spans, resolved to chunk ids here — against
        # *this* strategy's chunks. That is what makes the comparison across
        # strategies valid rather than an artifact of which set the eval was built on.
        chunks_by_doc: dict[str, list[dict]] = defaultdict(list)
        for c in chunks:
            chunks_by_doc[c["doc_id"]].append(c)

        for retriever in _retrievers_for(chunks, args, truncation):
            print(
                f"[{strategy} / {retriever.name}] scoring {len(queries)} queries "
                f"over {len(retriever):,} chunks ...",
                flush=True,
            )
            per_slice: dict[str, list[metrics.QueryResult]] = defaultdict(list)

            # A reranker only permutes its candidate list, so its recall is bounded
            # by what that list contained. Collected per query and reported per row,
            # because a rerank number read against 1.0 instead of against its own
            # ceiling credits or blames the cross-encoder for the first stage's work.
            is_rerank = isinstance(retriever, rerank.CrossEncoderReranker)
            ceilings: dict[str, list[float]] = defaultdict(list)

            for q in queries:
                hits = retriever.search(q.query, k=max(metrics.K_VALUES))
                relevant = evalset.relevant_chunks(q, chunks_by_doc)
                if is_rerank and relevant:
                    # Over the *effective* window: `candidate_k` is a floor, and the
                    # harness asks for max(K_VALUES)=20 results, so a smaller
                    # `--candidate-k` still scores 20 candidates. Measuring the
                    # ceiling over the nominal value made recall exceed it.
                    ceilings[q.slice_name].append(
                        rerank.ceiling_from_ids(
                            retriever.last_candidate_ids,
                            set(relevant),
                            retriever.effective_candidate_k,
                        )
                    )
                per_slice[q.slice_name].append(
                    metrics.QueryResult(
                        query_id=q.query_id,
                        slice_name=q.slice_name,
                        ranked_ids=[h.chunk_id for h in hits],
                        relevant_ids=frozenset(relevant),
                        top_score=hits[0].score if hits else 0.0,
                    )
                )

            # Separability compares like with like: `exact_identifier` and
            # `unanswerable` are phrased identically and differ only in whether the
            # cited regulation exists. Including section and title queries — long,
            # high-IDF strings that score highly for unrelated reasons — pushed the
            # answerable pool to 180 against 60 unanswerable, so the headline number
            # sat against a 0.75 majority-class baseline and the ranking flipped.
            like_for_like = per_slice.get("exact_identifier", []) + per_slice.get(
                "unanswerable", []
            )
            for slice_name, results in sorted(per_slice.items()):
                row = {
                    "chunking": strategy,
                    "retriever": retriever.name,
                    "slice_name": slice_name,
                    "metrics": metrics.aggregate(results),
                    "separability": metrics.separability(like_for_like)
                    if slice_name == "unanswerable"
                    else {},
                }
                if is_rerank:
                    row["candidate_k"] = retriever.candidate_k
                    if ceilings.get(slice_name):
                        row["recall_ceiling"] = sum(ceilings[slice_name]) / len(
                            ceilings[slice_name]
                        )
                runs.append(row)

    # What was actually scored on retrieval metrics, per *answerable* slice. Every run
    # reports the same per-slice count, so the first row for each slice is authoritative.
    #
    # `unanswerable` is excluded deliberately: retrieval accuracy is undefined on an
    # empty relevant set, so `metrics.aggregate` drops all 60 of those queries by design
    # and they contribute through `separability` instead. Counting them as "unscorable"
    # reported 62 missing queries when the real answer is 2 — the same wrong-population
    # mistake this accounting exists to expose.
    scored_by_slice: dict[str, int] = {}
    for r in runs:
        if r["slice_name"] == "unanswerable":
            continue
        scored_by_slice.setdefault(r["slice_name"], int(r["metrics"].get("n_queries", 0)))
    scored_total = sum(scored_by_slice.values())

    by_slice = Counter(q.slice_name for q in queries)
    summary = {
        "n_queries": len(queries),
        "n_answerable": sum(1 for q in queries if q.relevant_spans),
        # Counted separately from the eval-set file, because they can differ and the
        # difference is invisible otherwise. `metrics.aggregate` drops any query whose
        # relevant set resolves empty against the current chunks — which is correct —
        # so quarantining a document silently reduces the scored count while the file
        # still lists the query. Two `section_lookup` queries are drawn from the
        # character-spaced document and are now unscorable: the summary said 60 while
        # every result row said 58, with nothing to explain the gap.
        "n_scored": scored_total,
        "n_scored_by_slice": dict(sorted(scored_by_slice.items())),
        # Against the answerable population only, for the reason above.
        "n_unscorable": sum(1 for q in queries if q.relevant_spans) - scored_total,
        "n_unanswerable": sum(1 for q in queries if not q.relevant_spans),
        "seed": args.seed,
        "slices": [
            (name, by_slice[name], _GROUND_TRUTH_NOTE.get(name, "")) for name in sorted(by_slice)
        ],
    }
    payload = eval_report.build(runs, summary, truncation=truncation)
    report = REPORTS_DIR / "retrieval_eval.md"
    js = REPORTS_DIR / "retrieval_eval.json"
    eval_report.write(payload, report, js)
    print(f"\nWrote {report}\n      {js}")
    return 0


_GROUND_TRUTH_NOTE = {
    "exact_identifier": "spans where the cited regulation occurs",
    "section_lookup": "the named section's span (heading unique corpus-wide)",
    "title_lookup": "the whole named document (document-level task)",
    "unanswerable": "empty by construction; citation verified absent from the corpus",
}


def cmd_answer_eval(args: argparse.Namespace) -> int:
    """Phase 5: run grounded answering over a stratified slice of the golden set."""
    from ragpipe.generation import GenerationError, make_generator

    rows = list(read_jsonl(EVALSET_PATH))
    if not rows:
        print(f"No eval set at {EVALSET_PATH}. Run `make evalset` first.", file=sys.stderr)
        return 1

    # Re-judge an existing report without re-running generation. Tier 2 is a few
    # requests; generation is one per query, and the first two-tier run showed the
    # tier-2 unit was wrong -- correcting that would otherwise have cost a full
    # regeneration.
    if args.rejudge:
        existing_path = REPORTS_DIR / "generation_eval.json"
        if not existing_path.exists():
            print(f"No report at {existing_path}. Run without --rejudge first.", file=sys.stderr)
            return 1
        payload = json.loads(existing_path.read_text(encoding="utf-8"))
        result = gen_eval.load_gen_eval(payload)
        if not args.judge:
            print(
                "--rejudge needs --judge <model>: it re-runs tier 2 and nothing else.",
                file=sys.stderr,
            )
            return 1
        # A report written before Phase 6 has `checks` but no `answer_text`, so every judge
        # item would carry an empty claim -- quota spent, verdicts returned, and scored as
        # faithfulness. Refuse instead of guessing.
        answering = [o for o in result.scored() if not o.refused and o.checks]
        if answering and not any(o.answer_text for o in answering):
            print(
                "This report predates Phase 6 and stores no answer text, so tier 2 would "
                "judge empty claims. Re-run generation instead of --rejudge.",
                file=sys.stderr,
            )
            return 1
        print(
            f"[gen-eval] --rejudge: {len(result.scored())} stored outcomes, no generation",
            file=sys.stderr,
        )
        return _run_tier2(args, result, payload)

    chunk_path = CHUNKS_DIR / f"{args.chunking}.jsonl"
    if not chunk_path.exists():
        print(f"No chunks at {chunk_path}. Run `make chunk` first.", file=sys.stderr)
        return 1
    chunks = retrieval.load_chunks(chunk_path)
    chunk_index = {c["chunk_id"]: c for c in chunks}

    # BM25 alone, deliberately. It has the best refusal separability of any first
    # stage Phase 3 measured (best-threshold accuracy 0.983 on `structural`, against
    # 0.667 for min-max fusion and 0.942 for rank fusion plus reranking, all on the same
    # chunking -- an earlier version quoted min-max's 0.725, which is `semantic`), so a
    # score
    # gate calibrates against it most cleanly. Fused and reranked variants are a later
    # comparison row, not the baseline this phase's refusal numbers should rest on.
    retriever = retrieval.BM25Retriever(chunks, heading_mode=args.heading_mode)

    picked = gen_eval.stratified_sample(rows, args.per_slice)
    print(
        f"[gen-eval] {len(picked)} queries ({args.per_slice}/slice) over "
        f"{len(retriever):,} {args.chunking} chunks, top-{args.k} context",
        file=sys.stderr,
    )

    try:
        generator = make_generator(args.generator)
    except GenerationError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    def progress(i: int, row: dict, outcome: Any) -> None:
        if outcome is None or outcome.error:
            detail = outcome.error if outcome is not None else "unknown"
            print(
                f"  [{i}/{len(picked)}] {row['query_id']:<16} ERROR {detail}",
                file=sys.stderr,
                flush=True,
            )
            return
        state = "refused" if outcome.refused else f"{outcome.n_verified}/{outcome.n_claimed} cites"
        print(
            f"  [{i}/{len(picked)}] {row['query_id']:<16} {state}",
            file=sys.stderr,
            flush=True,
        )

    result = gen_eval.run_gen_eval(
        picked,
        retriever,
        chunk_index,
        generator=generator,
        k=args.k,
        refusal_threshold=args.refusal_threshold,
        progress=progress,
    )
    payload = result.to_payload()

    # Tier 2, optional and off by default. It costs a request per batch of judgments, and
    # a faithfulness number from an uncalibrated judge is worse than none — so the judge
    # is calibrated first and the run refuses to report faithfulness if it fails.
    if args.judge:
        rc = _run_tier2(args, result, payload, write=False)
        if rc:
            return rc

    md_path, json_path = gen_eval.write_gen_eval(payload, REPORTS_DIR)
    _print_gen_eval_summary(payload, md_path, json_path)
    return 0


def _print_gen_eval_summary(payload: dict, md_path: Any, json_path: Any) -> None:
    print(
        f"\nscored {payload['n_scored']}/{payload['n_queries']}"
        f" ({payload['n_errors']} errors); "
        f"mean citation precision {payload['mean_citation_precision']}; "
        f"unanswerable refusal {payload['refusal']['unanswerable']}; "
        f"separability AUC {payload['separability_auc']}",
        file=sys.stderr,
    )
    f = payload.get("faithfulness")
    if f:
        print(
            f"tier 2 ({payload.get('judge_unit', 'answer')}-level): "
            f"{f['answers_fully_supported']}/{f['answers_with_citations']} answers fully "
            f"supported = {f['answer_level_rate']}; "
            f"verdicts supported {f['n_supported']} / partial {f['n_partial']} / "
            f"unsupported {f['n_unsupported']} / unjudged {f['n_unjudged']}",
            file=sys.stderr,
        )
    print(f"wrote {md_path} and {json_path}", file=sys.stderr)


def _run_tier2(args: argparse.Namespace, result: Any, payload: dict, write: bool = True) -> int:
    """Run tier 2 over a result and attach faithfulness to `payload`."""
    if args.judge:
        from ragpipe import golden as golden_mod
        from ragpipe import judge as judge_mod
        from ragpipe.generation import GenerationError, make_generator

        items = gen_eval.judge_items_for(result, unit=args.judge_unit)
        if not items:
            print("[gen-eval] no located citations to judge", file=sys.stderr)
        else:
            if args.judge == args.generator:
                # A default is not an enforcement. `judge.py` presents "do not let one
                # model grade its own homework" as a property of the module, but `--judge`
                # is unconstrained, so `--generator X --judge X` self-graded silently.
                print(
                    f"[gen-eval] WARNING: judge and generator are both `{args.judge}` — "
                    "the model is grading its own output, and agreement will partly "
                    "measure self-consistency rather than support.",
                    file=sys.stderr,
                )
            try:
                judge_gen = make_generator(args.judge)
            except GenerationError as exc:
                print(f"{exc}", file=sys.stderr)
                return 1

            controls = judge_mod.build_control_set(
                golden_mod.load_golden(GOLDEN_PATH), args.judge_controls, args.judge_controls
            )
            all_items = controls + items
            judgments: dict[str, Any] = {}
            groups = judge_mod.batches(all_items, args.judge_batch_size)
            print(
                f"[gen-eval] tier 2: {len(items)} citations + {len(controls)} controls "
                f"in {len(groups)} request(s) via `{args.judge}`",
                file=sys.stderr,
            )
            for i, batch in enumerate(groups, 1):
                try:
                    res = judge_gen.generate(
                        judge_mod.build_prompt(batch),
                        system=judge_mod.JUDGE_SYSTEM_PROMPT,
                        schema=judge_mod.JUDGE_SCHEMA,
                        max_output_tokens=8192,
                        temperature=0.0,
                    )
                    judgments.update(judge_mod.parse_judgments(res.parse_json()))
                except GenerationError as exc:
                    print(f"  [{i}/{len(groups)}] {exc}"[:200], file=sys.stderr)
                    if "daily free-tier quota" in str(exc):
                        break
                    continue
                print(f"  [{i}/{len(groups)}] {len(batch)} judged", file=sys.stderr, flush=True)

            agreement = judge_mod.score_agreement(controls, judgments)
            payload["judge"] = args.judge
            payload["judge_calibration"] = agreement.as_dict()
            payload["judge_unit"] = args.judge_unit
            if agreement.usable:
                scorer = (
                    gen_eval.score_answer_faithfulness
                    if args.judge_unit == "answer"
                    else gen_eval.score_faithfulness
                )
                payload["faithfulness"] = scorer(result, judgments).as_dict()
            else:
                # *Delete*, not merely decline to write. On the `--rejudge` path the
                # payload is the previous report loaded from disk and already carries a
                # `faithfulness` block, so declining to overwrite it shipped a stale
                # figure beside a failed judge's calibration -- the exact opposite of what
                # the help text and the progress log both promise. Caught by the Phase 6
                # code-review gate, on the one path Phase 6c presents as a cost saving.
                payload.pop("faithfulness", None)
                print(
                    "[gen-eval] judge failed calibration — faithfulness omitted from the "
                    "report rather than reported with a caveat",
                    file=sys.stderr,
                )
            # Persist *all* items, not just the controls. Passing controls alone meant
            # the real per-answer verdicts were never written, so a correct faithfulness
            # figure could not be re-derived from disk -- the same auditability failure
            # this project keeps fixing elsewhere. Found by regenerating the report and
            # getting 0 of 8 supported.
            judge_mod.write_agreement(
                agreement,
                REPORTS_DIR / "judge_calibration.json",
                all_items,
                judgments,
                judge_model=args.judge,
                written_by="gen-eval --judge",
            )

    if write:
        md_path, json_path = gen_eval.write_gen_eval(payload, REPORTS_DIR)
        _print_gen_eval_summary(payload, md_path, json_path)
    return 0


def cmd_golden(args: argparse.Namespace) -> int:
    """Phase 6: draft golden Q&A pairs, validate them, and write the reject log.

    Documents are chosen by a deterministic stride over the sorted indexable set, so a
    calibration batch spans FDA guidances and clinical protocols rather than whichever
    documents happen to sort first. Every drafted pair is validated with the Phase 5
    tier-1 verifier before anything is written, so a hallucinated reference answer never
    reaches the curation pass.
    """
    from ragpipe import golden
    from ragpipe.generation import GenerationError, make_generator

    chunk_path = CHUNKS_DIR / f"{args.chunking}.jsonl"
    if not chunk_path.exists():
        print(f"No chunks at {chunk_path}. Run `make chunk` first.", file=sys.stderr)
        return 1
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for chunk in retrieval.load_chunks(chunk_path):
        by_doc[chunk["doc_id"]].append(chunk)

    # Built once over the whole corpus: the boilerplate check asks how many *documents*
    # contain a quote, so it needs every document, not just the ones being drafted from.
    doc_index = golden.build_document_index(c for cs in by_doc.values() for c in cs)
    existing = golden.load_golden(GOLDEN_PATH) if not args.overwrite else []
    seen_ids = {p.pair_id for p in existing}

    # Append mode draws only from documents not already covered. Without this the stride
    # re-picks the same documents on every run and every new pair collides with an
    # existing id, so a second invocation spends quota and produces nothing.
    already = {p.doc_id for p in existing}
    doc_ids = [d for d in sorted(by_doc) if d not in already]

    # Skip documents typeset with line numbers: extraction interleaves those numbers
    # inside sentences, the drafter drops them, and the quote cannot be located. Two such
    # documents returned 0/10 and 1/10. They remain fully indexed and retrievable — this
    # only says they are poor sources of verbatim reference quotes. `--doc-ids` overrides.
    dense = {
        d
        for d in doc_ids
        if golden.line_number_density("\n".join(c["text"] for c in by_doc[d]))
        >= golden.MAX_LINE_NUMBER_DENSITY
    }
    if dense:
        doc_ids = [d for d in doc_ids if d not in dense]
        print(
            f"[golden] skipping {len(dense)} line-numbered document(s) as drafting "
            f"sources (they stay in the index)",
            file=sys.stderr,
        )
    if already:
        print(
            f"[golden] {len(already)} document(s) already drafted; "
            f"{len(doc_ids)} remaining to choose from",
            file=sys.stderr,
        )
    if not doc_ids:
        print("[golden] every document has been drafted from already", file=sys.stderr)
        return 0
    if args.doc_ids:
        # Explicit targets override both the stride and the already-covered filter, so a
        # prompt fix can be tested against the exact documents that motivated it.
        missing = [d for d in args.doc_ids if d not in by_doc]
        if missing:
            print(f"[golden] unknown doc_id(s): {missing}", file=sys.stderr)
            return 1
        doc_ids = list(args.doc_ids)
        seen_ids = {p.pair_id for p in existing if p.doc_id not in set(args.doc_ids)}
        existing = [p for p in existing if p.doc_id not in set(args.doc_ids)]
        print(f"[golden] re-drafting {len(doc_ids)} named document(s)", file=sys.stderr)
    if args.docs >= len(doc_ids):
        picked_docs = doc_ids
    else:
        stride = len(doc_ids) / args.docs
        picked_docs = [doc_ids[int(i * stride)] for i in range(args.docs)]

    print(
        f"[golden] drafting {args.pairs_per_doc} pairs x {len(picked_docs)} documents "
        f"= {args.pairs_per_doc * len(picked_docs)} pairs in {len(picked_docs)} requests "
        f"(free tier allows {generation.FREE_TIER_REQUESTS_PER_DAY}/day/model)",
        file=sys.stderr,
    )
    if len(picked_docs) > generation.FREE_TIER_REQUESTS_PER_DAY:
        print(
            f"[golden] warning: {len(picked_docs)} requests exceeds the daily cap; "
            "the run will stop when the quota is exhausted and keep what it has",
            file=sys.stderr,
        )

    try:
        generator = make_generator(args.generator)
    except GenerationError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    drafted: list[golden.GoldenPair] = []
    for i, doc_id in enumerate(picked_docs, 1):
        chunks = by_doc[doc_id]
        try:
            result = generator.generate(
                golden.build_draft_prompt(doc_id, chunks, args.pairs_per_doc),
                system=golden.draft_system_prompt(args.pairs_per_doc),
                schema=golden.GOLDEN_SCHEMA,
                # 10 pairs with long regulatory answers and long evidence quotes
                # exceeded 8,192 and truncated mid-JSON, losing the whole batch to a
                # parse error. These models allow 65,536; a generous cap costs nothing
                # because output is billed on what is produced, not what is permitted.
                max_output_tokens=32768,
                temperature=0.4,
            )
            payload = result.parse_json()
        except GenerationError as exc:
            # A daily-quota exhaustion is terminal, not transient: stop and keep what
            # was drafted rather than burning the remaining documents on certain
            # failures.
            print(f"  [{i}/{len(picked_docs)}] {doc_id}: {exc}"[:300], file=sys.stderr)
            if "daily free-tier quota" in str(exc):
                print("[golden] stopping early — daily quota exhausted", file=sys.stderr)
                break
            continue

        pairs = golden.parse_pairs(payload, doc_id, start_index=len(drafted))
        validated = golden.validate_pairs(pairs, {doc_id: chunks}, doc_index=doc_index)
        drafted.extend(p for p in validated if p.pair_id not in seen_ids)
        n_ok = sum(1 for p in validated if p.accepted)
        print(
            f"  [{i}/{len(picked_docs)}] {doc_id:<34} {n_ok}/{len(validated)} accepted"
            f"  (prompt {result.prompt_tokens:,} tok)",
            file=sys.stderr,
            flush=True,
        )

    combined = existing + drafted
    combined = _finish_golden(combined, by_doc)
    golden.write_golden(combined, GOLDEN_PATH)
    report = golden.yield_report(combined)
    print(
        f"\n[golden] {report['n_accepted']}/{report['n_drafted']} accepted "
        f"(rate {report['acceptance_rate']}) across {report['documents']} documents",
        file=sys.stderr,
    )
    if report["reject_reasons"]:
        print("[golden] rejections:", file=sys.stderr)
        for reason, n in report["reject_reasons"].items():
            print(f"    {n:>3}  {reason}", file=sys.stderr)
    print(f"[golden] shapes: {report['shapes']}", file=sys.stderr)
    print(f"[golden] evidence methods: {report['accepted_evidence_methods']}", file=sys.stderr)
    print(f"wrote {GOLDEN_PATH}", file=sys.stderr)
    return 0


def _finish_golden(pairs: list, by_doc: dict[str, list[dict]]) -> list:
    """Apply the cross-pair and factual passes, then report what they caught.

    These ran nowhere. `drop_redundant`, `flag_factual_risk` and `sample_for_review` had
    **zero call sites outside `golden.py`** until the Phase 6 code-review gate said so:
    `cmd_golden` called only `validate_pairs` and `yield_report`. So the redundancy gate
    did not run in the pipeline (the one `redundant:` reject on disk was produced by an
    ad-hoc script), and the factual triage the progress log presents as "the check that
    answers 'is the reference answer right'" never executed at all.

    They are cross-pair or cross-corpus checks, which is why they sit outside
    `validate_pair` — but outside is not the same as absent.
    """
    from ragpipe import golden

    before = sum(1 for p in pairs if p.accepted)
    pairs = golden.drop_redundant(pairs)
    after = sum(1 for p in pairs if p.accepted)
    if before != after:
        print(
            f"[golden] redundancy pass rejected {before - after} pair(s) whose answer "
            f"duplicates an earlier one",
            file=sys.stderr,
        )

    chunk_index = {c["chunk_id"]: c for cs in by_doc.values() for c in cs}
    flagged = golden.flag_factual_risk(pairs, chunk_index)
    if flagged:
        print(
            f"[golden] factual triage: {len(flagged)} accepted pair(s) assert a number "
            f"absent from their cited chunk — review these:",
            file=sys.stderr,
        )
        for pair, missing in flagged:
            print(f"    {pair.pair_id}  unsupported {missing}", file=sys.stderr)
            print(f"      Q: {pair.question[:88]}", file=sys.stderr)
    else:
        print("[golden] factual triage: no unsupported numeric claims", file=sys.stderr)
    return pairs


def cmd_revalidate(args: argparse.Namespace) -> int:
    """Re-run every golden-set check against the pairs already on disk. No API calls.

    Phases 6a and 6b both claim re-validation is "free to re-run", and it was — as a
    throwaway script. There was no shipped way to do it, which meant a threshold or prompt
    change could not be tested against the existing set by anyone but the author of that
    script. This is that capability.
    """
    from ragpipe import golden

    pairs = golden.load_golden(GOLDEN_PATH)
    if not pairs:
        print(f"No golden set at {GOLDEN_PATH}.", file=sys.stderr)
        return 1
    chunk_path = CHUNKS_DIR / f"{args.chunking}.jsonl"
    chunks = retrieval.load_chunks(chunk_path)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk["doc_id"]].append(chunk)
    doc_index = golden.build_document_index(chunks)

    before = sum(1 for p in pairs if p.accepted)
    # Re-validate from the *drafted* state so a rule that has since been relaxed can
    # re-accept a pair, not only reject more.
    drafted = [
        golden.GoldenPair(
            pair_id=p.pair_id,
            question=p.question,
            answer=p.answer,
            evidence=p.evidence,
            doc_id=p.doc_id,
            shape=p.shape,
        )
        for p in pairs
    ]
    revalidated = golden.validate_pairs(drafted, by_doc, doc_index=doc_index)
    revalidated = _finish_golden(revalidated, by_doc)

    # Human curation verdicts override the machine checks, and this is load-bearing.
    # Re-validating from the drafted state is what lets a relaxed rule re-accept a pair --
    # but it also means a pair rejected by a *person*, for a defect no machine check can
    # see, gets silently re-accepted on the next re-run. Phase 8c rejected 8 pairs for
    # misstating an obligation level ("should" -> "must"), and every one of them passes all
    # five machine checks: that is precisely why they needed a human. Without this, the
    # next `revalidate --write` would undo a curation pass and report it as a clean run.
    from ragpipe import curate as curate_mod

    verdicts = curate_mod.load_verdicts(CURATION_VERDICTS_PATH)
    curation_rejects = {
        pid: v.get("reason", "unspecified")
        for pid, v in verdicts.items()
        if v.get("verdict") == "reject"
    }
    if curation_rejects:
        import dataclasses

        restored = 0
        held: list[golden.GoldenPair] = []
        for pair in revalidated:
            if pair.pair_id in curation_rejects and pair.accepted:
                held.append(
                    dataclasses.replace(
                        pair,
                        status="rejected",
                        reject_reason=f"curation: {curation_rejects[pair.pair_id]}",
                    )
                )
                restored += 1
            else:
                held.append(pair)
        revalidated = held
        print(
            f"[revalidate] {restored} pair(s) held rejected by human curation verdicts "
            f"that the machine checks would have re-accepted",
            file=sys.stderr,
        )

    report = golden.yield_report(revalidated)
    print(
        f"\n[revalidate] {before} -> {report['n_accepted']} accepted of "
        f"{report['n_drafted']} (rate {report['acceptance_rate']})",
        file=sys.stderr,
    )
    for reason, n in report["reject_reasons"].items():
        print(f"    {n:>3}  {reason}", file=sys.stderr)
    if args.write:
        golden.write_golden(revalidated, GOLDEN_PATH)
        print(f"wrote {GOLDEN_PATH}", file=sys.stderr)
    else:
        print("(dry run — pass --write to persist)", file=sys.stderr)
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    """Phase 6: calibrate the tier-2 judge against controls, then report agreement.

    Runs the calibration *first* and refuses to proceed if the judge fails it. A judge
    that cannot reject a claim paired with another document's evidence cannot be trusted
    to reject anything, and its faithfulness numbers would look exactly as confident.
    """
    from ragpipe import golden, judge
    from ragpipe.generation import GenerationError, make_generator

    pairs = golden.load_golden(GOLDEN_PATH)
    if not pairs:
        print(f"No golden set at {GOLDEN_PATH}. Run `make golden` first.", file=sys.stderr)
        return 1

    items = judge.build_control_set(pairs, args.positives, args.negatives)
    n_pos = sum(1 for i in items if i.control == "positive")
    n_neg = sum(1 for i in items if i.control == "negative")
    groups = judge.batches(items, args.batch_size)
    print(
        f"[judge] calibrating `{args.judge}` on {n_pos} positive + {n_neg} negative "
        f"controls in {len(groups)} request(s)",
        file=sys.stderr,
    )

    try:
        generator = make_generator(args.judge)
    except GenerationError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    judgments: dict[str, Any] = {}
    for i, batch in enumerate(groups, 1):
        try:
            result = generator.generate(
                judge.build_prompt(batch),
                system=judge.JUDGE_SYSTEM_PROMPT,
                schema=judge.JUDGE_SCHEMA,
                max_output_tokens=8192,
                temperature=0.0,
            )
            judgments.update(judge.parse_judgments(result.parse_json()))
        except GenerationError as exc:
            print(f"  [{i}/{len(groups)}] {exc}"[:220], file=sys.stderr)
            if "daily free-tier quota" in str(exc):
                print("[judge] stopping early — daily quota exhausted", file=sys.stderr)
                break
            continue
        print(f"  [{i}/{len(groups)}] {len(batch)} items judged", file=sys.stderr, flush=True)

    agreement = judge.score_agreement(items, judgments)
    d = agreement.as_dict()
    print(
        f"\n[judge] positives called supported: {d['positive_rate']} ({agreement.n_positive} items)"
    )
    print(
        f"[judge] negatives correctly rejected: {d['negative_rate']} ({agreement.n_negative} items)"
    )
    print(f"[judge] verdict mix: {d['verdicts']}   unjudged: {d['unjudged']}")
    print(f"[judge] USABLE: {d['usable']}")
    if not d["usable"]:
        print(
            "[judge] UNUSABLE — it failed the constructed negatives, which cannot be "
            "supported under any reading. Not producing faithfulness numbers from it.",
            file=sys.stderr,
        )
    elif d["inspect_golden_set"]:
        print(
            "[judge] usable, but the positive rate is low. Positives are only *presumed*"
            " supported, so this points at the golden set before the judge: read the "
            "disagreements below (--show) and check whether the answers outrun their "
            "cited evidence.",
            file=sys.stderr,
        )
    out = judge.write_agreement(agreement, REPORTS_DIR / "judge_calibration.json", items, judgments)
    print(f"wrote {out}", file=sys.stderr)

    # Surface disagreements so a human sees what the judge actually did.
    if args.show:
        print("\n[judge] control items the judge got wrong:", file=sys.stderr)
        for item in items:
            j = judgments.get(item.item_id)
            if j is None:
                continue
            wrong = (item.control == "positive" and not j.is_supported) or (
                item.control == "negative" and j.is_supported
            )
            if wrong:
                print(f"  {item.control:<8} {j.verdict:<12} {j.reason[:110]}", file=sys.stderr)
    return 0 if d["usable"] else 1


def _positive_int(raw: str) -> int:
    """An argparse type for counts that must be at least 1.

    `--builds 0` previously ran no builds and then died on a bare `StopIteration` from
    the `next(...)` that pulls the default row out of an empty aggregate.
    """
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def cmd_ann(args: argparse.Namespace) -> int:
    """Phase 7: measure what approximate search costs, against the exact baseline.

    This exists as a shipped command rather than a script because the first version of
    `reports/ann_recall.md` was produced by an ad-hoc file in a scratch directory. The
    numbers were right and the artifact was unreproducible, which is the same defect
    Phase 6 found three times over (`revalidate`): a check that runs somewhere other
    than where the project can re-run it is a check the project does not have.

    Requires a Qdrant **server**. Local mode brute-forces every query and would report
    recall 1.0000 at every setting; `sweep_recall` refuses it rather than printing that.
    """
    import httpx

    from ragpipe import vectorstore as vs

    chunk_path = CHUNKS_DIR / f"{args.chunking}.jsonl"
    if not chunk_path.exists():
        print(f"No chunks at {chunk_path}. Run `make chunk` first.", file=sys.stderr)
        return 1
    rows = list(read_jsonl(EVALSET_PATH))
    if not rows:
        print(f"No eval set at {EVALSET_PATH}. Run `make evalset` first.", file=sys.stderr)
        return 1

    # The same queries the retrieval ablation scores, stratified the same way. Index
    # recall does not depend on the golden spans, but drawing from a different query
    # distribution than every other table in this project would make the latency figures
    # incomparable for no gain.
    picked = gen_eval.stratified_sample(rows, args.per_slice)
    queries = [r["query"] for r in picked]

    chunks = retrieval.load_chunks(chunk_path)
    embedder = embedding.LocalEmbedder(args.embed_model, batch_size=args.batch_size)
    cache = embedding.VectorCache.open(
        VECTORS_DIR / f"{args.embed_model}-{args.chunking}", embedder.dim
    )
    dense = dense_mod.DenseRetriever(chunks, embedder, cache=cache)
    cache.flush()

    try:
        version = httpx.get(f"{args.url}/", timeout=5.0).json().get("version", "?")
    except Exception as exc:  # noqa: BLE001 - any failure here means no server
        print(
            f"No Qdrant at {args.url} ({exc}). Start one with "
            "`docker run -p 6333:6333 qdrant/qdrant`.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[ann] Qdrant {version} at {args.url}: {len(dense.matrix):,} x "
        f"{dense.matrix.shape[1]} vectors, {args.builds} build(s)",
        file=sys.stderr,
    )

    # Transport, measured separately and with no search behind it. Without this the
    # comparison against in-process numpy reads as "HNSW is 6x slower", which is not
    # what the numbers say -- most of the gap is a network hop.
    bare: list[float] = []
    with httpx.Client(timeout=10.0) as http:
        for _ in range(len(queries)):
            started = time.perf_counter()
            http.get(f"{args.url}/")
            bare.append((time.perf_counter() - started) * 1000.0)
    bare.sort()

    # Warm the query-vector cache before timing anything. `DenseRetriever` memoises
    # query embeddings per instance and `QdrantRetriever` delegates to it, so the
    # first search of each query pays a model forward pass -- tens of milliseconds,
    # which would swamp both sides -- and every later one does not. Timing the exact
    # pass cold and the ANN passes warm would have reported exact search as the
    # *slower* engine, off a difference that is entirely query encoding.
    #
    # Encoding is excluded from both sides deliberately: it is identical work for
    # both, so including it would shrink the ratio this measurement exists to show
    # without changing which engine does less work per query.
    for q in queries:
        dense.search(q, k=args.k)

    exact_latencies: list[float] = []
    for q in queries:
        started = time.perf_counter()
        dense.search(q, k=args.k)
        exact_latencies.append((time.perf_counter() - started) * 1000.0)
    exact_latencies.sort()

    # Interpolated median and nearest-rank p95 -- the same pair `aggregate_builds`,
    # `sweep_recall` and `service.Timings` use. An earlier version of this command took
    # `sorted[len//2]` here, the upper-middle value, which is not a median at even n. Two
    # conventions in one report is the defect; using two knowingly and labelling them is
    # not, so all four call sites now agree.
    exact_p50 = statistics.median(exact_latencies)
    exact_p95 = exact_latencies[min(len(exact_latencies) - 1, int(0.95 * len(exact_latencies)))]

    # `None` first: Qdrant's own default search settings, left unset rather than guessed
    # at. The headline claim is about what an untuned user gets, so reusing whichever
    # swept row matches would be asserting what that default is.
    efs: list[int | None] = [None, *args.efs]

    # Several independent builds, because one is a sample. The first version of this
    # measurement built once and reported recall and top-k agreement of exactly 1.0000
    # at the default; a rebuild did not reproduce either number. HNSW graph construction
    # is randomised, so the number depends on the graph -- see `aggregate_builds`.
    builds: list[list[vs.RecallPoint]] = []
    # Per build, not one variable reassigned in the loop. `indexed` was previously the
    # last build's count while the report presented it beside "over 5 independent index
    # builds" as though it described all of them -- and `assert_indexed` returns rather
    # than raising for any nonzero count, so a build that indexed part of the corpus
    # would have been overwritten and the report would still have claimed the full count.
    indexed_per_build: list[int] = []
    thresholds: list[Any] = []
    within_build_identical: bool | None = None
    for build in range(args.builds):
        store = vs.QdrantStore.from_dense(
            dense,
            collection=args.collection,
            url=args.url,
            m=args.hnsw_m,
            ef_construct=args.ef_construct,
        )
        try:
            indexed = store.assert_indexed()
            indexed_per_build.append(indexed)
            points = vs.sweep_recall(store, dense, queries, k=args.k, efs=efs)
            builds.append(points)

            # On the first build only, query the *same* index twice and compare the
            # ranked chunk ids themselves. This is the check that makes the
            # build-versus-search attribution a measurement rather than something
            # remembered from a one-off script: if these ever disagree, search is a
            # second source of variance and the reported range is too narrow.
            #
            # Comparing the ids and their order, not the summary statistics. An earlier
            # version compared each ef's mean recall and exact-match count, which are
            # invariant under reordering -- so it could have reported "bit-identical"
            # about two different result sets, and the report says "bit-identical".
            if build == 0:
                within_build_identical = all(
                    vs.topk_lists(store, dense, queries, k=args.k, ef=ef)
                    == vs.topk_lists(store, dense, queries, k=args.k, ef=ef)
                    for ef in efs
                )

            info = store.client.get_collection(args.collection)
            hnsw_config = getattr(getattr(info, "config", None), "hnsw_config", None)
            thresholds.append(getattr(hnsw_config, "full_scan_threshold", None))
        finally:
            store.close()
        default = next(p for p in points if p.ef is None)
        print(
            f"[ann] build {build + 1}/{args.builds}: {indexed:,} indexed, default recall "
            f"{default.recall_at_k:.4f}, identical {default.exact_match_rate:.4f}",
            file=sys.stderr,
        )

    if within_build_identical is False:
        print(
            "[ann] WARNING: two sweeps over one index disagreed. Search was assumed "
            "deterministic; the reported build range understates the spread.",
            file=sys.stderr,
        )

    rows = vs.aggregate_builds(builds)
    payload = {
        "qdrant_version": version,
        "n_vectors": len(dense.matrix),
        "indexed_vectors": min(indexed_per_build) if indexed_per_build else 0,
        "dim": int(dense.matrix.shape[1]),
        "m": args.hnsw_m,
        "ef_construct": args.ef_construct,
        # The minimum across builds, so a build that indexed less cannot be hidden by a
        # later one. `indexed_per_build` carries the evidence either way.
        "full_scan_threshold": thresholds[0] if len(set(map(str, thresholds))) == 1 else thresholds,
        "indexed_per_build": indexed_per_build,
        "k": args.k,
        "n_queries": len(queries),
        "n_builds": args.builds,
        "within_build_identical": within_build_identical,
        "exact_p50_ms": exact_p50,
        "exact_p95_ms": exact_p95,
        "bare_http_p50_ms": statistics.median(bare),
        "latency_conventions": "p50 = interpolated median; p95 = nearest rank",
        "default": next(r for r in rows if r["ef"] is None),
        "sweep": [r for r in rows if r["ef"] is not None],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "ann_recall.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (REPORTS_DIR / "ann_recall.md").write_text(vs.render_ann_report(payload), encoding="utf-8")
    d = payload["default"]
    print(
        f"[ann] default recall {d['recall_at_k']:.4f} ({d['recall_min']:.4f}-"
        f"{d['recall_max']:.4f}), identical {d['exact_match_rate']:.4f} "
        f"({d['identical_min']:.4f}-{d['identical_max']:.4f}), p50 {d['p50_ms']:.2f} ms "
        f"vs exact {payload['exact_p50_ms']:.2f} ms -> {REPORTS_DIR / 'ann_recall.md'}",
        file=sys.stderr,
    )
    return 0


def cmd_failures(args: argparse.Namespace) -> int:
    """Phase 8: the failure analysis, assembled from the artifacts rather than memory.

    Reads only committed artifacts and the chunk file, spends nothing, and omits any
    finding whose artifact is absent instead of estimating it.
    """
    from ragpipe import failures

    chunk_path = CHUNKS_DIR / f"{args.chunking}.jsonl"
    if not chunk_path.exists():
        print(f"No chunks at {chunk_path}. Run `make chunk` first.", file=sys.stderr)
        return 1

    artifacts = failures.load_artifacts(REPORTS_DIR, CORPUS_DIR)
    if artifacts["missing"]:
        print(
            f"[failures] missing artifacts: {', '.join(artifacts['missing'])} - the "
            "findings that depend on them will be omitted",
            file=sys.stderr,
        )
    chunks = retrieval.load_chunks(chunk_path)
    payload = failures.build_payload(artifacts, chunks)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "failure_modes.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (REPORTS_DIR / "failure_modes.md").write_text(failures.render_report(payload), encoding="utf-8")

    collapse = payload["identifier_collapse"]
    if collapse:
        worst = max(collapse, key=lambda r: r["ratio"] or 0)
        print(
            f"[failures] identifier collapse: sparse/dense {worst['ratio']:.0f}x on "
            f"{worst['chunking']}",
            file=sys.stderr,
        )
    print(
        f"[failures] obligation errors: {payload['obligations']['n_rejected']} of "
        f"{payload['obligations']['n_judged']} judged pairs rejected",
        file=sys.stderr,
    )
    ln = payload["line_numbers"]
    print(
        f"[failures] line-number exposure: {ln['n_affected']:,}/{ln['n_chunks']:,} chunks "
        f"({ln['share_affected']:.1%})",
        file=sys.stderr,
    )
    print(f"[failures] -> {REPORTS_DIR / 'failure_modes.md'}", file=sys.stderr)
    return 0


def cmd_curate(args: argparse.Namespace) -> int:
    """Phase 8: triage the golden set for the defect classes coverage cannot see.

    Costs nothing and touches no network: it compares each accepted pair's answer against
    the full text of the chunks it cites. Writes a worksheet a human reads, and never
    edits `golden.jsonl` -- rejecting a pair is a judgement call, and `ragpipe curate
    --apply` is the separate, explicit step that records one.
    """
    from ragpipe import curate as curate_mod

    chunk_path = CHUNKS_DIR / f"{args.chunking}.jsonl"
    if not chunk_path.exists():
        print(f"No chunks at {chunk_path}. Run `make chunk` first.", file=sys.stderr)
        return 1
    if not GOLDEN_PATH.exists():
        print(f"No golden set at {GOLDEN_PATH}. Run `make golden` first.", file=sys.stderr)
        return 1

    rows = list(read_jsonl(GOLDEN_PATH))
    verdicts = curate_mod.load_verdicts(CURATION_VERDICTS_PATH)

    # Accepted pairs **plus anything a verdict already names**. Without the second half,
    # `--apply` removes its own inputs: the rejected pairs leave the pool, the worksheet
    # loses every rejection it documented, and the header divides N recorded verdicts by a
    # smaller flagged count ("Judged: 25 of 18"). The committed artifact then cannot be
    # rebuilt by the command that produced it -- the one property every other report here
    # has, and the one plan.md's process section requires.
    pool = [r for r in rows if r.get("status") == "accepted" or r.get("pair_id") in verdicts]
    if not pool:
        print(f"No reviewable pairs in {GOLDEN_PATH}.", file=sys.stderr)
        return 1

    chunks = retrieval.load_chunks(chunk_path)
    chunk_index = {c["chunk_id"]: c for c in chunks}

    reviews = curate_mod.review_all(pool, chunk_index)
    summary = curate_mod.summarise(reviews)

    # A verdict naming a pair_id that does not exist is a hand-typed file's likeliest
    # error, and it silently removes a human judgement: `--apply` skips it and the
    # "N pairs moved" line simply reads one lower. Named loudly instead.
    known = {r.get("pair_id") for r in rows}
    orphans = sorted(set(verdicts) - known)
    if orphans:
        print(
            f"[curate] WARNING: {len(orphans)} verdict(s) name a pair_id absent from "
            f"{GOLDEN_PATH.name} - a typo here deletes a human judgement silently:",
            file=sys.stderr,
        )
        for pair_id in orphans:
            print(f"[curate]   {pair_id}", file=sys.stderr)

    summary["verdicts_recorded"] = len(verdicts)
    # Verdicts whose pair is in the pool but is no longer flagged: judged under an earlier,
    # buggier detector set that has since been fixed. Counted separately so the worksheet
    # never divides all recorded verdicts by the current flag count.
    flagged_ids = {r.pair_id for r in reviews if r.needs_review}
    summary["verdicts_retired_by_detector_fixes"] = len(
        {pid for pid in verdicts if pid in known and pid not in flagged_ids}
    )
    summary["orphan_verdicts"] = orphans
    summary["flagged_unjudged"] = sum(
        1 for r in reviews if r.needs_review and r.pair_id not in verdicts
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunking": args.chunking,
        "summary": summary,
        "verdicts": verdicts,
        "reviews": [r.as_dict() for r in reviews if r.needs_review],
    }
    (REPORTS_DIR / "golden_curation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (REPORTS_DIR / "golden_curation.md").write_text(
        curate_mod.render_worksheet(reviews, summary, verdicts), encoding="utf-8"
    )

    print(
        f"[curate] {summary['n_flagged']}/{summary['n_pairs']} pairs flagged "
        f"({summary['flagged_share']:.1%}): "
        f"{summary['by_severity']['high']} high, {summary['by_severity']['medium']} medium, "
        f"{summary['by_severity']['low']} low",
        file=sys.stderr,
    )
    for kind, count in summary["by_kind"].items():
        print(f"[curate]   {kind}: {count}", file=sys.stderr)
    if verdicts:
        print(
            f"[curate] {len(verdicts)} verdicts recorded, "
            f"{summary['flagged_unjudged']} flagged pairs still unjudged",
            file=sys.stderr,
        )
    print(f"[curate] -> {REPORTS_DIR / 'golden_curation.md'}", file=sys.stderr)

    if args.apply:
        # `edit` is in VERDICT_VALUES and no consumer handles it. Recording one would leave
        # the defect in the set *and* strip `revalidate`'s protection for that pair, so
        # refuse rather than silently no-op.
        edits = sorted(pid for pid, v in verdicts.items() if v.get("verdict") == "edit")
        if edits:
            print(
                f"--apply cannot act on {len(edits)} 'edit' verdict(s): editing a reference "
                "answer by hand is not implemented, and leaving one recorded would keep a "
                "pair a human found defective in the accepted set. Change them to keep or "
                f"reject. {', '.join(edits)}",
                file=sys.stderr,
            )
            return 1

        rejected = {pid: v for pid, v in verdicts.items() if v.get("verdict") == "reject"}
        if not rejected:
            print("[curate] --apply: no 'reject' verdicts to apply", file=sys.stderr)
            return 0
        changed = 0
        for row in rows:
            pid = row.get("pair_id")
            if pid in rejected and row.get("status") == "accepted":
                row["status"] = "rejected"
                row["reject_reason"] = f"curation: {rejected[pid].get('reason', 'unspecified')}"
                changed += 1
        write_jsonl_dicts(GOLDEN_PATH, rows)
        print(
            f"[curate] --apply: {changed} pairs moved accepted -> rejected in {GOLDEN_PATH}",
            file=sys.stderr,
        )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Phase 8: per-stage serving latency, written to a report rather than a chat log.

    Deliberately runs the pipeline in-process rather than against a container. The point
    is the *stages*, and adding an HTTP hop would fold ~0.8 ms of transport (measured
    separately in `ann_recall.md`) into every local row.
    """
    from ragpipe import bench as bench_mod

    chunk_path = CHUNKS_DIR / f"{args.chunking}.jsonl"
    if not chunk_path.exists():
        print(f"No chunks at {chunk_path}. Run `make chunk` first.", file=sys.stderr)
        return 1
    rows = list(read_jsonl(EVALSET_PATH))
    if not rows:
        print(f"No eval set at {EVALSET_PATH}. Run `make evalset` first.", file=sys.stderr)
        return 1

    chunks = retrieval.load_chunks(chunk_path)
    chunk_index = {c["chunk_id"]: c for c in chunks}
    retriever = retrieval.BM25Retriever(chunks, heading_mode=args.heading_mode)
    verify_by_method: dict[str, Any] = {}

    # Answerable queries only. `unanswerable` probes are real queries and retrieval runs
    # on them identically, but including them would make the sample half questions the
    # corpus cannot answer -- fine for scoring refusal, wrong for a latency profile meant
    # to describe serving traffic.
    queries = [r["query"] for r in rows if r.get("slice_name") != "unanswerable"]
    print(
        f"[bench] {len(queries)} queries x {args.repeats} passes over "
        f"{len(retriever):,} {args.chunking} passages",
        file=sys.stderr,
    )

    results = [bench_mod.bench_retrieve(retriever, queries, k=args.k, repeats=args.repeats)]

    gen_eval = bench_mod.read_json(REPORTS_DIR / "generation_eval.json")
    if gen_eval:
        citation_sets = bench_mod.load_citation_sets(gen_eval, chunk_index)
        if citation_sets:
            results.append(bench_mod.bench_verify(citation_sets, repeats=args.repeats))
            verify_by_method = bench_mod.verify_cost_by_method(citation_sets)
        else:
            print(
                "[bench] generation_eval.json has no stored citations; skipping verify",
                file=sys.stderr,
            )
    else:
        print(
            "[bench] no reports/generation_eval.json: verify and generate need it. "
            "Run `make gen-eval` first, or accept a retrieve-only profile.",
            file=sys.stderr,
        )

    if args.live_generate:
        # The help text promises a cap; enforce it here rather than trusting the promise.
        # A rejected request still spends a quota unit, so an accidental `--live-generate
        # 50` would burn the day's budget twice over and produce nothing.
        if args.live_generate > 5:
            print(
                f"--live-generate is capped at 5 (got {args.live_generate}): each call "
                "spends one of 20 daily free-tier requests, and a rejected call spends "
                "one too.",
                file=sys.stderr,
            )
            return 1

        from ragpipe.generation import GenerationError, make_generator

        try:
            generator = make_generator(args.generator)
        except (GenerationError, ValueError) as exc:
            print(f"[bench] --live-generate needs a working generator: {exc}", file=sys.stderr)
            return 1
        print(
            f"[bench] measuring generation live: {args.live_generate} calls, "
            f"{args.live_generate} of today's 20-request budget",
            file=sys.stderr,
        )
        results.append(
            bench_mod.bench_generate_live(
                queries,
                chunk_index,
                retriever,
                generator=generator,
                k=args.k,
                limit=args.live_generate,
            )
        )
    elif gen_eval:
        results.append(bench_mod.generate_from_stored(gen_eval))

    payload = bench_mod.build_payload(
        results,
        meta={
            "verify_by_method": verify_by_method,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "chunking": args.chunking,
            "heading_mode": args.heading_mode,
            "n_passages": len(retriever),
            "n_queries": len(queries),
            "repeats": args.repeats,
            "k": args.k,
            "generate_measured_live": bool(args.live_generate),
        },
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "serving_bench.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (REPORTS_DIR / "serving_bench.md").write_text(
        bench_mod.render_report(payload), encoding="utf-8"
    )

    for row in payload["stages"]:
        if row["n"]:
            print(
                f"[bench] {row['stage']:9} n={row['n']:<6} p50={row['p50_ms']:.4f} ms "
                f"p95={row['p95_ms']:.4f} ms ({row['source']})",
                file=sys.stderr,
            )
    print(f"[bench] -> {REPORTS_DIR / 'serving_bench.md'}", file=sys.stderr)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Phase 7: serve retrieval + grounded answering over HTTP.

    Loads the index once, before binding the port, so a slow start is a slow start and
    not a fast start that serves 30-second requests. Generation is optional: without a
    key the service comes up retrieval-only and says so on `/health`.
    """
    import uvicorn

    from ragpipe import service as svc

    chunk_path = CHUNKS_DIR / f"{args.chunking}.jsonl"
    if not chunk_path.exists():
        print(f"No chunks at {chunk_path}. Run `make chunk` first.", file=sys.stderr)
        return 1

    print(f"[serve] loading {args.chunking} chunks + BM25 index", file=sys.stderr)
    state = svc.load_state(
        chunk_path,
        chunking=args.chunking,
        heading_mode=args.heading_mode,
        generator_spec=None if args.no_generator else args.generator,
    )
    if state.generator_available:
        print(f"[serve] generator: {args.generator}", file=sys.stderr)
    else:
        why = state.generator_error or "disabled with --no-generator"
        print(f"[serve] retrieval only ({why})", file=sys.stderr)
    print(
        f"[serve] {len(state.chunks):,} chunks ready on http://{args.host}:{args.port}",
        file=sys.stderr,
    )
    uvicorn.run(svc.create_app(state), host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    manifest = load_manifest(MANIFEST_PATH)
    if not manifest:
        print(f"No manifest at {MANIFEST_PATH}. Run `make manifest` first.", file=sys.stderr)
        return 1

    payload = stats_mod.build(manifest, load_fetched(FETCHED_PATH))
    report = REPORTS_DIR / "corpus_report.md"
    js = REPORTS_DIR / "corpus_stats.json"
    stats_mod.write(payload, report, js)

    c = payload["counts"]
    print(f"{c['indexable']}/{c['in_manifest']} documents indexable")
    print(
        f"~{payload['estimated_tokens']:,} tokens, "
        f"~{payload['estimated_chunks']['512_tok']:,} chunks at 512 tokens"
    )
    print(f"\nWrote {report}\n      {js}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Chunk-size sweep: does the best chunk size depend on the query type?

    Chunk size is the parameter most RAG tutorials pick by feel and never revisit. It
    trades precision against fragmentation — smaller chunks carry less irrelevant text
    around a match, but split a section across more of them, so a section-scoped query
    needs more retrieved to reach the same recall.

    Chunks are built in memory rather than written to `data/chunks/`: the sweep is a
    measurement, not an artifact, and writing size-suffixed files would leave the main
    pipeline with several competing definitions of `structural`. Embeddings still
    persist, because `VectorCache` keys on the prepared text — so a re-run is free.
    """
    queries = [evalset.EvalQuery(**q) for q in read_jsonl(EVALSET_PATH)]
    if not queries:
        print(f"No eval set at {EVALSET_PATH}. Run `make evalset` first.", file=sys.stderr)
        return 1

    index_rows = list(read_jsonl(EXTRACT_INDEX_PATH))
    docs: list[dict] = []
    for row in index_rows:
        path = EXTRACTED_DIR / row["source"] / f"{row['doc_id']}.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            extracted = json.load(fh)
        # Literally the same predicate as chunking, not a re-statement of it — see
        # `_is_quarantined`. Letting a corrupt document in would put an identical floor
        # under every size, which is indistinguishable from a real result.
        if _is_quarantined(extracted):
            continue
        docs.append(extracted)

    rows: list[dict] = []
    shape: dict[str, dict] = {}

    for target in args.target_chars:
        overlap = int(target * args.overlap_ratio)
        chunks: list[dict] = []
        for extracted in docs:
            chunks.extend(
                c.as_dict()
                for c in chunking.build_chunks(
                    extracted, args.strategy, target=target, overlap=overlap
                )
            )
        chunks_by_doc: dict[str, list[dict]] = defaultdict(list)
        for c in chunks:
            chunks_by_doc[c["doc_id"]].append(c)

        embedder = embedding.LocalEmbedder(args.embed_model, batch_size=args.batch_size)
        cache = embedding.VectorCache.open(
            VECTORS_DIR / f"{args.embed_model}-{args.strategy}-t{target}", embedder.dim
        )
        dense = dense_mod.DenseRetriever(chunks, embedder, cache=cache, count_truncation=True)
        cache.flush()

        sizes = sorted(c["n_chars"] for c in chunks)
        shape[str(target)] = {
            "n_chunks": len(chunks),
            "median_chars": sizes[len(sizes) // 2] if sizes else 0,
            "index_tokens": sum(len(c["embed_text"]) for c in chunks) // chunking.CHARS_PER_TOKEN,
            "pct_truncated": (dense.truncation or {}).get("pct_truncated", 0.0),
        }

        sparse = retrieval.BM25Retriever(chunks, name="bm25", use_identifier_atoms=True)
        retrievers = [
            sparse,
            dense,
            fusion.HybridRetriever(
                [sparse, dense], method="rrf", name="hybrid rrf", fetch_k=args.fetch_k
            ),
            fusion.HybridRetriever(
                [sparse, dense],
                method="minmax",
                weights=[0.9, 0.1],
                name="hybrid minmax(s0.9/d0.1)",
                fetch_k=args.fetch_k,
            ),
        ]

        for retriever in retrievers:
            print(
                f"[t={target} / {retriever.name}] scoring {len(queries)} queries "
                f"over {len(chunks):,} chunks ...",
                flush=True,
            )
            per_slice: dict[str, list[metrics.QueryResult]] = defaultdict(list)
            for q in queries:
                hits = retriever.search(q.query, k=max(metrics.K_VALUES))
                relevant = evalset.relevant_chunks(q, chunks_by_doc)
                per_slice[q.slice_name].append(
                    metrics.QueryResult(
                        query_id=q.query_id,
                        slice_name=q.slice_name,
                        ranked_ids=[h.chunk_id for h in hits],
                        relevant_ids=frozenset(relevant),
                        top_score=hits[0].score if hits else 0.0,
                    )
                )
            like_for_like = per_slice.get("exact_identifier", []) + per_slice.get(
                "unanswerable", []
            )
            for slice_name, results in sorted(per_slice.items()):
                rows.append(
                    {
                        "target_chars": target,
                        "overlap_chars": overlap,
                        "strategy": args.strategy,
                        "retriever": retriever.name,
                        "slice_name": slice_name,
                        "metrics": metrics.aggregate(results),
                        "separability": metrics.separability(like_for_like)
                        if slice_name == "unanswerable"
                        else {},
                    }
                )

    payload = {
        "generated_at": __import__("datetime")
        .datetime.now(__import__("datetime").UTC)
        .isoformat(timespec="seconds"),
        "strategy": args.strategy,
        "embed_model": args.embed_model,
        "overlap_ratio": args.overlap_ratio,
        "shape": shape,
        "rows": rows,
    }
    report = REPORTS_DIR / "chunk_size_sweep.md"
    js = REPORTS_DIR / "chunk_size_sweep.json"
    eval_report.write_sweep(payload, report, js)
    print(f"\nWrote {report}\n      {js}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ragpipe", description="Phase 0 corpus acquisition for the hybrid-search RAG project"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest", help="discover + sample documents, write the manifest")
    p.add_argument("--fda-n", type=int, default=120, help="FDA guidance documents to sample")
    p.add_argument("--ctgov-n", type=int, default=40, help="protocol documents to sample")
    p.add_argument(
        "--ctgov-pool",
        type=int,
        default=1200,
        help="studies to scan when building the protocol candidate pool",
    )
    p.add_argument("--seed", type=int, default=20260813, help="sampling seed (reproducibility)")
    p.add_argument("--min-interval", type=float, default=0.4, help="seconds between requests/host")
    p.add_argument("--dry-run", action="store_true", help="report the sample without writing")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("fetch", help="download manifest PDFs, verify hashes, classify text layer")
    p.add_argument(
        "--refetch", action="store_true", help="re-download every document, ignoring local copies"
    )
    p.add_argument(
        "--reinspect",
        action="store_true",
        help="re-run PDF inspection on local files without any downloads",
    )
    p.add_argument(
        "--allow-drift",
        action="store_true",
        help="accept and re-pin documents whose upstream content changed",
    )
    p.add_argument("--min-interval", type=float, default=0.5, help="seconds between requests/host")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("evalset", help="build the golden retrieval set (exact ground truth)")
    p.add_argument(
        "--per-slice", type=int, default=evalset.DEFAULT_PER_SLICE, help="queries per slice"
    )
    p.add_argument("--seed", type=int, default=20260813)
    p.set_defaults(func=cmd_evalset)

    p = sub.add_parser("eval", help="score retrievers against the golden set")
    p.add_argument(
        "--chunkings",
        nargs="*",
        default=None,
        help=f"defaults to all: {sorted(chunking.STRATEGIES)}",
    )
    p.add_argument(
        "--with-ablations",
        action="store_true",
        help="also score variants that isolate one design decision",
    )
    p.add_argument(
        "--dense-models",
        nargs="*",
        default=[],
        help=f"dense models to score, and fuse with BM25: {sorted(embedding.MODELS)}",
    )
    p.add_argument(
        "--fusion-weights",
        nargs="*",
        type=float,
        default=[0.7],
        help="dense weight(s) for weighted fusion; sparse gets 1-w",
    )
    p.add_argument(
        "--fetch-k",
        type=int,
        default=100,
        help="candidates pulled from each component before fusion",
    )
    p.add_argument("--batch-size", type=int, default=64, help="embedding batch size")
    p.add_argument(
        "--rerank-model",
        default=None,
        help=f"cross-encoder to rerank with: {sorted(rerank.RERANKERS)}",
    )
    p.add_argument(
        "--candidate-k",
        type=int,
        default=rerank.DEFAULT_CANDIDATE_K,
        help="candidates the reranker scores; sets its recall ceiling",
    )
    p.add_argument("--seed", type=int, default=20260813)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("sweep", help="chunk-size sweep at fixed overlap ratio")
    p.add_argument(
        "--strategy",
        default="structural",
        help=f"chunking strategy to sweep: {sorted(chunking.STRATEGIES)}",
    )
    p.add_argument(
        "--target-chars",
        nargs="*",
        type=int,
        default=[512, 1024, 2048, 4096],
        help="target chunk sizes in characters",
    )
    p.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.15,
        help="overlap as a fraction of target, held constant so size is the only variable",
    )
    p.add_argument(
        "--embed-model",
        default=embedding.DEFAULT_MODEL,
        help=f"dense model: {sorted(embedding.MODELS)}",
    )
    p.add_argument(
        "--fetch-k",
        type=int,
        default=100,
        help="candidates pulled from each component before fusion",
    )
    p.add_argument("--batch-size", type=int, default=64, help="embedding batch size")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("gen-eval", help="Phase 5: grounded answers with verified citations")
    p.add_argument("--chunking", default="structural", choices=sorted(chunking.STRATEGIES))
    p.add_argument("--heading-mode", default="prepend", choices=["prepend", "source", "strip"])
    p.add_argument("--k", type=int, default=5, help="chunks of context per query")
    p.add_argument("--per-slice", type=int, default=20, help="queries sampled per eval slice")
    p.add_argument("--generator", default=generation.DEFAULT_GENERATOR)
    p.add_argument(
        "--refusal-threshold",
        type=float,
        default=None,
        help="refuse before generating when the top retrieval score is below this. "
        "Off by default: the threshold is retriever-specific and must be calibrated "
        "against the retriever that ships.",
    )
    p.add_argument(
        "--judge",
        default=None,
        help="also run tier 2 (does the located quote support the answer?) with this "
        "model. Off by default: it costs a request per batch, and the judge is "
        "calibrated against controls first — faithfulness is omitted entirely if it fails.",
    )
    p.add_argument(
        "--judge-unit",
        default="answer",
        choices=["answer", "citation"],
        help="what tier 2 judges. 'answer' (default) asks whether all of an answer's "
        "located quotes together support it. 'citation' judges each quote against the "
        "whole answer, which returned 'partial' for 18 of 22 citations in the first real "
        "run because that is the wrong question when an answer cites several — kept as a "
        "diagnostic for localising which citation is weak.",
    )
    p.add_argument(
        "--rejudge",
        action="store_true",
        help="re-run tier 2 over the existing report without re-generating",
    )
    p.add_argument("--judge-controls", type=int, default=8, help="controls per side")
    p.add_argument("--judge-batch-size", type=int, default=judge_mod.DEFAULT_BATCH_SIZE)
    p.set_defaults(func=cmd_answer_eval)

    p = sub.add_parser("golden", help="Phase 6: draft + validate golden Q&A pairs")
    p.add_argument("--chunking", default="structural", choices=sorted(chunking.STRATEGIES))
    p.add_argument("--docs", type=int, default=4, help="documents to draft from (= API requests)")
    p.add_argument("--pairs-per-doc", type=int, default=golden_mod.DEFAULT_PAIRS_PER_CALL)
    p.add_argument(
        "--doc-ids",
        nargs="+",
        default=None,
        help="re-draft these documents specifically, replacing their existing pairs",
    )
    p.add_argument("--generator", default=generation.DEFAULT_GENERATOR)
    p.add_argument(
        "--overwrite", action="store_true", help="discard the existing set instead of appending"
    )
    p.set_defaults(func=cmd_golden)

    p = sub.add_parser(
        "revalidate", help="Phase 6: re-run every golden-set check on disk, no API calls"
    )
    p.add_argument("--chunking", default="structural", choices=sorted(chunking.STRATEGIES))
    p.add_argument("--write", action="store_true", help="persist the result")
    p.set_defaults(func=cmd_revalidate)

    p = sub.add_parser("judge", help="Phase 6: calibrate the tier-2 support judge")
    p.add_argument("--judge", default=judge_mod.DEFAULT_JUDGE)
    p.add_argument("--positives", type=int, default=16)
    p.add_argument("--negatives", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=judge_mod.DEFAULT_BATCH_SIZE)
    p.add_argument("--show", action="store_true", help="print the control items it got wrong")
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("ann", help="Phase 7: what approximate search costs vs exact")
    p.add_argument("--chunking", default="structural", choices=sorted(chunking.STRATEGIES))
    p.add_argument("--embed-model", default="bge-small")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--per-slice",
        type=int,
        default=15,
        help="queries per eval slice; 15 over four slices is the 60-query sample the "
        "shipped report used",
    )
    p.add_argument(
        "--url",
        default="http://localhost:6333",
        help="Qdrant server. A server is required: local mode brute-forces every query "
        "and would report recall 1.0000 at every ef, which is what the first attempt did.",
    )
    p.add_argument("--collection", default="ragpipe_ann")
    p.add_argument(
        "--builds",
        type=_positive_int,
        default=5,
        help="independent index rebuilds to measure over. HNSW construction is randomised, "
        "so one build is a sample: the first version of this report built once and reported "
        "recall 1.0000 at the default, which a rebuild did not reproduce.",
    )
    p.add_argument("--hnsw-m", type=int, default=vectorstore.DEFAULT_HNSW_M)
    p.add_argument("--ef-construct", type=int, default=vectorstore.DEFAULT_HNSW_EF_CONSTRUCT)
    p.add_argument(
        "--efs",
        type=int,
        nargs="+",
        default=list(vectorstore.DEFAULT_EF_SWEEP),
        help="hnsw_ef values to sweep, spanning below k through well above it",
    )
    p.set_defaults(func=cmd_ann)

    p = sub.add_parser("failures", help="Phase 8: the failure analysis, from the artifacts")
    p.add_argument("--chunking", default="structural", choices=sorted(chunking.STRATEGIES))
    p.set_defaults(func=cmd_failures)

    p = sub.add_parser(
        "curate", help="Phase 8: triage the golden set for modal/polarity/numeric errors"
    )
    p.add_argument("--chunking", default="structural", choices=sorted(chunking.STRATEGIES))
    p.add_argument(
        "--apply",
        action="store_true",
        help="rewrite golden.jsonl, moving pairs with a recorded 'reject' verdict from "
        "accepted to rejected. Verdicts come from corpus/curation_verdicts.json and are "
        "written by a human, never by this command.",
    )
    p.set_defaults(func=cmd_curate)

    p = sub.add_parser("bench", help="Phase 8: per-stage serving latency -> reports/")
    p.add_argument("--chunking", default="structural", choices=sorted(chunking.STRATEGIES))
    p.add_argument("--heading-mode", default="prepend", choices=["prepend", "source", "strip"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument(
        "--repeats",
        type=int,
        default=bench.DEFAULT_REPEATS,
        help="passes over the query set for the local stages. Retrieval is sub-millisecond, "
        "so one pass is mostly measuring the rest of the machine.",
    )
    p.add_argument(
        "--live-generate",
        type=int,
        default=0,
        metavar="N",
        help="measure generation for real with N calls instead of reading the stored "
        "figures. Each call spends one of the free tier's 20 daily requests, and a "
        "rejected call spends one too, so this is capped at 5.",
    )
    p.add_argument("--generator", default=generation.DEFAULT_GENERATOR)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("serve", help="Phase 7: serve retrieval + verified answers over HTTP")
    p.add_argument("--chunking", default="structural", choices=sorted(chunking.STRATEGIES))
    p.add_argument("--heading-mode", default="prepend", choices=["prepend", "source", "strip"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--generator", default=generation.DEFAULT_GENERATOR)
    p.add_argument(
        "--no-generator",
        action="store_true",
        help="serve retrieval only, without contacting a generation provider. Distinct "
        "from having no key: this is a deliberate choice and /health reports it as one.",
    )
    p.add_argument("--log-level", default="info")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("stats", help="build the Phase 0 corpus report")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("extract", help="PDF -> normalised text, section structure, and identifiers")
    p.add_argument(
        "--force",
        action="store_true",
        help="re-extract every document, ignoring the content-hash cache",
    )
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("chunk", help="chunk extracted text with each strategy, cluster duplicates")
    p.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help=f"defaults to all implemented: {sorted(chunking.STRATEGIES)}",
    )
    p.add_argument(
        "--target-chars",
        type=int,
        default=chunking.DEFAULT_TARGET_CHARS,
        help="target chunk size in characters (~4 chars per token)",
    )
    p.add_argument(
        "--overlap-chars",
        type=int,
        default=chunking.DEFAULT_OVERLAP_CHARS,
        help="overlap between adjacent chunks, in characters",
    )
    p.add_argument(
        "--dup-threshold",
        type=float,
        default=dedup.DUPLICATE_THRESHOLD,
        help="Jaccard threshold above which chunks are clustered as duplicates",
    )
    p.add_argument(
        "--embed-model",
        default=embedding.DEFAULT_MODEL,
        help=f"embedder for `semantic`: {sorted(embedding.MODELS)}",
    )
    p.set_defaults(func=cmd_chunk)

    args = parser.parse_args(argv)
    if getattr(args, "refetch", False) and getattr(args, "reinspect", False):
        parser.error("--refetch and --reinspect are mutually exclusive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
