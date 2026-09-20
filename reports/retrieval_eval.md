# Retrieval evaluation — the ablation table

_Generated 2026-08-18T11:32:18+00:00_

Sparse, dense, three rank-fusion methods across a weight sweep, and cross-encoder reranking.
Built on the Phase 2 harness, which established the ruler and the sparse
baseline every later row is measured against.

Retrieval metrics need only `question -> relevant chunk ids`, so every row here
is free to recompute once embeddings are cached — which is why this table gates
changes instead of being produced once at the end.

## Evaluation set

Queries are **programmatic probes with exact ground truth**, derived from
structure already in the corpus, not natural-language questions. They test
retrieval mechanics precisely and cost nothing to run. Natural-language
questions need an LLM to draft and a human to curate; those arrive with the
generation eval in Phase 6 and answer a different question.

| Slice | Queries | Primary metric | Ground truth |
|---|---|---|---|
| `exact_identifier` | 60 | recall@10 | spans where the cited regulation occurs |
| `section_lookup` | 60 | recall@10 | the named section's span (heading unique corpus-wide) |
| `title_lookup` | 60 | hit@1 | the whole named document (document-level task) |
| `unanswerable` | 60 | separability | empty by construction; citation verified absent from the corpus |

Total: **240 queries** (180 answerable, 60 unanswerable). Seed `20260813`.

## Context-window truncation

Text past a model's context window is silently discarded, and text that was
never embedded can never be retrieved. That makes truncation a *retrieval*
defect that shows up as mediocre metrics rather than as an error, so it is
counted here rather than assumed away.

Truncation has two independent causes, and this table measures only the
second. **Across** chunk sizes the rate is set by the target — see
`chunk_size_sweep.md`, where it runs 0.0% at 512 characters to 31.1% at 4,096.
**Within** a fixed size, which particular chunks get clipped is set by
tokenization density rather than by chunks running over target: regulatory text
runs ~4.5 chars/token at the median but far lower at the floor, and the floor is
table-of-contents dot leaders, where a run of `....................` costs
roughly one token per character. Collapsing those runs is what this table's
figures are measured after.

Figures below are **after** normalization (leader and rule runs collapsed),
since that is what the model receives. Normalization is applied to `embed_text`
only, never to `text`, so citation offsets are untouched.

| Chunking / model | Chunks | Truncated | % chunks truncated | % tokens dropped |
|---|---|---|---|---|
| `fixed/bge-small` | 11,527 | 3 | **0.0%** | 0.0% |
| `semantic/bge-small` | 11,468 | 2 | **0.0%** | 0.0% |
| `structural/bge-small` | 13,423 | 4 | **0.0%** | 0.0% |

## Results

Each row is one (chunking strategy, retriever) pair. Metrics are means over the
answerable queries in that slice.

**Comparing `recall@k` across chunking strategies overstates the gap.** Span-based
ground truth removes the chunk-id remapping artifact, but recall still has a
per-strategy denominator: one section resolves to fewer chunks under `structural`
than under `fixed`, so `fixed` must retrieve more chunks to reach the same recall.
The exact ratio moves with the chunk target, so it is not quoted here — the point
is the direction. `hit@k` is denominator-free and is the figure to use for
cross-strategy claims. The same trap applies across chunk *sizes*, and there it is
large enough to invert a conclusion: see `chunk_size_sweep.md`.

### `exact_identifier` — primary metric **recall@10**

| Chunking | Retriever | n | hit@1 | hit@10 | recall@10 | precision@10 | mrr@10 | ndcg@10 | recall@10 ceiling |
|---|---|---|---|---|---|---|---|---|---|
| `semantic` | `bm25` | 60 | 0.950 | 0.983 | **0.972** | 0.175 | 0.964 | 0.949 | — |
| `semantic` | `bm25 heading=source` | 60 | 0.950 | 0.983 | **0.972** | 0.175 | 0.964 | 0.949 | — |
| `semantic` | `bm25 heading=strip` | 60 | 0.950 | 0.983 | **0.972** | 0.175 | 0.964 | 0.949 | — |
| `semantic` | `hybrid bge-small minmax(s0.9/d0.1)` | 60 | 0.933 | 0.983 | **0.972** | 0.175 | 0.956 | 0.941 | — |
| `semantic` | `hybrid bge-small wrrf(s0.9/d0.1)` | 60 | 0.400 | 0.983 | **0.964** | 0.172 | 0.597 | 0.667 | — |
| `structural` | `bm25 heading=strip` | 60 | 0.900 | 0.983 | **0.963** | 0.187 | 0.929 | 0.927 | — |
| `fixed` | `bm25` | 60 | 0.933 | 1.000 | **0.955** | 0.200 | 0.957 | 0.932 | — |
| `fixed` | `bm25 heading=source` | 60 | 0.933 | 1.000 | **0.955** | 0.200 | 0.957 | 0.932 | — |
| `fixed` | `bm25 heading=strip` | 60 | 0.933 | 1.000 | **0.955** | 0.200 | 0.957 | 0.932 | — |
| `fixed` | `hybrid bge-small minmax(s0.9/d0.1)` | 60 | 0.933 | 1.000 | **0.955** | 0.200 | 0.956 | 0.926 | — |
| `structural` | `bm25` | 60 | 0.833 | 0.983 | **0.951** | 0.183 | 0.897 | 0.903 | — |
| `structural` | `bm25 heading=source` | 60 | 0.850 | 0.983 | **0.951** | 0.183 | 0.903 | 0.911 | — |
| `semantic` | `hybrid bge-small minmax(s0.7/d0.3)` | 60 | 0.883 | 0.983 | **0.948** | 0.165 | 0.926 | 0.891 | — |
| `structural` | `hybrid bge-small minmax(s0.9/d0.1)` | 60 | 0.833 | 0.983 | **0.948** | 0.182 | 0.890 | 0.892 | — |
| `structural` | `hybrid bge-small wrrf(s0.9/d0.1)` | 60 | 0.333 | 0.983 | **0.943** | 0.180 | 0.547 | 0.636 | — |
| `fixed` | `hybrid bge-small wrrf(s0.9/d0.1)` | 60 | 0.350 | 0.983 | **0.932** | 0.192 | 0.582 | 0.650 | — |
| `structural` | `bm25 + rerank bge-reranker-base(c50)` | 60 | 0.867 | 0.967 | **0.932** | 0.180 | 0.907 | 0.877 | 1.000 |
| `structural` | `hybrid bge-small minmax(s0.7/d0.3)` | 60 | 0.783 | 0.967 | **0.931** | 0.177 | 0.851 | 0.849 | — |
| `fixed` | `hybrid bge-small minmax(s0.7/d0.3)` | 60 | 0.867 | 0.983 | **0.928** | 0.192 | 0.911 | 0.868 | — |
| `fixed` | `hybrid bge-small rrf + rerank bge-reranker-base(c50)` | 60 | 0.783 | 0.983 | **0.926** | 0.195 | 0.855 | 0.848 | 0.979 |
| `fixed` | `bm25 + rerank bge-reranker-base(c50)` | 60 | 0.700 | 0.983 | **0.922** | 0.193 | 0.802 | 0.806 | 0.990 |
| `structural` | `hybrid bge-small rrf + rerank bge-reranker-base(c50)` | 60 | 0.883 | 0.950 | **0.919** | 0.180 | 0.908 | 0.875 | 0.971 |
| `semantic` | `hybrid bge-small rrf + rerank bge-reranker-base(c50)` | 60 | 0.667 | 0.933 | **0.891** | 0.162 | 0.762 | 0.771 | 0.985 |
| `semantic` | `bm25 + rerank bge-reranker-base(c50)` | 60 | 0.617 | 0.900 | **0.867** | 0.160 | 0.722 | 0.735 | 1.000 |
| `semantic` | `hybrid bge-small minmax(s0.5/d0.5)` | 60 | 0.283 | 0.967 | **0.849** | 0.140 | 0.472 | 0.532 | — |
| `fixed` | `hybrid bge-small minmax(s0.5/d0.5)` | 60 | 0.200 | 0.967 | **0.796** | 0.147 | 0.451 | 0.497 | — |
| `structural` | `hybrid bge-small minmax(s0.5/d0.5)` | 60 | 0.200 | 0.933 | **0.777** | 0.132 | 0.416 | 0.479 | — |
| `semantic` | `bm25 -atoms` | 60 | 0.400 | 0.733 | **0.607** | 0.107 | 0.485 | 0.476 | — |
| `structural` | `hybrid bge-small wrrf(s0.7/d0.3)` | 60 | 0.083 | 0.700 | **0.592** | 0.097 | 0.224 | 0.288 | — |
| `fixed` | `bm25 -atoms` | 60 | 0.417 | 0.667 | **0.553** | 0.110 | 0.492 | 0.462 | — |
| `structural` | `bm25 -atoms` | 60 | 0.367 | 0.650 | **0.546** | 0.097 | 0.462 | 0.453 | — |
| `semantic` | `hybrid bge-small wrrf(s0.7/d0.3)` | 60 | 0.117 | 0.683 | **0.540** | 0.092 | 0.248 | 0.282 | — |
| `fixed` | `hybrid bge-small wrrf(s0.7/d0.3)` | 60 | 0.050 | 0.700 | **0.519** | 0.093 | 0.193 | 0.243 | — |
| `semantic` | `hybrid bge-small rrf` | 60 | 0.083 | 0.333 | **0.239** | 0.040 | 0.147 | 0.140 | — |
| `semantic` | `hybrid bge-small wrrf(s0.5/d0.5)` | 60 | 0.083 | 0.333 | **0.239** | 0.040 | 0.147 | 0.140 | — |
| `structural` | `hybrid bge-small rrf` | 60 | 0.050 | 0.283 | **0.202** | 0.033 | 0.105 | 0.107 | — |
| `structural` | `hybrid bge-small wrrf(s0.5/d0.5)` | 60 | 0.050 | 0.283 | **0.202** | 0.033 | 0.105 | 0.107 | — |
| `semantic` | `hybrid bge-small minmax(s0.3/d0.7)` | 60 | 0.050 | 0.250 | **0.175** | 0.030 | 0.104 | 0.100 | — |
| `structural` | `hybrid bge-small minmax(s0.3/d0.7)` | 60 | 0.033 | 0.233 | **0.165** | 0.027 | 0.087 | 0.088 | — |
| `fixed` | `hybrid bge-small rrf` | 60 | 0.033 | 0.283 | **0.163** | 0.035 | 0.088 | 0.088 | — |
| `fixed` | `hybrid bge-small wrrf(s0.5/d0.5)` | 60 | 0.033 | 0.283 | **0.163** | 0.035 | 0.088 | 0.088 | — |
| `fixed` | `hybrid bge-small minmax(s0.3/d0.7)` | 60 | 0.017 | 0.217 | **0.136** | 0.028 | 0.060 | 0.068 | — |
| `structural` | `hybrid bge-small wrrf(s0.3/d0.7)` | 60 | 0.050 | 0.183 | **0.132** | 0.023 | 0.078 | 0.072 | — |
| `semantic` | `hybrid bge-small wrrf(s0.3/d0.7)` | 60 | 0.033 | 0.217 | **0.130** | 0.025 | 0.080 | 0.075 | — |
| `fixed` | `hybrid bge-small wrrf(s0.3/d0.7)` | 60 | 0.017 | 0.133 | **0.085** | 0.020 | 0.043 | 0.045 | — |
| `structural` | `dense bge-small` | 60 | 0.017 | 0.050 | **0.022** | 0.008 | 0.023 | 0.017 | — |
| `semantic` | `dense bge-small` | 60 | 0.017 | 0.050 | **0.017** | 0.007 | 0.021 | 0.015 | — |
| `fixed` | `dense bge-small` | 60 | 0.017 | 0.033 | **0.013** | 0.007 | 0.019 | 0.013 | — |

A reranked row can only permute the candidate list its first stage
returned, so `rerank ceiling` is the best recall@10 that list allowed. Read
those rows against the ceiling, not against 1.000 — a row at its ceiling has
a first-stage recall problem, not a reranking one. A row *above* its own
ceiling is a wiring bug.

### `section_lookup` — primary metric **recall@10**

| Chunking | Retriever | n | hit@1 | hit@10 | recall@10 | precision@10 | mrr@10 | ndcg@10 | recall@10 ceiling |
|---|---|---|---|---|---|---|---|---|---|
| `structural` | `bm25 + rerank bge-reranker-base(c50)` | 60 | 0.983 | 1.000 | **0.935** | 0.218 | 0.986 | 0.935 | 0.935 |
| `structural` | `hybrid bge-small rrf + rerank bge-reranker-base(c50)` | 60 | 0.967 | 0.983 | **0.921** | 0.212 | 0.969 | 0.922 | 0.924 |
| `structural` | `hybrid bge-small minmax(s0.7/d0.3)` | 60 | 0.683 | 0.933 | **0.841** | 0.180 | 0.787 | 0.743 | — |
| `structural` | `hybrid bge-small minmax(s0.9/d0.1)` | 60 | 0.683 | 0.933 | **0.837** | 0.178 | 0.785 | 0.737 | — |
| `structural` | `hybrid bge-small wrrf(s0.7/d0.3)` | 60 | 0.733 | 0.933 | **0.830** | 0.177 | 0.814 | 0.750 | — |
| `structural` | `bm25` | 60 | 0.617 | 0.933 | **0.828** | 0.175 | 0.745 | 0.711 | — |
| `structural` | `bm25 -atoms` | 60 | 0.617 | 0.933 | **0.828** | 0.175 | 0.745 | 0.710 | — |
| `structural` | `hybrid bge-small wrrf(s0.9/d0.1)` | 60 | 0.667 | 0.933 | **0.828** | 0.177 | 0.769 | 0.724 | — |
| `structural` | `hybrid bge-small minmax(s0.5/d0.5)` | 60 | 0.717 | 0.917 | **0.817** | 0.175 | 0.802 | 0.755 | — |
| `structural` | `hybrid bge-small minmax(s0.3/d0.7)` | 60 | 0.717 | 0.900 | **0.800** | 0.173 | 0.781 | 0.742 | — |
| `structural` | `hybrid bge-small rrf` | 60 | 0.700 | 0.917 | **0.797** | 0.170 | 0.790 | 0.730 | — |
| `structural` | `hybrid bge-small wrrf(s0.5/d0.5)` | 60 | 0.700 | 0.917 | **0.797** | 0.170 | 0.790 | 0.730 | — |
| `structural` | `hybrid bge-small wrrf(s0.3/d0.7)` | 60 | 0.717 | 0.900 | **0.784** | 0.170 | 0.788 | 0.732 | — |
| `structural` | `dense bge-small` | 60 | 0.717 | 0.883 | **0.738** | 0.155 | 0.765 | 0.702 | — |
| `fixed` | `bm25 + rerank bge-reranker-base(c50)` | 60 | 0.833 | 0.950 | **0.620** | 0.155 | 0.873 | 0.616 | 0.678 |
| `fixed` | `hybrid bge-small rrf + rerank bge-reranker-base(c50)` | 60 | 0.833 | 0.933 | **0.599** | 0.150 | 0.869 | 0.608 | 0.697 |
| `structural` | `bm25 heading=source` | 60 | 0.433 | 0.850 | **0.587** | 0.098 | 0.577 | 0.496 | — |
| `fixed` | `hybrid bge-small minmax(s0.9/d0.1)` | 60 | 0.383 | 0.900 | **0.575** | 0.140 | 0.566 | 0.466 | — |
| `fixed` | `bm25` | 60 | 0.383 | 0.900 | **0.573** | 0.140 | 0.567 | 0.464 | — |
| `fixed` | `bm25 -atoms` | 60 | 0.383 | 0.900 | **0.573** | 0.140 | 0.571 | 0.466 | — |
| `fixed` | `bm25 heading=source` | 60 | 0.383 | 0.900 | **0.573** | 0.140 | 0.567 | 0.464 | — |
| `fixed` | `bm25 heading=strip` | 60 | 0.383 | 0.900 | **0.573** | 0.140 | 0.567 | 0.464 | — |
| `fixed` | `hybrid bge-small minmax(s0.7/d0.3)` | 60 | 0.467 | 0.900 | **0.573** | 0.142 | 0.610 | 0.492 | — |
| `fixed` | `hybrid bge-small wrrf(s0.9/d0.1)` | 60 | 0.433 | 0.900 | **0.567** | 0.138 | 0.594 | 0.479 | — |
| `semantic` | `bm25 + rerank bge-reranker-base(c50)` | 60 | 0.783 | 0.933 | **0.545** | 0.125 | 0.834 | 0.566 | 0.631 |
| `semantic` | `hybrid bge-small rrf + rerank bge-reranker-base(c50)` | 60 | 0.750 | 0.900 | **0.542** | 0.127 | 0.799 | 0.554 | 0.632 |
| `fixed` | `hybrid bge-small minmax(s0.5/d0.5)` | 60 | 0.517 | 0.850 | **0.534** | 0.135 | 0.610 | 0.478 | — |
| `fixed` | `hybrid bge-small wrrf(s0.7/d0.3)` | 60 | 0.433 | 0.817 | **0.529** | 0.133 | 0.574 | 0.465 | — |
| `fixed` | `hybrid bge-small rrf` | 60 | 0.483 | 0.817 | **0.521** | 0.132 | 0.591 | 0.464 | — |
| `fixed` | `hybrid bge-small wrrf(s0.5/d0.5)` | 60 | 0.483 | 0.817 | **0.521** | 0.132 | 0.591 | 0.464 | — |
| `semantic` | `hybrid bge-small minmax(s0.7/d0.3)` | 60 | 0.417 | 0.900 | **0.509** | 0.120 | 0.586 | 0.441 | — |
| `semantic` | `hybrid bge-small minmax(s0.9/d0.1)` | 60 | 0.400 | 0.900 | **0.507** | 0.118 | 0.581 | 0.437 | — |
| `semantic` | `bm25` | 60 | 0.383 | 0.883 | **0.503** | 0.117 | 0.561 | 0.426 | — |
| `semantic` | `bm25 -atoms` | 60 | 0.383 | 0.883 | **0.503** | 0.117 | 0.561 | 0.426 | — |
| `semantic` | `bm25 heading=source` | 60 | 0.383 | 0.883 | **0.503** | 0.117 | 0.561 | 0.426 | — |
| `semantic` | `bm25 heading=strip` | 60 | 0.383 | 0.883 | **0.503** | 0.117 | 0.561 | 0.426 | — |
| `fixed` | `hybrid bge-small minmax(s0.3/d0.7)` | 60 | 0.450 | 0.800 | **0.500** | 0.123 | 0.550 | 0.440 | — |
| `structural` | `bm25 heading=strip` | 60 | 0.317 | 0.650 | **0.495** | 0.078 | 0.436 | 0.402 | — |
| `semantic` | `hybrid bge-small wrrf(s0.9/d0.1)` | 60 | 0.400 | 0.883 | **0.491** | 0.117 | 0.571 | 0.426 | — |
| `fixed` | `hybrid bge-small wrrf(s0.3/d0.7)` | 60 | 0.483 | 0.783 | **0.484** | 0.120 | 0.572 | 0.437 | — |
| `semantic` | `hybrid bge-small minmax(s0.5/d0.5)` | 60 | 0.467 | 0.817 | **0.483** | 0.113 | 0.594 | 0.440 | — |
| `semantic` | `hybrid bge-small wrrf(s0.7/d0.3)` | 60 | 0.467 | 0.833 | **0.467** | 0.110 | 0.600 | 0.434 | — |
| `semantic` | `hybrid bge-small minmax(s0.3/d0.7)` | 60 | 0.467 | 0.750 | **0.454** | 0.107 | 0.552 | 0.414 | — |
| `semantic` | `hybrid bge-small rrf` | 60 | 0.400 | 0.783 | **0.447** | 0.105 | 0.552 | 0.405 | — |
| `semantic` | `hybrid bge-small wrrf(s0.5/d0.5)` | 60 | 0.400 | 0.783 | **0.447** | 0.105 | 0.552 | 0.405 | — |
| `semantic` | `hybrid bge-small wrrf(s0.3/d0.7)` | 60 | 0.467 | 0.767 | **0.441** | 0.103 | 0.552 | 0.406 | — |
| `fixed` | `dense bge-small` | 60 | 0.400 | 0.667 | **0.416** | 0.098 | 0.501 | 0.376 | — |
| `semantic` | `dense bge-small` | 60 | 0.433 | 0.633 | **0.388** | 0.090 | 0.498 | 0.366 | — |

A reranked row can only permute the candidate list its first stage
returned, so `rerank ceiling` is the best recall@10 that list allowed. Read
those rows against the ceiling, not against 1.000 — a row at its ceiling has
a first-stage recall problem, not a reranking one. A row *above* its own
ceiling is a wiring bug.

### `title_lookup` — primary metric **hit@1**

| Chunking | Retriever | n | hit@1 | hit@10 | recall@10 | precision@10 | mrr@10 | ndcg@10 | recall@10 ceiling |
|---|---|---|---|---|---|---|---|---|---|
| `fixed` | `bm25` | 60 | **0.967** | 1.000 | 0.234 | 0.685 | 0.983 | 0.763 | — |
| `fixed` | `bm25 -atoms` | 60 | **0.967** | 1.000 | 0.232 | 0.683 | 0.983 | 0.762 | — |
| `fixed` | `bm25 heading=source` | 60 | **0.967** | 1.000 | 0.234 | 0.685 | 0.983 | 0.763 | — |
| `fixed` | `bm25 heading=strip` | 60 | **0.967** | 1.000 | 0.234 | 0.685 | 0.983 | 0.763 | — |
| `fixed` | `hybrid bge-small wrrf(s0.9/d0.1)` | 60 | **0.967** | 1.000 | 0.237 | 0.702 | 0.983 | 0.776 | — |
| `fixed` | `hybrid bge-small minmax(s0.9/d0.1)` | 60 | **0.950** | 1.000 | 0.237 | 0.695 | 0.975 | 0.768 | — |
| `fixed` | `bm25 + rerank bge-reranker-base(c50)` | 60 | **0.950** | 1.000 | 0.237 | 0.693 | 0.975 | 0.771 | — |
| `fixed` | `hybrid bge-small rrf + rerank bge-reranker-base(c50)` | 60 | **0.950** | 1.000 | 0.248 | 0.703 | 0.975 | 0.779 | — |
| `fixed` | `hybrid bge-small wrrf(s0.7/d0.3)` | 60 | **0.933** | 1.000 | 0.243 | 0.713 | 0.964 | 0.779 | — |
| `fixed` | `hybrid bge-small minmax(s0.7/d0.3)` | 60 | **0.917** | 1.000 | 0.246 | 0.723 | 0.956 | 0.787 | — |
| `semantic` | `hybrid bge-small minmax(s0.7/d0.3)` | 60 | **0.917** | 1.000 | 0.246 | 0.713 | 0.956 | 0.779 | — |
| `structural` | `bm25 heading=source` | 60 | **0.917** | 1.000 | 0.206 | 0.687 | 0.958 | 0.753 | — |
| `structural` | `bm25 heading=strip` | 60 | **0.917** | 1.000 | 0.203 | 0.680 | 0.958 | 0.748 | — |
| `fixed` | `dense bge-small` | 60 | **0.900** | 1.000 | 0.254 | 0.702 | 0.933 | 0.774 | — |
| `fixed` | `hybrid bge-small rrf` | 60 | **0.900** | 1.000 | 0.249 | 0.717 | 0.944 | 0.782 | — |
| `fixed` | `hybrid bge-small wrrf(s0.5/d0.5)` | 60 | **0.900** | 1.000 | 0.249 | 0.717 | 0.944 | 0.782 | — |
| `fixed` | `hybrid bge-small minmax(s0.5/d0.5)` | 60 | **0.900** | 1.000 | 0.255 | 0.732 | 0.943 | 0.794 | — |
| `fixed` | `hybrid bge-small wrrf(s0.3/d0.7)` | 60 | **0.900** | 1.000 | 0.252 | 0.728 | 0.938 | 0.790 | — |
| `fixed` | `hybrid bge-small minmax(s0.3/d0.7)` | 60 | **0.900** | 1.000 | 0.256 | 0.722 | 0.939 | 0.789 | — |
| `semantic` | `hybrid bge-small rrf` | 60 | **0.900** | 1.000 | 0.250 | 0.720 | 0.947 | 0.785 | — |
| `semantic` | `hybrid bge-small wrrf(s0.9/d0.1)` | 60 | **0.900** | 1.000 | 0.228 | 0.688 | 0.950 | 0.757 | — |
| `semantic` | `hybrid bge-small minmax(s0.9/d0.1)` | 60 | **0.900** | 1.000 | 0.229 | 0.682 | 0.950 | 0.753 | — |
| `semantic` | `hybrid bge-small wrrf(s0.7/d0.3)` | 60 | **0.900** | 1.000 | 0.238 | 0.710 | 0.950 | 0.775 | — |
| `semantic` | `hybrid bge-small wrrf(s0.5/d0.5)` | 60 | **0.900** | 1.000 | 0.250 | 0.720 | 0.947 | 0.785 | — |
| `semantic` | `hybrid bge-small minmax(s0.5/d0.5)` | 60 | **0.900** | 1.000 | 0.242 | 0.703 | 0.943 | 0.773 | — |
| `semantic` | `hybrid bge-small rrf + rerank bge-reranker-base(c50)` | 60 | **0.900** | 1.000 | 0.228 | 0.688 | 0.942 | 0.754 | — |
| `structural` | `hybrid bge-small minmax(s0.3/d0.7)` | 60 | **0.900** | 1.000 | 0.234 | 0.732 | 0.947 | 0.791 | — |
| `semantic` | `bm25` | 60 | **0.883** | 1.000 | 0.227 | 0.678 | 0.942 | 0.746 | — |
| `semantic` | `bm25 -atoms` | 60 | **0.883** | 1.000 | 0.228 | 0.682 | 0.942 | 0.747 | — |
| `semantic` | `bm25 heading=source` | 60 | **0.883** | 1.000 | 0.227 | 0.678 | 0.942 | 0.746 | — |
| `semantic` | `bm25 heading=strip` | 60 | **0.883** | 1.000 | 0.227 | 0.678 | 0.942 | 0.746 | — |
| `semantic` | `hybrid bge-small wrrf(s0.3/d0.7)` | 60 | **0.883** | 1.000 | 0.250 | 0.720 | 0.935 | 0.782 | — |
| `structural` | `dense bge-small` | 60 | **0.883** | 1.000 | 0.235 | 0.730 | 0.933 | 0.785 | — |
| `structural` | `hybrid bge-small wrrf(s0.7/d0.3)` | 60 | **0.883** | 1.000 | 0.222 | 0.732 | 0.942 | 0.781 | — |
| `structural` | `hybrid bge-small minmax(s0.5/d0.5)` | 60 | **0.883** | 1.000 | 0.235 | 0.747 | 0.939 | 0.796 | — |
| `semantic` | `dense bge-small` | 60 | **0.867** | 0.983 | 0.260 | 0.700 | 0.922 | 0.769 | — |
| `semantic` | `hybrid bge-small minmax(s0.3/d0.7)` | 60 | **0.867** | 1.000 | 0.258 | 0.708 | 0.922 | 0.775 | — |
| `semantic` | `bm25 + rerank bge-reranker-base(c50)` | 60 | **0.867** | 1.000 | 0.226 | 0.685 | 0.925 | 0.748 | — |
| `structural` | `bm25` | 60 | **0.867** | 1.000 | 0.213 | 0.700 | 0.933 | 0.758 | — |
| `structural` | `bm25 -atoms` | 60 | **0.867** | 1.000 | 0.213 | 0.700 | 0.933 | 0.758 | — |
| `structural` | `hybrid bge-small rrf` | 60 | **0.867** | 1.000 | 0.224 | 0.728 | 0.933 | 0.781 | — |
| `structural` | `hybrid bge-small wrrf(s0.9/d0.1)` | 60 | **0.867** | 1.000 | 0.217 | 0.717 | 0.933 | 0.768 | — |
| `structural` | `hybrid bge-small minmax(s0.9/d0.1)` | 60 | **0.867** | 1.000 | 0.218 | 0.712 | 0.933 | 0.766 | — |
| `structural` | `hybrid bge-small wrrf(s0.5/d0.5)` | 60 | **0.867** | 1.000 | 0.224 | 0.728 | 0.933 | 0.781 | — |
| `structural` | `hybrid bge-small wrrf(s0.3/d0.7)` | 60 | **0.867** | 1.000 | 0.231 | 0.732 | 0.928 | 0.784 | — |
| `structural` | `bm25 + rerank bge-reranker-base(c50)` | 60 | **0.867** | 1.000 | 0.212 | 0.700 | 0.925 | 0.759 | — |
| `structural` | `hybrid bge-small rrf + rerank bge-reranker-base(c50)` | 60 | **0.867** | 1.000 | 0.210 | 0.702 | 0.925 | 0.760 | — |
| `structural` | `hybrid bge-small minmax(s0.7/d0.3)` | 60 | **0.850** | 1.000 | 0.228 | 0.735 | 0.925 | 0.783 | — |

The `recall@10 ceiling` column is blank here on purpose. This slice's
primary metric is **hit@1**, and the ceiling that was measured is a
*recall* ceiling — the fraction of relevant chunks the candidate list
contained. On this slice every chunk of the named document is relevant, so a
50-candidate window covers only a small fraction of them and the recall
ceiling measured 0.451-0.531 here, while `hit@1` needs just
one of them and reads near 1.0. Printing the two side by side makes a correct
row look like it beat its own ceiling, which the prose below calls a wiring
bug. The stored figure is in `retrieval_eval.json` for anyone who wants it.

### `unanswerable` — score separability

Retrieval accuracy is undefined on an empty relevant set. What matters is
whether the retriever's top score distinguishes a query whose answer exists
from one whose answer does not — that is the evidence for whether Phase 5 can
refuse on a threshold instead of guessing one.

Computed over `exact_identifier` versus `unanswerable` **only**: those two are
phrased identically and differ solely in whether the cited regulation exists,
which is the comparison that means something. An earlier version pooled all
answerable queries, so section headings and document titles — long, high-IDF
strings that score highly for unrelated reasons — made up two thirds of the
pool. That pushed the class balance to 180/60, so the headline accuracy sat
against a 0.750 majority baseline, and it reversed which chunking strategy
looked better. The baseline is now reported alongside, because an accuracy
without one is unreadable.

| Chunking | Retriever | Mean top-1 (answerable) | Mean top-1 (unanswerable) | Best threshold | Best accuracy | Baseline | Gain |
|---|---|---|---|---|---|---|---|
| `semantic` | `bm25` | 10.74 | 5.81 | 7.10 | **1.000** | 0.500 | +0.500 |
| `semantic` | `bm25 heading=source` | 10.74 | 5.81 | 7.10 | **1.000** | 0.500 | +0.500 |
| `semantic` | `bm25 heading=strip` | 10.74 | 5.81 | 7.10 | **1.000** | 0.500 | +0.500 |
| `fixed` | `bm25` | 10.76 | 5.95 | 7.45 | **0.992** | 0.500 | +0.492 |
| `fixed` | `bm25 heading=source` | 10.76 | 5.95 | 7.45 | **0.992** | 0.500 | +0.492 |
| `fixed` | `bm25 heading=strip` | 10.76 | 5.95 | 7.45 | **0.992** | 0.500 | +0.492 |
| `structural` | `bm25` | 10.99 | 6.13 | 7.73 | **0.983** | 0.500 | +0.483 |
| `structural` | `bm25 heading=source` | 10.93 | 6.02 | 7.46 | **0.983** | 0.500 | +0.483 |
| `structural` | `bm25 heading=strip` | 10.87 | 6.02 | 7.49 | **0.983** | 0.500 | +0.483 |
| `structural` | `hybrid bge-small rrf + rerank bge-reranker-base(c50)` | 0.84 | 0.13 | 0.34 | **0.942** | 0.500 | +0.442 |
| `semantic` | `bm25 -atoms` | 8.60 | 5.88 | 6.62 | **0.933** | 0.500 | +0.433 |
| `structural` | `bm25 + rerank bge-reranker-base(c50)` | 0.85 | 0.17 | 0.34 | **0.933** | 0.500 | +0.433 |
| `fixed` | `bm25 -atoms` | 8.64 | 6.01 | 7.03 | **0.925** | 0.500 | +0.425 |
| `fixed` | `bm25 + rerank bge-reranker-base(c50)` | 0.77 | 0.18 | 0.45 | **0.917** | 0.500 | +0.417 |
| `semantic` | `hybrid bge-small rrf + rerank bge-reranker-base(c50)` | 0.80 | 0.17 | 0.36 | **0.917** | 0.500 | +0.417 |
| `structural` | `bm25 -atoms` | 9.03 | 6.19 | 7.12 | **0.908** | 0.500 | +0.408 |
| `fixed` | `hybrid bge-small rrf + rerank bge-reranker-base(c50)` | 0.76 | 0.21 | 0.25 | **0.875** | 0.500 | +0.375 |
| `semantic` | `bm25 + rerank bge-reranker-base(c50)` | 0.83 | 0.25 | 0.31 | **0.867** | 0.500 | +0.367 |
| `fixed` | `dense bge-small` | 0.72 | 0.69 | 0.70 | **0.792** | 0.500 | +0.292 |
| `semantic` | `dense bge-small` | 0.71 | 0.69 | 0.70 | **0.783** | 0.500 | +0.283 |
| `structural` | `hybrid bge-small wrrf(s0.3/d0.7)` | 0.02 | 0.01 | 0.02 | **0.758** | 0.500 | +0.258 |
| `structural` | `hybrid bge-small rrf` | 0.03 | 0.03 | 0.03 | **0.733** | 0.500 | +0.233 |
| `structural` | `hybrid bge-small wrrf(s0.5/d0.5)` | 0.02 | 0.01 | 0.01 | **0.733** | 0.500 | +0.233 |
| `semantic` | `hybrid bge-small minmax(s0.3/d0.7)` | 0.77 | 0.74 | 0.70 | **0.725** | 0.500 | +0.225 |
| `structural` | `dense bge-small` | 0.72 | 0.71 | 0.71 | **0.692** | 0.500 | +0.192 |
| `fixed` | `hybrid bge-small wrrf(s0.3/d0.7)` | 0.02 | 0.02 | 0.02 | **0.675** | 0.500 | +0.175 |
| `fixed` | `hybrid bge-small minmax(s0.3/d0.7)` | 0.75 | 0.73 | 0.71 | **0.675** | 0.500 | +0.175 |
| `structural` | `hybrid bge-small wrrf(s0.7/d0.3)` | 0.02 | 0.01 | 0.02 | **0.667** | 0.500 | +0.167 |
| `structural` | `hybrid bge-small minmax(s0.3/d0.7)` | 0.74 | 0.72 | 0.72 | **0.667** | 0.500 | +0.167 |
| `semantic` | `hybrid bge-small wrrf(s0.3/d0.7)` | 0.02 | 0.02 | 0.02 | **0.658** | 0.500 | +0.158 |
| `structural` | `hybrid bge-small minmax(s0.5/d0.5)` | 0.65 | 0.59 | 0.60 | **0.658** | 0.500 | +0.158 |
| `semantic` | `hybrid bge-small rrf` | 0.03 | 0.03 | 0.03 | **0.633** | 0.500 | +0.133 |
| `semantic` | `hybrid bge-small wrrf(s0.5/d0.5)` | 0.02 | 0.01 | 0.02 | **0.633** | 0.500 | +0.133 |
| `semantic` | `hybrid bge-small wrrf(s0.7/d0.3)` | 0.02 | 0.01 | 0.02 | **0.625** | 0.500 | +0.125 |
| `fixed` | `hybrid bge-small rrf` | 0.03 | 0.03 | 0.03 | **0.600** | 0.500 | +0.100 |
| `fixed` | `hybrid bge-small wrrf(s0.5/d0.5)` | 0.02 | 0.01 | 0.02 | **0.600** | 0.500 | +0.100 |
| `structural` | `hybrid bge-small minmax(s0.9/d0.1)` | 0.91 | 0.90 | 0.93 | **0.600** | 0.500 | +0.100 |
| `structural` | `hybrid bge-small minmax(s0.7/d0.3)` | 0.74 | 0.72 | 0.80 | **0.592** | 0.500 | +0.092 |
| `structural` | `hybrid bge-small wrrf(s0.9/d0.1)` | 0.02 | 0.02 | 0.02 | **0.583** | 0.500 | +0.083 |
| `fixed` | `hybrid bge-small wrrf(s0.7/d0.3)` | 0.02 | 0.02 | 0.02 | **0.575** | 0.500 | +0.075 |
| `semantic` | `hybrid bge-small minmax(s0.7/d0.3)` | 0.75 | 0.75 | 0.86 | **0.567** | 0.500 | +0.067 |
| `semantic` | `hybrid bge-small minmax(s0.5/d0.5)` | 0.66 | 0.67 | 0.86 | **0.550** | 0.500 | +0.050 |
| `fixed` | `hybrid bge-small minmax(s0.5/d0.5)` | 0.65 | 0.67 | 0.56 | **0.542** | 0.500 | +0.042 |
| `semantic` | `hybrid bge-small wrrf(s0.9/d0.1)` | 0.02 | 0.02 | 0.01 | **0.533** | 0.500 | +0.033 |
| `semantic` | `hybrid bge-small minmax(s0.9/d0.1)` | 0.91 | 0.91 | 0.96 | **0.533** | 0.500 | +0.033 |
| `fixed` | `hybrid bge-small wrrf(s0.9/d0.1)` | 0.02 | 0.02 | 0.02 | **0.517** | 0.500 | +0.017 |
| `fixed` | `hybrid bge-small minmax(s0.9/d0.1)` | 0.91 | 0.92 | 0.97 | **0.508** | 0.500 | +0.008 |
| `fixed` | `hybrid bge-small minmax(s0.7/d0.3)` | 0.74 | 0.76 | 0.90 | **0.508** | 0.500 | +0.008 |

