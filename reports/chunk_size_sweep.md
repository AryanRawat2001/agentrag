# Chunk-size sweep

_Generated 2026-09-20T11:03:24+00:00_

`structural` chunking at 4 target sizes, scored with the
same golden set and the same retrievers as the ablation table. Overlap is held
at 15% of target throughout, so size is the only variable.

Chunk size is the parameter most RAG tutorials pick by feel and never revisit.
It trades two effects against each other: smaller chunks are more precise (less
irrelevant text around a match) but fragment sections, so a query answered by a
whole section needs more of them retrieved to score the same recall.

## Corpus shape by size

| Target chars | Chunks | Median chars | Index tokens (est, chars/4) | % truncated at 512 tok (real) |
|---|---|---|---|---|
| **512** | 21,896 | 564 | 3,281,106 | 0.0% |
| **1,024** | 13,423 | 1,040 | 3,193,185 | 0.0% |
| **2,048** | 8,759 | 1,494 | 3,103,675 | 9.8% |
| **4,096** | 6,484 | 1,227 | 3,002,030 | 31.0% |

The truncation column is the interaction worth noticing: the embedding model's
512-token window is a fixed budget, so the share of chunks it clips is a
*function of chunk size*. A size chosen purely for retrieval precision can be
silently discarding text before the vector is ever computed.

## Retrieval by size

### `exact_identifier` — primary metric **recall@10**

| Retriever | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|
| `bm25` | 0.946 | **0.951** | 0.943 | 0.936 |
| `dense bge-small` | 0.011 | 0.022 | 0.019 | 0.042 |
| `hybrid minmax(s0.9/d0.1)` | 0.942 | 0.948 | 0.931 | 0.936 |
| `hybrid rrf` | 0.148 | 0.202 | 0.200 | 0.184 |

Best on this slice: **1,024 chars** with `bm25` at 0.951.

### `section_lookup` — primary metric **recall@10**

| Retriever | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|
| `bm25` | 0.753 | 0.828 | 0.872 | 0.900 |
| `dense bge-small` | 0.699 | 0.738 | 0.831 | 0.867 |
| `hybrid minmax(s0.9/d0.1)` | 0.772 | 0.837 | 0.872 | 0.900 |
| `hybrid rrf` | 0.742 | 0.797 | 0.865 | **0.917** |

Best on this slice: **4,096 chars** with `hybrid rrf` at 0.917.

### `title_lookup` — primary metric **hit@1**

| Retriever | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|
| `bm25` | 0.867 | 0.867 | 0.933 | 0.950 |
| `dense bge-small` | 0.867 | 0.883 | 0.883 | 0.917 |
| `hybrid minmax(s0.9/d0.1)` | 0.867 | 0.867 | 0.933 | **0.967** |
| `hybrid rrf` | 0.883 | 0.867 | 0.900 | 0.917 |

Best on this slice: **4,096 chars** with `hybrid minmax(s0.9/d0.1)` at 0.967.

