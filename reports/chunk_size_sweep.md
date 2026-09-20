# Chunk-size sweep

_Generated 2026-08-17T15:47:14+00:00_

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
| **512** | 22,365 | 564 | 3,349,453 | 0.0% |
| **1,024** | 13,706 | 1,040 | 3,260,502 | 0.6% |
| **2,048** | 8,945 | 1,495 | 3,169,954 | 10.4% |
| **4,096** | 6,624 | 1,225 | 3,067,335 | 31.1% |

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
| `hybrid rrf` | 0.148 | 0.202 | 0.196 | 0.184 |

Best on this slice: **1,024 chars** with `bm25` at 0.951.

### `section_lookup` — primary metric **recall@10**

| Retriever | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|
| `bm25` | 0.809 | 0.876 | 0.897 | 0.932 |
| `dense bge-small` | 0.776 | 0.864 | 0.917 | 0.948 |
| `hybrid minmax(s0.9/d0.1)` | 0.818 | 0.881 | 0.903 | 0.935 |
| `hybrid rrf` | 0.823 | 0.893 | 0.932 | **0.953** |

Best on this slice: **4,096 chars** with `hybrid rrf` at 0.953.

### `title_lookup` — primary metric **hit@10**

| Retriever | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|
| `bm25` | 0.983 | 0.983 | 0.983 | 0.983 |
| `dense bge-small` | **1.000** | **1.000** | **1.000** | **1.000** |
| `hybrid minmax(s0.9/d0.1)` | **1.000** | **1.000** | **1.000** | **1.000** |
| `hybrid rrf` | **1.000** | **1.000** | **1.000** | **1.000** |

Best on this slice: **512 chars** with `dense bge-small` at 1.000.

