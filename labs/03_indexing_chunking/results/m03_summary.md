# M03 Completion Summary

Status: **DONE**  
Benchmark: `benchmarks/m03_chunking@v1`  
Retriever control: BM25 at every searchable layer  
Final evaluation CI: `32407289218` — **39 tests passed**

## What M03 isolates

M03 changes chunk/index representation while keeping the retrieval scoring family fixed. This prevents a better encoder or different ranking model from being mistaken for a chunking improvement.

## Phase 1

| Strategy | Doc Hit@1 | Evidence@1 | Evidence@3 | Source-token utilization@3 |
| --- | ---: | ---: | ---: | ---: |
| Fixed 24 | 1.000 | 0.200 | 0.800 | 1.000 |
| Fixed 24 + overlap 8 | 1.000 | 0.400 | 1.000 | 0.886 |
| Sentence 35 | 0.800 | 0.800 | 1.000 | 1.000 |
| Paragraph 80 | 0.800 | 0.800 | 1.000 | 1.000 |
| Hashing-similarity semantic boundaries | 0.800 | 0.200 | 0.800 | 1.000 |
| Sentence 35 + metadata | 1.000 | 0.800 | 1.000 | 0.635 |

Overlap improves evidence coverage but duplicates context. Sentence/paragraph boundaries package evidence much better at rank 1, but body-only retrieval cannot distinguish the Arctic and Tropical documents. Repeating metadata in every sentence chunk fixes routing at a substantial context-utilization cost. The simple semantic-boundary heuristic over-splits and is retained as a negative result.

## Phase 2

| Strategy | Doc Hit@1 | Evidence@1 | Evidence@3 | Utilization@3 | Searchable words | Route Hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed 24 + overlap 8 | 1.000 | 0.400 | 1.000 | 0.886 | 367 | — |
| Sentence 35 + metadata | 1.000 | 0.800 | 1.000 | 0.635 | 434 | — |
| Parent-child | 0.800 | 0.600 | 1.000 | 1.000 | 271 | — |
| Hierarchical metadata root → plain sentence leaf | 1.000 | 1.000 | 1.000 | 1.000 | 601 | 1.000 |

The hierarchy uses metadata only in its document-level route representation. Returned leaves remain plain source sentences, so the context does not pay the metadata-prefix penalty. It solves every held-out route and gives complete rank-1 evidence, but its searchable representation is the largest. Parent-child has the smallest searchable representation and perfect context utilization, yet cannot solve the metadata ambiguity by itself.

## Error analysis retained

- `q3` deliberately distinguishes Arctic from Tropical only through metadata. Body-only sentence, paragraph, semantic, and parent-child strategies rank Tropical first; metadata-flat and hierarchical strategies recover Arctic.
- `semantic_50` produces 19 chunks and Evidence@1 `0.2`; this is evidence that a weak boundary representation can make “semantic chunking” worse than simpler natural boundaries.
- fixed 24-word chunks achieve Doc Hit@1 `1.0` but Evidence@1 `0.2`; document hit alone is therefore insufficient for evaluating RAG chunking.

## Scope

This is a tiny controlled mechanism benchmark, not a universal chunking leaderboard. GitHub Actions CPU build/query timings are sanity measurements. Generation evaluation is intentionally out of scope; M03 isolates retrieval/index representation and selected-context quality.
