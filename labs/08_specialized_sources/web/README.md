# M08.1 — Web RAG

Status: **DONE** — final full-suite + Web RAG gate passed in GitHub Actions run `32447405977`.

## Hypothesis

A web-augmented RAG system needs source semantics that a static corpus does not expose. In particular:

- lexical relevance alone can rank a stale but query-shaped page first;
- recency alone can rank a freshly updated low-authority page above an official source;
- duplicate/mirrored pages can waste a small evidence budget;
- an answer can be perfectly grounded in the retrieved web page and still be wrong because the page is stale or misleading.

The sub-lab therefore keeps source acquisition, ranking policy, answer extraction, and evaluation separate.

## Mechanism

Shared M08 contract:

`Source.search(query, limit) -> SourceHit[SourceRecord]`

Web-specific implementation:

`SnapshotWebSource -> BM25 candidates -> WebRankingPolicy -> canonical dedupe -> top-k web evidence -> extractive answer + URL citation`

The ranking policy is deliberately transparent. For explicit current/latest intent it fuses normalized lexical relevance, source authority, and recency. For static/historical intent it removes the freshness term so old evidence is not automatically punished. Canonical URLs are used to collapse mirrored pages after scoring.

No orchestration framework is used.

## Controlled benchmark

`benchmarks/m08_web/` contains 14 frozen web pages and 8 held-out queries. It includes:

- official current and historical pages;
- stale pages that still say they are "current";
- fresh low-authority forum/blog conflicts;
- exact mirrored/canonical duplicates;
- current, historical, authority-conflict, duplicate, and static questions.

The evaluation date is pinned to `2026-08-20`.

The frozen snapshot is intentional: a live search engine would make CI non-reproducible and would change both method and benchmark between runs. The `Source` boundary is what later lets the snapshot adapter be replaced by a real search provider without changing the RAG/evaluation contract.

Research motivation: WebGPT (arXiv:2112.09332) makes evidence collection/citations explicit, while FreshLLMs/FreshQA (arXiv:2310.03214) focuses on fast-changing knowledge and search-engine augmentation.

## Baselines

1. body-only BM25;
2. domain/title/body metadata BM25;
3. metadata BM25 candidates + query-aware authority/freshness reranking + canonical dedupe.

All systems use the same pages, queries, candidate budget, and extractive answerer.

## Evaluation contract

Retrieval/source quality:

- Hit@1, Recall@3, MRR;
- stale-top1 rate on freshness-sensitive queries;
- low-authority-top1 rate;
- canonical duplicate rate@3.

Answer/citation behavior:

- answer contains reference;
- grounded-answer rate;
- top URL citation trace.

System behavior:

- source calls;
- search, rerank, and end-to-end latency.

The extractive answerer returns the top page verbatim. This is deliberate: groundedness can be `1.0` even when the wrong/stale page wins, making the separation between **grounded** and **correct/current** visible.

## Persisted evaluation

Initial CI run `32447107848` passed the full repository suite (**82 tests**) and the Web RAG evaluator.

Final source-of-truth gate `32447405977` passed the same full-suite/evaluator sequence before this completion update.

| System | Hit@1 | Recall@3 | MRR | Stale top1 | Low-authority top1 | Duplicate@3 | Answer contains ref | Grounded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| body BM25 | 0.500 | 0.875 | 0.688 | 0.400 | 0.375 | 0.167 | 0.500 | **1.000** |
| metadata BM25 | 0.375 | **1.000** | 0.667 | 0.800 | 0.625 | 0.167 | 0.375 | **1.000** |
| Web policy | **1.000** | **1.000** | **1.000** | **0.000** | **0.000** | **0.000** | **1.000** | **1.000** |

The policy weights were not tuned after seeing these test results.

## Error analysis / findings

- **Metadata can hurt ranking.** Adding domain/title text increases Recall@3 to `1.0`, but Hit@1 drops from `0.500` to `0.375`. Query-shaped forum/blog titles strengthen the wrong pages more than the official answer pages.
- **Freshness alone is not enough.** `w1`, `w2`, and `w3` expose pages that were updated very recently but are stale or low-authority. A recency-only policy would still be vulnerable.
- **Groundedness is not freshness or correctness.** Both BM25 baselines have grounded-answer rate `1.0` because the answer is copied verbatim from the selected page, while answer correctness is only `0.500`/`0.375`.
- **Canonical duplicates waste evidence budget.** On `w4`, the official security advisory and its mirror occupy two of three lexical slots. Mean duplicate rate@3 is `0.167` for both baselines and `0.0` after canonical collapse.
- **Historical questions need different temporal semantics.** `w5` asks for the previous release; the policy deliberately disables the freshness term for that intent and keeps the historical page at rank 1.
- **A single authority scalar is only a teaching control.** Production trust should depend on provenance, publisher type, corroboration, claim type, and possibly domain-specific policy; this lab only isolates the mechanism.
- **Perfect policy scores are a benchmark limitation, not a general Web RAG claim.** The snapshot is tiny and deliberately constructed around known failure modes. The next step should not add more tuning to this same test set.

Machine-readable evidence: `results/results.json`. Human-readable aggregate table: `results/results.md`.

## Definition of Done

- [x] source/tool contract implemented without a framework
- [x] deterministic web-snapshot benchmark defined
- [x] body-only and metadata BM25 baselines implemented
- [x] freshness/authority policy implemented
- [x] canonical deduplication implemented
- [x] retrieval/source and answer metrics separated
- [x] regression tests added
- [x] CI full-suite + Web RAG evaluator passes
- [x] machine-readable and human-readable results reviewed
- [x] representative stale/authority/duplicate failures written down
- [x] ROADMAP marks Web RAG sub-lab DONE only after the final gate passes

Web RAG satisfies the sub-lab evaluation contract and is eligible to merge; M08 overall remains IN PROGRESS.
