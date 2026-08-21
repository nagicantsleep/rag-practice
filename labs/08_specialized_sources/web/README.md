# M08.1 — Web RAG

Status: **IN PROGRESS** — implementation/evaluation candidate pending CI evidence.

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

The extractive answerer returns the top page verbatim. This is deliberate: groundedness should be `1.0` even when the wrong/stale page wins, making the separation between **grounded** and **correct/current** visible.

## Definition of Done

- [x] source/tool contract implemented without a framework
- [x] deterministic web-snapshot benchmark defined
- [x] body-only and metadata BM25 baselines implemented
- [x] freshness/authority policy implemented
- [x] canonical deduplication implemented
- [x] retrieval/source and answer metrics separated
- [x] regression tests added
- [ ] CI full-suite + Web RAG evaluator passes
- [ ] machine-readable and human-readable results reviewed
- [ ] representative stale/authority/duplicate failures written down
- [ ] ROADMAP marks Web RAG sub-lab DONE only after evidence passes
