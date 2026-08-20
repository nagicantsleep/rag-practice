# Lab 07 — Hierarchical, Graph, and Memory-oriented RAG

Status: `IN PROGRESS`.

M07 isolates structural retrieval before adding another generator. The question is not whether a graph/tree sounds more advanced than BM25, but **which structure helps which information need, at what construction/update cost, and with what failure modes?**

## Hypothesis

- hierarchical summaries should improve collection-wide evidence packaging when leaf text omits the collection name;
- explicit graph paths should improve local and multi-hop relation retrieval;
- community expansion should improve global relation evidence coverage;
- associative graph propagation should help bridge two query entities without a direct lexical match;
- version-aware memory should retrieve the current fact without destroying access to historical versions;
- no single structure should dominate every task class.

## Controlled benchmark

`benchmarks/m07_structured/` contains one static corpus plus a separate temporal-memory stream.

Static query classes: `local`, `multi_hop`, `associative`, `global_relation`, and `hierarchical`. Memory classes: `memory_current` and `memory_previous`.

The static documents include controlled triples. This deliberately **isolates graph retrieval/reasoning from information extraction**; M07 does not claim to solve open-domain KG extraction.

## Systems

Baselines / controls:

- flat text-only BM25;
- flat metadata-enriched BM25.

Mechanisms:

- `RaptorStyleIndex` — leaves → deterministic source groups → extractive collection summaries; recursive-summary/hierarchical-routing mechanics without RAPTOR's learned clustering or generative summaries;
- `KAGPathRetriever` — query entity/relation detection plus provenance-preserving shortest-path reasoning;
- `GlobalGraphRetriever` — GraphRAG-style network/community expansion for global evidence;
- `HippoRAGRetriever` — query-seeded personalized PageRank; multi-seed queries use bridge scoring so evidence must be associated with every query entity;
- `TemporalMemoryIndex` — relevance selects a memory key, temporal policy selects current or previous version.

These are transparent educational mechanisms, not claims of reproducing the trained/full research systems.

## Evaluation contract

Static retrieval records Recall@3/5/10, recall/evidence completeness at the exact evidence budget, MRR, task-class breakdown, build/query latency, structural footprint, and per-query rankings.

Memory records Hit@1, current/previous Hit@1, stale-current rate, build/update/query latency, event/key counts, and per-query rankings.

Generation is intentionally **not involved** in this milestone phase: M07 isolates whether the retrieval/context structure supplies the required evidence. A language model is therefore not allowed to hide structural retrieval failures.

## Definition of Done

- [x] learning objective and hypothesis written
- [x] shared benchmark and explicit task classes defined
- [x] flat BM25 baseline and metadata control defined
- [x] hierarchical summary tree implemented
- [x] graph path / global community / associative PPR retrieval implemented
- [x] temporal current + historical memory behavior implemented
- [x] core behavior covered by automated tests
- [x] retrieval and memory metrics separated from any generation
- [x] construction/update/query cost instrumentation implemented
- [ ] persisted CI benchmark results reviewed
- [ ] representative failures and trade-offs written down
- [ ] ROADMAP M07 updated to `DONE`
- [ ] final completion tree passes full repository CI + M07 evaluation

M07 is not merged until every unchecked item is satisfied.
