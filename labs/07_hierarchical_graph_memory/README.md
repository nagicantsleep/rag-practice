# Lab 07 — Hierarchical, Graph, and Memory-oriented RAG

Status: **COMPLETION CANDIDATE** — final full-suite/evaluation gate pending.

Final-gate trigger note: this documentation-only commit intentionally leaves the M07 implementation, benchmark, tests, and evaluator unchanged; it exists only to run the repaired M07 workflow against the exact completion candidate tree.

M07 isolates structural retrieval before adding another generator. The question is not whether a graph, tree, or memory system sounds more advanced than BM25, but **which structure helps which information need, at what construction/update cost, and with what failure modes?**

## Hypothesis

- hierarchical summaries should improve collection-wide evidence packaging when leaf text omits the collection name;
- explicit graph paths should improve local and multi-hop relation retrieval;
- community expansion should improve global relation evidence coverage;
- associative graph propagation should help bridge two query entities without a direct lexical match;
- version-aware memory should retrieve the current fact without destroying access to historical versions;
- no single predeclared structure should dominate every task class.

## Controlled benchmark

`benchmarks/m07_structured/` contains one static corpus plus a separate temporal-memory stream.

Static query classes are `local`, `multi_hop`, `associative`, `global_relation`, and `hierarchical`. Memory classes are `memory_current` and `memory_previous`.

The 19 static documents include controlled triples. This deliberately **isolates graph retrieval/reasoning from information extraction**; M07 does not claim to solve open-domain KG extraction. Runtime retrieval never receives qrels, reference answers, or task labels.

## Systems

Baselines / controls:

- flat text-only BM25;
- flat metadata-enriched BM25;
- flat BM25 over all memory versions for freshness comparison.

Mechanisms:

- `RaptorStyleIndex` — leaves → deterministic source groups → extractive collection summaries; recursive-summary/hierarchical-routing mechanics without RAPTOR's learned clustering or generative summaries;
- `KAGPathRetriever` — query entity/relation detection plus provenance-preserving shortest-path reasoning;
- `GlobalGraphRetriever` — GraphRAG-style network/community expansion for global evidence;
- `HippoRAGRetriever` — query-seeded personalized PageRank; multi-seed queries use bridge scoring so evidence must be associated with every query entity;
- `TemporalMemoryIndex` — relevance selects a memory key, temporal policy selects current or previous version;
- `DualLevelGraphRetriever` — LightRAG-style transparent low/high controller between path and global graph retrieval.

These are transparent educational mechanisms, not claims of reproducing the trained/full research systems.

**Important evaluation caveat:** `DualLevelGraphRetriever` was added after inspecting the first M07 benchmark results. Its numbers are therefore recorded as **post-hoc exploratory**, not as fresh held-out evidence. The scientific conclusions below use the predeclared systems; a fresh untouched benchmark or separate route-development split is required before claiming generalization for the dual-level controller.

## Static retrieval results

Persisted CI artifact:

| System | Recall@3 | Recall@5 | Evidence complete@budget | MRR | Mean query ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat BM25 | 0.447 | 0.567 | 0.100 | 0.660 | 0.027 |
| flat metadata BM25 | 0.497 | 0.700 | 0.100 | 0.708 | 0.025 |
| RAPTOR-style hierarchy | 0.630 | 0.683 | 0.300 | 0.725 | 0.151 |
| KAG-style path | **0.717** | 0.717 | **0.600** | 0.800 | **0.024** |
| GraphRAG-style global | 0.580 | **0.767** | **0.600** | **0.808** | 0.066 |
| HippoRAG-style PPR | 0.630 | **0.767** | 0.200 | **0.808** | 0.286 |
| LightRAG-style dual-level — post-hoc exploratory | 0.880 | 0.933 | 1.000 | 1.000 | 0.047 |

The exact-evidence-budget metric is the primary completeness metric: a 3-document relation chain only passes if all three evidence documents are retrieved within a budget of three, and a global query only passes when its full evidence set fits inside its corresponding budget.

### Predeclared task specialization

| System | Local EC | Multi-hop EC | Associative EC | Global EC | Hierarchical EC |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat BM25 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 |
| RAPTOR-style | 0.500 | 0.000 | 0.000 | 0.000 | **1.000** |
| KAG-style path | **1.000** | **1.000** | **1.000** | 0.000 | 0.000 |
| GraphRAG-style global | **1.000** | 0.000 | 0.000 | **1.000** | **1.000** |
| HippoRAG-style PPR | 0.500 | 0.000 | **1.000** | 0.000 | 0.000 |

The table is deliberately not a universal leaderboard. It exposes where each structural assumption helps and where it breaks.

## Temporal memory results

| System | Hit@1 | Current Hit@1 | Previous Hit@1 | Stale-current rate | Mean query ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat BM25 over all versions | 0.250 | 0.000 | 1.000 | 1.000 | 0.019 |
| temporal memory | **1.000** | **1.000** | **1.000** | **0.000** | 0.025 |

The temporal index retains 6 events across 3 memory keys after three updates. In the persisted run its mean update cost is roughly `0.028 ms` on the tiny CI corpus. These timings are mechanism sanity measurements, not serving benchmarks.

## Representative failures retained

- **MRR can look healthy while evidence is incomplete.** Flat BM25 reaches MRR `0.660` but only `0.100` Evidence Complete@budget; returning one relevant document early does not solve a multi-evidence information need.
- **Hierarchy is not graph reasoning.** RAPTOR-style retrieval is perfect on the two collection-wide hierarchical queries but its Atlas-country global query deliberately fails closed with `[]` once the root route has no matching subgroup evidence. The regression test keeps this behavior from becoming a crash or fabricated candidate set.
- **Path retrieval is local, not a global aggregator.** KAG-style retrieval exactly recovers the three controlled currency chains but cannot enumerate a network-wide evidence set; the broad global query returns no path and the Atlas-country query only follows a partial path.
- **Global expansion can over-broaden local reasoning.** GraphRAG-style expansion gets complete global/hierarchical evidence but mixes community facts on entity-specific currency chains, so multi-hop Evidence Complete@budget is `0`.
- **Associative diffusion is not ordered path execution.** HippoRAG-style PPR solves the two-seed Atlas↔euro association, but diffusion usually places only two of three ordered relation-chain documents inside the exact evidence budget.
- **Freshness is not implied by lexical relevance.** Flat BM25 ranks `mem1` before `mem4`, `mem2` before `mem5`, and `mem3` before `mem6`; explicit temporal/version policy is what removes stale-current retrieval while preserving historical access.
- **A routed combination can hide development leakage.** The LightRAG-style controller achieves perfect evidence completeness on this tiny benchmark by choosing complementary low/high mechanisms, but the controller was designed after inspecting phase-1 failures. That result is useful mechanistic evidence about routing, not a valid untouched-test generalization claim.

## Structural cost / scope

- RAPTOR-style representation: 19 leaves, 10 summary nodes, 236 summary words.
- Controlled knowledge graph: 24 entities/nodes and 24 provenance-carrying edges.
- Temporal memory after updates: 6 events and 3 versioned keys.
- Query/build/update latencies are recorded per run in `results/results.json`; corpus scale is too small for production performance conclusions.

## Limitations

The benchmark is intentionally tiny and template-like: 19 static documents, 10 static queries, and 4 memory queries. Graph triples are gold annotations, relation/entity detection is rule-based, RAPTOR grouping is deterministic rather than learned, GraphRAG-style traversal assumes clean directed network roots, and temporal behavior relies on explicit sequence metadata. The post-hoc LightRAG-style controller additionally requires a fresh benchmark before any claim beyond this teaching corpus.

Generation/groundedness evaluation is **not applicable to M07 by design**: no generator participates in this milestone. M07 isolates whether each structure retrieves the required evidence independently, so an LLM cannot mask structural retrieval failures.

## Evaluation evidence

- M07 post-fix PR run `32423393947`: **77 tests passed** and the hierarchy/graph/memory evaluator succeeded.
- M07 dual-level PR run `32423789873`: **78 tests passed** and the evaluator succeeded with LightRAG-style route traces persisted.
- Machine-readable rankings, per-task metrics, timings, structure sizes, memory freshness traces, and the post-hoc evaluation flag are persisted in `results/results.json`.
- Human-readable aggregate results are persisted in `results/results.md`; full interpretation is in `results/m07_summary.md`.

## Definition of Done

- [x] learning objective and hypothesis written
- [x] shared benchmark and explicit task classes defined
- [x] flat BM25 baseline and metadata control defined
- [x] hierarchical summary tree implemented
- [x] graph path / global community / associative PPR retrieval implemented
- [x] LightRAG-style low/high graph controller implemented with post-hoc caveat
- [x] temporal current + historical memory behavior implemented
- [x] core behavior covered by automated tests
- [x] retrieval and memory metrics separated from any generation
- [x] construction/update/query cost instrumentation implemented
- [x] persisted CI benchmark results reviewed
- [x] representative failures and trade-offs written down
- [ ] final completion tree passes full repository CI + M07 evaluation and ROADMAP is updated on that successful run

M07 is not merged until the final unchecked gate passes.
