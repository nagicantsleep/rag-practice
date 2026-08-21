# M07 Completion Summary — Hierarchical, Graph, and Memory-oriented RAG

## Hypothesis

Different structural retrieval mechanisms should solve different information needs: hierarchical summaries for collection-wide evidence, explicit graph paths for relation chains, community traversal for global aggregation, associative propagation for bridge queries, and version-aware memory for freshness. The predeclared hypothesis explicitly rejects the assumption that one structure should dominate every task class.

## Controls

- shared 19-document static corpus across every static retrieval system;
- flat text-only BM25 baseline and flat metadata-enriched BM25 control;
- exact same 10 static held-out queries for all systems;
- exact evidence-budget completeness so retrieving only one relevant document cannot pass a multi-evidence query;
- separate temporal stream with old and updated facts indexed together by the flat memory baseline;
- runtime receives no qrels, answer references, or task labels;
- gold triples deliberately isolate graph retrieval/reasoning from information extraction;
- no generator is present, so generation cannot hide retrieval failure;
- LightRAG-style dual-level routing is explicitly tagged post-hoc exploratory because it was added after phase-1 benchmark inspection.

## Persisted static results

| System | Recall@3 | Recall@5 | Evidence complete@budget | MRR | Mean query ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat BM25 | 0.447 | 0.567 | 0.100 | 0.660 | 0.027 |
| flat metadata BM25 | 0.497 | 0.700 | 0.100 | 0.708 | 0.025 |
| RAPTOR-style hierarchy | 0.630 | 0.683 | 0.300 | 0.725 | 0.151 |
| KAG-style path | 0.717 | 0.717 | 0.600 | 0.800 | 0.024 |
| GraphRAG-style global | 0.580 | 0.767 | 0.600 | 0.808 | 0.066 |
| HippoRAG-style PPR | 0.630 | 0.767 | 0.200 | 0.808 | 0.286 |
| LightRAG-style dual-level — exploratory | 0.880 | 0.933 | 1.000 | 1.000 | 0.047 |

### Predeclared task evidence completeness

| System | Local | Multi-hop | Associative | Global | Hierarchical |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat BM25 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 |
| RAPTOR-style | 0.500 | 0.000 | 0.000 | 0.000 | **1.000** |
| KAG-style path | **1.000** | **1.000** | **1.000** | 0.000 | 0.000 |
| GraphRAG-style global | **1.000** | 0.000 | 0.000 | **1.000** | **1.000** |
| HippoRAG-style PPR | 0.500 | 0.000 | **1.000** | 0.000 | 0.000 |

The LightRAG-style controller routes local/multi-hop/associative queries to the low-level path mechanism and global/hierarchical queries to the high-level community mechanism, producing perfect evidence completeness on this benchmark. Because that controller was designed after observing the table above, this is a **post-hoc mechanism demonstration**, not fresh held-out evidence.

## Temporal memory result

| System | Hit@1 | Current Hit@1 | Previous Hit@1 | Stale-current rate | Mean query ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat BM25 all versions | 0.250 | 0.000 | 1.000 | 1.000 | 0.019 |
| temporal memory | **1.000** | **1.000** | **1.000** | **0.000** | 0.025 |

The temporal index records 6 events over 3 keys after updates. Latest persisted mean update cost is roughly `0.028 ms` on the CI toy corpus.

## Error analysis and findings

1. **Evidence completeness is more diagnostic than MRR for structured questions.** Flat BM25 has MRR `0.660`, yet Evidence Complete@budget is only `0.100`. A relevant first hit does not imply that a graph chain or global evidence set is usable.
2. **Hierarchy packages global-within-collection evidence but does not execute relations.** RAPTOR-style retrieval gets both hierarchical questions complete, while the Atlas-country global query returns an empty result after the routed subgroup has no lexical bridge. The system now fails closed instead of constructing an illegal empty BM25 index.
3. **KAG-style path reasoning is excellent when the information need maps to one explicit relation path.** All local, three-hop currency, and the controlled associative path queries are complete. It is not an enumerator: broad network-level questions remain incomplete.
4. **GraphRAG-style community expansion reverses that trade-off.** It completes global and collection-wide evidence but over-expands entity-specific relation questions, mixing unrelated country/currency edges before all path evidence fits the budget.
5. **HippoRAG-style associative propagation finds bridges but does not preserve ordered relation semantics.** Multi-seed PPR recovers the Atlas↔euro bridge, while 3-hop currency chains generally place only two of three evidence documents inside the exact budget.
6. **Temporal freshness requires version policy.** Lexical BM25 ranks old/current versions by lexical score and therefore returns the stale version first for every current-fact query. Explicit sequence-aware selection reaches current Hit@1 `1.0` while retaining previous-version Hit@1 `1.0`.
7. **Routing complementary structures is powerful but easy to overfit.** The LightRAG-style low/high controller is perfect on the inspected teaching benchmark. The correct scientific conclusion is that low/high retrieval modes are complementary; the perfect score itself requires fresh validation and is not evidence of generalization.
8. **Construction and update costs belong in the comparison.** The hierarchy stores 10 summary nodes/236 summary words over 19 leaves; the graph stores 24 nodes/24 edges; temporal memory maintains 6 events/3 keys. Timings are persisted but too small to extrapolate to serving scale.

## Limitations

This is a mechanism benchmark, not a research-system reproduction or production leaderboard. It uses 19 synthetic/template documents, 10 static queries, 4 temporal queries, gold triples, deterministic grouping, rule-based relation/entity recognition, clean graph roots, and explicit event sequence metadata. Graph extraction quality, learned community summarization, semantic entity linking, real long-term memory consolidation, and large-scale graph/index serving remain outside M07.

## Evaluation evidence

- PR run `32423393947`: 77 repository tests passed after the fail-closed hierarchy regression fix; M07 evaluation succeeded.
- PR run `32423789873`: 78 tests passed after adding the explicit LightRAG-style dual-level mechanism; M07 evaluation succeeded.
- `results.json` persists per-query rankings, task classes, exact evidence-budget metrics, timings, structure sizes, temporal traces, dual-level route decisions, and the post-hoc flag.
- `results.md` persists the compact aggregate table.
- Generation evaluation is N/A by design because M07 intentionally evaluates structural retrieval independently.

## Completion decision

The mechanism/evaluation contract is satisfied. M07 remains a completion candidate until the documentation/source-of-truth tree itself passes the same full repository CI + M07 evaluator. Only then may the ROADMAP completion and merge be finalized.
