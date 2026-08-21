# M08.3 — Metadata / Filter-aware RAG

Status: **COMPLETION CANDIDATE** — final source-of-truth gate pending.

## Hypothesis

Filter placement is part of retrieval correctness. An unfiltered retriever can have excellent apparent Recall while exposing a different tenant or an unauthorized role. Filtering only after top-k retrieval can remove the leak from the returned list but still lose the relevant authorized document because invalid candidates consumed the ranking budget.

## Mechanism

The corpus uses the shared M08 `SourceRecord` shape with explicit metadata:

- hard authorization: `tenant`, `allowed_roles`;
- explicit query filters: `product`, `region`, `updated_at` bounds.

Four systems hold BM25 text/scoring fixed:

1. unfiltered BM25 — relevance-only control;
2. post-filter BM25 with candidate `k=2`;
3. post-filter BM25 with oversampling `k=5`;
4. pre-filter BM25 — apply authorization/query predicates before building the lexical candidate index.

Hard security predicates are never treated as soft relevance boosts in the candidate system.

## Controlled benchmark

`benchmarks/m08_metadata/` contains 19 records across alpha/beta/shared tenants and 9 queries covering:

- cross-tenant lexical collisions;
- role/ACL collisions inside one tenant;
- product + region filters;
- time bounds;
- empty filter result;
- the same query under two tenant identities.

Several wrong-tenant/wrong-filter records are deliberately more lexically similar than the authorized relevant record. This makes the distinction between ranking quality and constraint correctness observable.

## Evaluation contract

Retrieval/filter quality:
- Recall@3 and Hit@1 against filter-aware qrels;
- constraint satisfaction rate;
- security leakage rate;
- explicit-filter violation rate;
- empty-filter accuracy.

Answer behavior:
- top-record answer correctness;
- grounded-answer rate when an answer is returned.

System behavior:
- records indexed for lexical search;
- ranked candidates examined;
- records rejected after ranking;
- eligible-record count;
- latency.

## Persisted evaluation

Initial PR gate `32450243891` passed the full repository suite (**95 tests**) and the metadata/filter evaluator.

| System | Recall@3 | Hit@1 | Constraint satisfied | Security leakage | Filter violation | Empty correct | Answer correct | Indexed records | Examined candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unfiltered BM25 | **1.000** | 0.250 | 0.333 | 0.444 | 0.667 | 0.000 | 0.222 | 19.0 | 3.0 |
| post-filter k=2 | 0.375 | 0.375 | **1.000** | **0.000** | **0.000** | **1.000** | 0.444 | 19.0 | 2.0 |
| post-filter oversample k=5 | **1.000** | **1.000** | **1.000** | **0.000** | **0.000** | **1.000** | **1.000** | 19.0 | 5.0 |
| pre-filter BM25 | **1.000** | **1.000** | **1.000** | **0.000** | **0.000** | **1.000** | **1.000** | **2.7** | **1.4** |

## Error analysis / findings

- **High Recall can coexist with a security failure.** Unfiltered BM25 retrieves the relevant record somewhere in top-3 for every non-empty query, yet leaks unauthorized records on `44.4%` of queries and violates explicit filters on `66.7%`.
- **Post-filtering can be safe but incomplete.** With candidate `k=2`, invalid records consume the lexical window before filtering, reducing Recall@3 to `0.375` even though returned records satisfy all constraints.
- **Oversampling buys recall with candidate cost.** Increasing the post-filter window to `k=5` restores Recall/Hit/answer correctness to `1.0`, but every query still indexes all 19 records, examines 5 candidates, and rejects about 3.56 after ranking on average.
- **Pre-filtering separates authorization from relevance.** The pre-filter path reaches the same `1.0` controlled quality with zero leakage while indexing only about `2.7` eligible records and examining `1.4` candidates per query on this tiny corpus.
- **Empty-filter semantics matter.** `m6` has no eligible record. Unfiltered retrieval still returns unrelated records; filter-aware systems correctly return no answer/evidence.
- **Groundedness does not imply authorization.** Returned answers are extractive, so groundedness is `1.0` whenever a system answers—even the unfiltered system that cites an unauthorized record.
- **Application-side predicates are only a teaching mechanism.** Production systems should enforce tenant/ACL constraints at an authoritative storage/index boundary and defend against identity-policy drift, cache leakage, and side channels.
- **The benchmark is intentionally small and adversarial.** The perfect pre-filter/oversampling scores demonstrate filter placement mechanics, not general retrieval or IAM performance.

Machine-readable evidence: `results/results.json`. Human-readable aggregate table: `results/results.md`.

## Definition of Done

- [x] tenant/role and explicit metadata predicates implemented
- [x] unfiltered/post-filter/pre-filter controls implemented with identical BM25 text
- [x] post-filter oversampling control implemented
- [x] tenant/ACL, region, product, time, and empty-result benchmark defined
- [x] leakage/filter/retrieval metrics separated
- [x] candidate-cost metrics added
- [x] regression tests added
- [x] full repository CI + metadata/filter evaluator passes
- [x] persisted JSON/Markdown results reviewed
- [x] representative leakage/recall/cost failures written down
- [ ] ROADMAP marks metadata/filter-aware sub-lab DONE only after the final gate passes

Metadata / Filter-aware RAG is not merged until the final unchecked gate passes.
