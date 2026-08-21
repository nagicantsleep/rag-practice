# M08.3 — Metadata / Filter-aware RAG

Status: **IN PROGRESS** — implementation/evaluation candidate pending CI evidence.

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

## Definition of Done

- [x] tenant/role and explicit metadata predicates implemented
- [x] unfiltered/post-filter/pre-filter controls implemented with identical BM25 text
- [x] post-filter oversampling control implemented
- [x] tenant/ACL, region, product, time, and empty-result benchmark defined
- [x] leakage/filter/retrieval metrics separated
- [x] candidate-cost metrics added
- [x] regression tests added
- [ ] full repository CI + metadata/filter evaluator passes
- [ ] persisted JSON/Markdown results reviewed
- [ ] representative leakage/recall/cost failures written down
- [ ] ROADMAP marks metadata/filter-aware sub-lab DONE only after evidence passes

This is a controlled authorization/filter-placement lab, not a production IAM system. Real deployments should enforce access at the storage/index layer with authoritative identity/policy infrastructure rather than trusting application-side ranking code alone.
