# M10.2 frozen production-serving benchmark

Status: **FROZEN BEFORE PRODUCTION IMPLEMENTATION**

This workload is frozen after M10.1 training/reranker evidence and before implementing the M10.2 serving controls.

## Hypothesis

Production RAG controls should prevent stale, unauthorized, or explicitly untrusted evidence from being served while preserving correct fresh/trusted retrieval. Cache and incremental-index speed are useful only when invalidation and policy isolation remain correct.

## Shared scripted workload

`scenarios.json` defines one fixed clock, five initial documents, and fourteen ordered operations. The same operations are evaluated against:

1. **unsafe baseline** — lexical index, query-only cache, no ACL/freshness/trust filtering, no cache invalidation after mutation;
2. **guarded serving path** — incremental lexical index, cache key includes roles + index generation + policy parameters, and filtering occurs ACL → freshness → trust before lexical ranking.

The workload covers cold/warm cache, role denial/allow, cache isolation across roles, upsert and delete invalidation, new-document insertion, stale evidence, an explicitly untrusted prompt-injection document, trusted retrieval, and abstention after filtering/deletion.

## Frozen correctness contract

- Clock: `2026-08-01T00:00:00+00:00`.
- Freshness max age: `30 days`.
- Trusted evidence required in guarded mode.
- Top-k: `1`.
- Empty result is the only no-evidence representation.
- Mutations increment index generation in guarded mode.
- Guarded cache key fields are exactly query, sorted roles, index generation, max-age, and trust requirement.
- Unsafe baseline cache is keyed by query only and is intentionally not invalidated on mutation.

Runtime systems must not inspect `expected_ids`, `expected_text_contains`, operation classes, or expected cache-hit labels; those fields are evaluator-only.

## Evaluation contract

Persist separately for unsafe and guarded systems:

- scenario expected-result accuracy;
- expected cache-hit accuracy;
- cache invalidation correctness after upsert/delete;
- mutation correctness;
- unauthorized exposure rate;
- stale exposure rate;
- untrusted/adversarial exposure rate;
- no-evidence accuracy;
- observability completeness;
- mean query/update latency;
- index generation and posting footprint traces.

Every guarded query trace must record cache hit, index generation, candidate count, ACL-filter count, stale-filter count, untrusted-filter count, returned ids, and latency.

## Frozen scale sanity check

Generate deterministic corpora of 100 and 1000 documents with seed `41`, add one exact target document, and measure build/query/upsert/delete latency plus posting-entry footprint. This is an implementation sanity check, not a production throughput claim or ANN benchmark.

## Integrity rule

Do not change scenarios, expected outcomes, clock, max age, cache-key policy, filter order, unsafe baseline behavior, scale sizes/seed, or required metrics after the first production result is inspected. Objective implementation bugs may be fixed without altering this contract.
