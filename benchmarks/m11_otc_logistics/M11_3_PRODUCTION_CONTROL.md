# M11.3 production-serving control contract

Status: **FROZEN BEFORE M11.3 IMPLEMENTATION**

This contract is frozen after M11.2 evidence is recorded and before implementing M11.3 serving controls. It does not modify the frozen M11.0 benchmark or M11.2 investigation policy.

## Hypothesis

A generation-aware serving layer around the integrated copilot can preserve M11.2 answer/evidence/security behavior while adding safe cache reuse, incremental source mutation, complete request traces, and deterministic latency/scale sanity measurements.

## Runtime boundary

The serving runtime may use only normal M11 runtime sources and the M11.2 integrated copilot. It must never inspect `benchmark.json`, qrels, expected answers, task classes, forbidden-evidence labels, or evaluator-only cache expectations.

## Serving state

The serving layer has an explicit integer `generation`, initially `0`.

A serving request is identified by:

- question;
- user ID and current sorted roles;
- source snapshot ID;
- serving generation;
- policy version `m11.2-v1`.

The guarded cache key contains **all five** of those dimensions. Cache entries store the complete immutable integrated result payload and must never be shared across role/snapshot/generation boundaries.

## Frozen request sequence

Use these production-control requests in order. They reuse normal business questions but the cache expectations below are evaluator-only.

1. `p1`: U-OPS, g0, Helios current-state query — cold cache.
2. `p2`: repeat p1 — warm cache.
3. `p3`: U-OPS, g0, Cedar sensitive finance query — must be denied and must not cache/expose finance payload.
4. `p4`: U-FIN, g0, same Cedar finance query — separate role cache key; authorized finance result.
5. `p5`: U-OPS, g0, Epsilon current-contract query — selects v2 and rejects v1.
6. `p6`: U-OPS, g0, Gamma address-exception query — trusted SOP, untrusted note rejected.
7. `p7`: apply the already-frozen g0 → g1 Helios mutation incrementally; generation increments exactly once.
8. `p8`: U-OPS, g1, Helios exception/escalation query — cache miss after mutation and returns VEHICLE_BREAKDOWN evidence.
9. `p9`: repeat p8 — warm cache.

No serving implementation may receive the request IDs or expected cache labels.

## Incremental mutation contract

The M11.3 serving layer must apply the already-frozen M11.0 g0 → g1 mutation without rebuilding the complete corpus representation:

- append `EV-H003`;
- replace the current Helios shipment representation with `SH-1008@g1`;
- advance active snapshot from `g0` to `g1`;
- increment serving generation once;
- make all prior-generation cache entries unreachable through the cache key.

The evaluator records mutation latency and verifies that the post-mutation answer differs from pre-mutation state where the frozen source data requires it.

## Observability contract

Every request trace must contain:

- request sequence number;
- user ID and sorted roles;
- snapshot ID and serving generation;
- cache hit;
- cache-key dimensions (never sensitive record payloads);
- integrated action sequence;
- evidence IDs and source families;
- rejected unauthorized/stale/untrusted IDs;
- stop reason;
- action count;
- integrated latency;
- serving latency;
- synthetic tool cost.

Synthetic tool cost is fixed at `1.0` per source action and `0.05` per cache hit. It is an educational accounting metric, not currency or provider billing.

A denied finance request may expose only authorization decision/provenance in its trace, never the forbidden finance record.

## Correctness metrics

Persist independently:

- answer field accuracy against the unchanged frozen benchmark tasks used by the serving sequence;
- evidence recall and forbidden exposure;
- cache expectation accuracy;
- role-isolation correctness;
- generation-invalidation correctness;
- mutation correctness;
- unauthorized/stale/untrusted exposure rates;
- observability completeness;
- mean/p50/p95 serving latency;
- mean action count and synthetic tool cost;
- cache hit rate.

The production evaluator may use benchmark labels after the runtime returns, but must never pass those labels into the serving system.

## Deterministic scale sanity

Run the same guarded serving implementation over deterministic irrelevant-record expansion sizes `100` and `1000`, seed `53`.

The expansion adds synthetic irrelevant customer/order/shipment records to a separate scale-only runtime copy. It must not alter the frozen M11 benchmark records or evaluation labels.

For each size record:

- build/load latency;
- target cold-query latency;
- target warm-query latency;
- one incremental upsert latency;
- one delete latency;
- cache-entry count;
- indexed/logical record count;
- target answer stability.

The target is the Juno `SO-1010` current status/ETA request. It must remain semantically unchanged. These measurements are Python/GitHub-Actions implementation sanity checks, not ANN, database, concurrency, or production-throughput claims.

## Release thresholds

M11.3 is acceptable only if:

- full repository regression passes;
- M11.1 and M11.2 evaluators remain unchanged and pass;
- serving-sequence answer/evidence behavior matches the corresponding repaired integrated results;
- cache expectation accuracy is `1.0`;
- role isolation is `1.0`;
- generation invalidation is `1.0`;
- mutation correctness is `1.0`;
- unauthorized/stale/untrusted exposure rates are all `0.0`;
- observability completeness is `1.0`;
- target answer stability is `1.0` at both scale sizes.

Latency has no pass/fail numeric target; it is reported as a sanity measurement only.

## Integrity rule

After this freeze commit, do not change this request sequence, cache-key dimensions, mutation policy, trace fields, cost accounting, scale sizes/seed/target, or release thresholds in response to M11.3 outputs. Objective implementation defects may be repaired only without changing these semantics and must be documented.
