# M11.3 — Production serving findings

Status: **EVIDENCE RECORDED / FINAL SOURCE-OF-TRUTH GATE NEXT**

The M11.3 serving contract was frozen in `benchmarks/m11_otc_logistics/M11_3_PRODUCTION_CONTROL.md` at commit `075919a2fbd816f93ecb2139eab477fa6bd25c95` before production-serving implementation.

PR gate `32558661583` / job `96996711770` passed **164 repository tests**, the unchanged M11.1 baseline evaluator, the unchanged repaired M11.2 integrated evaluator, and the M11.3 production evaluator.

## Serving evidence

| Metric | Result |
| --- | ---: |
| answer field accuracy on frozen serving sequence | **1.000** |
| evidence recall | **1.000** |
| cache expectation accuracy | **1.000** |
| role isolation | **1.000** |
| generation invalidation | **1.000** |
| mutation correctness | **1.000** |
| unauthorized exposure | **0.000** |
| stale exposure | **0.000** |
| untrusted exposure | **0.000** |
| observability completeness | **1.000** |
| cache hit rate | 0.250 |
| mean source actions | 1.750 |
| mean synthetic tool cost | 1.387 |

The production sequence verifies two important isolation boundaries directly: the denied U-OPS finance request is not cached and never exposes `FIN-1003`, while the identical U-FIN request uses a distinct role-aware key and returns the authorized finance evidence; after the frozen Helios g0→g1 mutation, generation increments once and the g1 request cannot reuse a prior-generation cache entry.

## Scale sanity

The deterministic scale-only expansion keeps the real Juno target request unchanged while adding irrelevant serving-state records.

| Extra records | Target stable | Build ms | Cold ms | Warm ms | Upsert ms | Delete ms | Logical records |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | **1** | ~0.26–0.30 | ~0.25 | ~0.03 | ~0.004 | ~0.001 | 215 |
| 1000 | **1** | ~2.49–2.52 | ~0.24–0.25 | ~0.03–0.04 | ~0.004–0.006 | ~0.001 | 1115 |

These are Python/GitHub-Actions implementation sanity measurements only. They are not database, ANN, concurrency, provider-cost, or production-throughput claims.

## Findings

- **Cache correctness is part of answer correctness.** A fast hit is useful only when user roles, source snapshot, serving generation, and policy version are part of the identity of the cached result.
- **Denied sensitive results are safer uncached in this control.** Repeating the U-OPS finance request remains a fresh fail-closed authorization decision, while the authorized U-FIN request cannot collide with it.
- **Generation-aware invalidation makes source mutation explicit.** Old cache entries can remain physically present but become unreachable because the serving generation is part of the key; this avoids relying on best-effort cache deletion for correctness.
- **Serving observability must preserve policy decisions, not sensitive payloads.** Traces expose actions, evidence IDs, rejection IDs, stop reason, latency, and cost while the denied finance trace never contains the forbidden finance record.
- **M11.2 quality limitations remain visible.** M11.3 does not change the frozen integrated benchmark or erase the retained Atlas/Foxtrot evidence-policy mismatches; production serving demonstrates preservation and system controls, not a new offline-quality claim.
- **Scale sanity is deliberately modest.** The expansion tests state/cache mechanics and answer stability only; it does not claim a production database or vector-index benchmark.

Persisted evidence: `labs/11_otc_logistics/results/production_results.json` and `production_results.md`.
