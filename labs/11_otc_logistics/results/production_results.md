# M11.3 production-serving results

Frozen benchmark clock: `2026-08-15T12:00:00Z`.

| Metric | Value |
| --- | ---: |
| answer field accuracy | 1.000 |
| evidence recall | 1.000 |
| cache expectation accuracy | 1.000 |
| cache hit rate | 0.250 |
| role isolation | 1.000 |
| generation invalidation | 1.000 |
| mutation correctness | 1.000 |
| unauthorized exposure | 0.000 |
| stale exposure | 0.000 |
| untrusted exposure | 0.000 |
| observability completeness | 1.000 |
| p50 serving ms | 0.201 |
| p95 serving ms | 0.238 |
| mean actions | 1.750 |
| mean synthetic tool cost | 1.387 |

## Scale sanity

| Extra records | Stable | Build ms | Cold ms | Warm ms | Upsert ms | Delete ms | Cache entries | Logical records |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1 | 0.200 | 0.190 | 0.027 | 0.003 | 0.001 | 1 | 215 |
| 1000 | 1 | 1.954 | 0.197 | 0.026 | 0.004 | 0.001 | 1 | 1115 |

Timings and synthetic tool cost are educational implementation sanity measurements, not provider billing, database throughput, ANN performance, or concurrency claims.
