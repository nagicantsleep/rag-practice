# M11.1 baseline results

Frozen benchmark split: `test`. Benchmark clock: `2026-08-15T12:00:00Z`.

| System | Task success | Field acc | Evidence recall | Evidence precision | Source recall | Unauthorized exposure | Stale exposure | Untrusted exposure | Mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_retrieval | 0.000 | 0.046 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
| document_only | 0.000 | 0.167 | 0.156 | 0.148 | 0.244 | 0.000 | 1.000 | 1.000 | 0.242 |
| structured_only | 0.333 | 0.537 | 0.626 | 0.619 | 0.680 | 0.000 | 0.000 | 0.000 | 0.200 |
| fixed_mixed | 0.500 | 0.856 | 0.782 | 0.392 | 0.924 | 0.000 | 1.000 | 1.000 | 0.257 |

Task success is strict: all expected answer fields must match, all required evidence must be present, and forbidden evidence must not be exposed.
Timings are implementation sanity measurements, not production throughput claims.
