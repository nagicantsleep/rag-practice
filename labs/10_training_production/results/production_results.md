# M10.2 production-serving results

| System | Scenario accuracy | Cache expectation | Invalidation | No-evidence | Unauthorized exposure | Stale exposure | Untrusted exposure | Observability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unsafe baseline | 0.455 | 0.727 | 0.000 | 0.000 | 0.091 | 0.091 | 0.091 | 1.000 |
| guarded | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |

## Scale sanity

| Documents | Hit@1 | Build ms | Query ms | Upsert ms | Delete ms | Posting entries |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1 | 0.325 | 0.078 | 0.003 | 0.002 | 797 |
| 1000 | 1 | 3.111 | 0.702 | 0.004 | 0.003 | 7996 |

Timings are GitHub Actions CPU implementation sanity measurements, not production throughput claims.
