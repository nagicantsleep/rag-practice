# M06 Adaptive/Corrective Control — Phase 1

| System | Route acc | Evidence recall | Evidence complete | Mean calls | Unnecessary retrieval | Iterative under-route | Correction P | Correction R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_single | 0.583 | 0.812 | 0.625 | 1.33 | 1.000 | 1.000 | 0.500 | 1.000 |
| keyword_router | 1.000 | 1.000 | 1.000 | 1.25 | 0.000 | 0.000 | 1.000 | 1.000 |
| naive_bayes_router | 1.000 | 1.000 | 1.000 | 1.25 | 0.000 | 0.000 | 1.000 | 1.000 |
| oracle_route_ceiling | 1.000 | 1.000 | 1.000 | 1.25 | 0.000 | 0.000 | 1.000 | 1.000 |

Evidence metrics exclude no-retrieval questions. Correction labels evaluate whether the qrel-blind retrieval judge triggers fallback only for the deliberately stale primary-source cases.
