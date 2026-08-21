# M08.3 Metadata / Filter-aware RAG results

Benchmark: 19 records, 9 queries.

| System | Recall@3 | Hit@1 | Constraint satisfied | Security leakage | Filter violation | Empty correct | Answer correct | Indexed records | Examined candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unfiltered_bm25 | 1.000 | 0.250 | 0.333 | 0.444 | 0.667 | 0.000 | 0.222 | 19.0 | 3.0 |
| postfilter_k2 | 0.375 | 0.375 | 1.000 | 0.000 | 0.000 | 1.000 | 0.444 | 19.0 | 2.0 |
| postfilter_oversample_k5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 19.0 | 5.0 |
| prefilter_bm25 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 2.7 | 1.4 |

## Interpretation guardrails

- Unfiltered Recall can look excellent while returning unauthorized or filter-invalid records.
- Post-filtering removes leaks from returned results, but a small pre-filter candidate budget can destroy recall.
- Oversampling can recover post-filter recall at higher candidate cost; it does not make hard authorization a relevance problem.
- Pre-filtering applies hard predicates before lexical ranking and is the security-oriented candidate path.
- This is a tiny deterministic corpus; latency/candidate counts illustrate mechanisms rather than production serving performance.
