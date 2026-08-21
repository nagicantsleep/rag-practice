# M08.2 SQL / Structured RAG results

Benchmark: 4 tables, 20 rows, 9 queries.

| System | Evidence recall | Evidence complete | Answer exact | Execution success | Unsafe reject | Empty correct | Unsupported handled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat row BM25@5 | 0.500 | 0.500 | n/a | n/a | n/a | n/a | n/a |
| schema-aware validated SQL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Interpretation guardrails

- Flat BM25 is a retrieval-only control; it cannot compute joins or aggregates.
- SQL answer correctness is evaluated separately from row-level evidence completeness.
- Unsafe mutation and unsupported-schema requests are expected to fail closed.
- SQLite latency is only a deterministic sanity measurement, not a production database claim.
