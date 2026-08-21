# M08 Web RAG results

Frozen web snapshot as of `2026-08-20`: 14 pages, 8 queries.

| System | Hit@1 | Recall@3 | MRR | Stale top1 | Low-authority top1 | Duplicate@3 | Answer contains ref | Grounded | E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| body_bm25 | 0.500 | 0.875 | 0.688 | 0.400 | 0.375 | 0.167 | 0.500 | 1.000 | 0.068 |
| metadata_bm25 | 0.375 | 1.000 | 0.667 | 0.800 | 0.625 | 0.167 | 0.375 | 1.000 | 0.061 |
| web_policy | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.097 |

## Interpretation guardrails

- Retrieval/source metrics are evaluated independently from answer text.
- The answerer returns the top page verbatim; groundedness therefore cannot hide stale-source errors.
- Authority scores are controlled benchmark metadata, not a claim that production trust can be reduced to one scalar.
- The snapshot is deterministic and intentionally does not claim live-search coverage or production latency.
