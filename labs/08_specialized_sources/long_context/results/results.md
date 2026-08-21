# M08.7 Long-context vs retrieval routing results

Frozen benchmark: 4 context bundles / 12 queries. Direct reading and retrieval share the same evidence bundles; only context selection changes.

| System | Route acc | Evidence recall | Evidence complete | Answer acc | Grounded | Abstention | Context words | Context fraction | Retrieval calls | Unnecessary retrieval | Unnecessary full context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_direct | 0.583 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 490.5 | 1.000 | 0.00 | 0.000 | 1.000 |
| always_retrieve | 0.417 | 0.858 | 0.700 | 0.750 | 1.000 | 1.000 | 100.2 | 0.322 | 1.00 | 1.000 | 0.000 |
| explicit_router | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 275.5 | 0.667 | 0.42 | 0.000 | 0.000 |

## Interpretation guardrails

- `always_direct` is a full-context mechanism ceiling, not evidence that every transformer should receive all available context.
- `always_retrieve` uses the frozen BM25 top-2 budget. Global questions can require more evidence sections than the retrieval window can hold.
- `explicit_router` sees only question text and bundle word count; qrels, expected answers, answerability, and pretrained outputs are unavailable at runtime.
- Context footprint and retrieval calls are part of the policy objective; answer correctness alone does not decide the preferred route.
- The benchmark is tiny and synthetic. Perfect controlled routing demonstrates the declared mechanism boundary only.
