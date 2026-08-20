# M06 Adaptive/Active/Corrective Generation — Phase 2

Generator: `google/flan-t5-small` @ `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`.

| System | Answer F1 | Contains ref | Grounded | Evidence complete | Unsupported answer | Answerable refusal | Unanswerable refusal recall | Mean retrieval calls | Active calls | Attempts | E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_single_rag | 0.517 | 0.600 | 0.875 | 0.625 | 1.000 | 0.000 | 0.000 | 1.33 | 0.00 | 1.00 | 93.90 |
| adaptive_control | 0.717 | 0.800 | 0.875 | 1.000 | 1.000 | 0.000 | 0.000 | 1.25 | 0.00 | 1.00 | 100.43 |
| adaptive_active_reflect | 0.717 | 0.800 | 0.875 | 1.000 | 0.500 | 0.100 | 0.500 | 1.42 | 0.17 | 1.17 | 122.63 |

Correctness is evaluated only on answerable questions. Grounded-token recall excludes no-retrieval questions. Unsupported-answer rate is measured only on the two deliberately unanswerable held-out questions.

The active/reflection system may legitimately be a negative result if token confidence is poorly calibrated or lexical reflection over-refuses; those failures are retained for error analysis rather than tuned on this held-out set.
