# M10.1 retrieval training results

Pinned baseline: `sentence-transformers/all-MiniLM-L6-v2@1c82ace116a2629de82404c4be48c0e5d4cf08be`

The benchmark, split, model revision, mining policy, and optimization hyperparameters were frozen before fine-tuned results were inspected.

## Held-out test summary

| System | Recall@1 | Recall@3 | MRR | Mean relevant-minus-best-negative margin | Training ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pinned pretrained baseline | 1.000 | 1.000 | 1.000 | 0.1334 | 0.0 |
| Pair-only fine-tune | 1.000 | 1.000 | 1.000 | 0.1686 | 745.7 |
| Hard-negative fine-tune | 1.000 | 1.000 | 1.000 | 0.1588 | 957.8 |

## Hard negatives

Hard negatives are mined only from TRAIN documents with the untouched pinned baseline. Dev/test documents are never candidates for mining.

| Query | Positive | Positive rank | Mined negative | Negative rank |
| --- | --- | ---: | --- | ---: |
| tq1 | tr1 | 1 | tr5 | 2 |
| tq2 | tr2 | 1 | tr1 | 2 |
| tq3 | tr3 | 1 | tr4 | 2 |
| tq4 | tr4 | 1 | tr8 | 2 |
| tq5 | tr5 | 1 | tr6 | 2 |
| tq6 | tr6 | 1 | tr5 | 2 |
| tq7 | tr7 | 1 | tr8 | 2 |
| tq8 | tr8 | 1 | tr4 | 2 |

## Interpretation guardrail

This is a tiny synthetic domain-adaptation control. A positive delta is not a general fine-tuning claim, and a zero/negative delta is retained rather than tuned away. Dev metrics are diagnostic only; no post-test hyperparameter selection is allowed.
