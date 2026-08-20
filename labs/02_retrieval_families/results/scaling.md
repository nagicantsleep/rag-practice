# M02 Scaling Stress Test

Experiment: `m02_scaling_stress_v1`

This experiment keeps the same held-out M02 target questions and grows the candidate corpus with deterministic, deliberately off-topic distractors. It measures candidate-set robustness, build/search latency, and representation growth; it does **not** claim broader language-domain coverage.

Hybrid BM25 weight selected on the separate dev set: **0.0** (pretrained dense: **1.0**).

## Quality and latency by corpus size

| Docs | Method | Recall@1 | Recall@3 | MRR | Mean query ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10 | bm25 | 0.800 | 0.900 | 0.850 | 0.05 |
| 10 | hashing | 0.800 | 0.800 | 0.835 | 0.42 |
| 10 | pretrained | 0.900 | 0.900 | 0.925 | 9.79 |
| 10 | hybrid | 0.900 | 0.900 | 0.925 | 9.88 |
| 100 | bm25 | 0.800 | 0.900 | 0.850 | 0.07 |
| 100 | hashing | 0.700 | 0.800 | 0.733 | 3.32 |
| 100 | pretrained | 0.900 | 0.900 | 0.925 | 9.94 |
| 100 | hybrid | 0.900 | 0.900 | 0.925 | 10.11 |
| 1000 | bm25 | 0.800 | 0.900 | 0.850 | 0.22 |
| 1000 | hashing | 0.700 | 0.800 | 0.733 | 32.09 |
| 1000 | pretrained | 0.900 | 0.900 | 0.925 | 11.03 |
| 1000 | hybrid | 0.900 | 0.900 | 0.925 | 11.36 |

## Representation growth

| Docs | BM25 postings | Hashing float32 payload | MiniLM float32 payload |
| ---: | ---: | ---: | ---: |
| 10 | 150 | 10240 B | 15360 B |
| 100 | 1636 | 102400 B | 153600 B |
| 1000 | 16584 | 1024000 B | 1536000 B |

## Interpretation

The benchmark deliberately separates quality stability from systems cost. Exact quality can remain flat while exhaustive vector scans become linearly more expensive. Later production milestones will replace these educational exhaustive scans with ANN/index-serving systems; M02 keeps them visible so the cost model is obvious.
