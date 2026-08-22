# M12 calibrated RAG results

Thresholds are selected on the frozen calibration split only and then applied unchanged to test-ID and test-OOD.

## test_id

| Method | Accuracy | Brier | ECE | AURC | Threshold | Coverage | Selective risk | False-answer rate | Abstention acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 0.300 | 0.250 | 0.200 | 0.371 | 0.50 | 1.000 | 0.700 | 0.700 | 0.000 |
| top1 | 0.300 | 0.547 | 0.593 | 0.371 | 1.00 | 0.800 | 0.625 | 0.500 | 1.000 |
| margin | 0.300 | 0.178 | 0.067 | 0.371 | 0.30 | 0.700 | 0.571 | 0.400 | 1.000 |
| hand_composed | 0.300 | 0.287 | 0.365 | 0.371 | 0.65 | 0.700 | 0.571 | 0.400 | 1.000 |
| logistic | 0.300 | 0.158 | 0.219 | 0.430 | 0.35 | 0.600 | 0.500 | 0.300 | 1.000 |

## test_ood

| Method | Accuracy | Brier | ECE | AURC | Threshold | Coverage | Selective risk | False-answer rate | Abstention acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 0.500 | 0.250 | 0.000 | 0.183 | 0.50 | 1.000 | 0.500 | 0.500 | 0.000 |
| top1 | 0.500 | 0.411 | 0.463 | 0.260 | 1.00 | 0.750 | 0.500 | 0.375 | 1.000 |
| margin | 0.500 | 0.240 | 0.229 | 0.239 | 0.30 | 0.625 | 0.400 | 0.250 | 1.000 |
| hand_composed | 0.500 | 0.166 | 0.236 | 0.239 | 0.65 | 0.500 | 0.250 | 0.125 | 1.000 |
| logistic | 0.500 | 0.112 | 0.319 | 0.183 | 0.35 | 0.625 | 0.200 | 0.125 | 1.000 |

## Drift: OOD minus ID

| Method | Accuracy Δ | Brier Δ | ECE Δ | AURC Δ | Coverage Δ | Selective risk Δ | Mean confidence Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | +0.200 | +0.000 | -0.200 | -0.189 | +0.000 | -0.200 | +0.000 |
| top1 | +0.200 | -0.136 | -0.131 | -0.111 | -0.050 | -0.125 | -0.056 |
| margin | +0.200 | +0.062 | +0.163 | -0.132 | -0.075 | -0.171 | +0.038 |
| hand_composed | +0.200 | -0.121 | -0.129 | -0.132 | -0.200 | -0.321 | -0.033 |
| logistic | +0.200 | -0.047 | +0.100 | -0.247 | +0.025 | -0.300 | -0.012 |

## Implementation sanity

| Metric | Value |
| --- | ---: |
| mean trace + feature ms/query | 0.3105 |
| max trace + feature ms/query | 0.4021 |
| logistic fit ms | 10.5198 |
| mean logistic predict ms/query | 0.001896 |
| model calls | 0 |

Full reliability bins, discrete risk–coverage curve points, and per-query runtime/evaluator traces are persisted in the JSON artifact.

Calibration quality and selective risk are reported separately. Timings are educational Python/GitHub-Actions implementation sanity measurements, not production throughput claims.
