# M10 — Training and Production RAG

Status: **IN PROGRESS — TRAINING + PRODUCTION EVIDENCE COMPLETE / FINAL GATE PENDING**

M10 intentionally separates model-training evidence from production-system evidence. Offline retrieval gains do not imply serving readiness, and serving mechanics do not imply retrieval quality.

## M10.1 — Retriever training and hard negatives

The benchmark is frozen at `benchmarks/m10_training/` before fine-tuned result inspection. Freeze commit: `2c8c060c17420d8ec82ad12b916601416a5fc532`.

Systems:

1. pinned pretrained MiniLM baseline;
2. pair-only fine-tune from the same pinned checkpoint;
3. hard-negative fine-tune from the same checkpoint, optimizer, seed, epochs, and batch size, adding exactly one baseline-mined TRAIN negative per query.

### Retriever evidence

The first valid fine-tuned run was PR gate `32508731504` / job `96854692711`: **141 repository tests passed** and the pinned M10.1 evaluator passed. The prior run `32508502384` is diagnostic only; it failed before producing fine-tuned metrics because Sentence Transformers 5.6.1 returned non-tensor tokenization metadata. Commit `b0467e0f37cb6b04458365469266ab58caa4934f` repaired only that compatibility path and did not change the frozen benchmark or training contract.

| System | Test Recall@1 | Test Recall@3 | Test MRR | Mean score margin | Training ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| pinned pretrained baseline | **1.000** | **1.000** | **1.000** | 0.1334 | — |
| pair-only fine-tune | **1.000** | **1.000** | **1.000** | **0.1686** | ~943 |
| explicit hard-negative fine-tune | **1.000** | **1.000** | **1.000** | 0.1588 | ~1,278 |

### Retriever findings

- **The frozen held-out test is rank-saturated by the pretrained baseline.** Recall@1/3 and MRR are already `1.0`, so this benchmark cannot demonstrate a rank-quality gain from fine-tuning. That limitation is retained instead of changing the test set after inspection.
- **Representation geometry still changes when rank metrics saturate.** Pair-only fine-tuning increases mean relevant-minus-best-negative margin from `0.1334` to `0.1686` (`+0.0352`) while preserving rank-perfect test behavior.
- **Hard negatives are not automatically better.** Every mined TRAIN negative is baseline rank 2, yet the explicit-negative variant reaches margin `0.1588`, below pair-only `0.1686`, and takes more CPU training time under the frozen contract.
- **Dev exposed a real confuser before any training selection.** The untouched baseline has dev Recall@1 `0.875`; the status-alias class is the weak class (`0.5` Recall@1), including Eon North where its deployment manual outranks the service-status card. Dev remains diagnostic and is not used to select a new test-facing configuration.
- **Rank and margin answer different questions.** A larger cosine margin is useful evidence that training changed separation, but it is not counted as a retrieval win when held-out ranking is unchanged.

Persisted evidence: `labs/10_training_production/results/training_results.json` and `training_results.md`. All eight hard negatives are recorded with baseline rank/score; no dev/test document participates in mining.

## M10.1b — Learned reranker

The transparent reranker contract was frozen at commit `888b20306a3ac416e78c7b5cc3d5465604c648f5` before its first result. It reorders only MiniLM baseline top-3 candidates using four fixed features — dense cosine, BM25 score, query-token overlap fraction, and reciprocal first-stage rank — with a five-parameter linear scorer trained only on TRAIN candidate pairs.

PR gate `32509314025` / job `96856483683` passed **144 repository tests**, the unchanged retriever evaluator, and the learned-reranker evaluator.

| Split | Candidate Recall@3 | Rerank Recall@1 | MRR | Mean learned margin | Mean rerank ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | **1.000** | 0.875 | 0.9375 | 3.9366 | ~0.054 |
| test | **1.000** | **1.000** | **1.000** | 5.5844 | ~0.051 |

### Learned-reranker findings

- **Candidate recall remains the ceiling.** The relevant document is present in top-3 for every dev/test query; the learned scorer is never credited for evidence the first stage did not retrieve.
- **The learned reranker does not improve held-out rank quality here.** Dev Recall@1/MRR remain `0.875/0.9375`, and test remains `1.0/1.0` because the first-stage baseline was already rank-perfect on test.
- **A larger learned-score margin is not a retrieval win by itself.** The linear scorer separates candidates strongly, but rank metrics do not change; the margin lives in an arbitrary learned-score scale and is reported only as model behavior.
- **The tiny learned scorer is cheap but not magically useful.** It adds only five float32 parameters and roughly `0.05 ms/query` rerank computation, yet adds no quality on this frozen test.
- **This reranker is post-hoc on the same benchmark.** Its contract was created only after retriever evidence was recorded, so it is a mechanism/training control rather than fresh held-out generalization evidence.

Persisted evidence: `labs/10_training_production/results/reranker_results.json` and `reranker_results.md`.

### M10.1 Definition of Done

- [x] Freeze train/dev/test entities, qrels, terminology mismatches, model revision, mining rule, and optimizer hyperparameters before fine-tuned inspection.
- [x] Implement baseline, pair-only, and explicit hard-negative training controls.
- [x] Add rank/margin/class-level evaluation and unit tests.
- [x] Pass full repository regression + pinned M10.1 retriever training evaluator.
- [x] Persist results and inspect representative errors before changing scope.
- [x] Freeze and evaluate an explicit learned-reranker training control without changing the retrieval split.

## M10.2 — Production contracts

The deterministic serving workload was frozen before production implementation at commit `0495168691a0bc5f70f275a78978bcdc57879f90`. The same ordered operations compare an intentionally unsafe baseline against a guarded serving path with generation-aware/role-aware caching, incremental indexing, ACL → freshness → trust filtering, lexical evidence checks, and complete per-query traces.

The first implementation gate `32509852290` / job `96858161124` is diagnostic only: **146 tests passed, 2 failed** before any production evaluator result was accepted. After ACL filtering, guarded lexical ranking could still return a public document matching only the generic word `code` for a Finance query. Commit `f5385d8829fc085d3ac1ce5323b3f7059700aecf` repaired only guarded lexical evidence semantics by requiring all distinct query terms before ranking; the frozen workload, expected outcomes, clock, cache policy, filter order, unsafe baseline, and scale contract did not change.

The valid production gate `32510837455` / job `96861322550` passed **148 repository tests**, the unchanged M10.1 retriever evaluator, the frozen learned-reranker evaluator, and the frozen M10.2 production evaluator.

| System | Scenario accuracy | Cache expectation | Invalidation | No-evidence | Unauthorized exposure | Stale exposure | Untrusted exposure | Observability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unsafe baseline | 0.455 | 0.727 | 0.000 | 0.000 | 0.091 | 0.091 | 0.091 | 1.000 |
| guarded | **1.000** | **1.000** | **1.000** | **1.000** | **0.000** | **0.000** | **0.000** | **1.000** |

Scale sanity persisted by the push evaluator keeps target Hit@1 at `1` for both 100 and 1000 documents. On that GitHub Actions CPU run, build time was about `0.400 ms` and `4.084 ms`, query time about `0.101 ms` and `0.923 ms`, with `797` and `7996` posting entries respectively. These are implementation sanity measurements, not production throughput or ANN claims.

### Production findings

- **A fast cache is unsafe without identity and version boundaries.** The query-only unsafe cache reuses results across role and mutation boundaries, producing cache-invalidation accuracy `0.0` despite a higher cache-hit rate.
- **Policy filtering must happen before evidence exposure.** The guarded path records zero unauthorized, stale, and untrusted exposure on the frozen scenarios; the unsafe baseline exposes each category on one of eleven query operations (`0.0909`).
- **No-evidence behavior is a production correctness feature.** The unsafe lexical fallback never abstains correctly on the frozen negative cases (`0.0` no-evidence accuracy); guarded full-term evidence semantics plus policy filters reach `1.0`.
- **Incremental mutations and cache correctness are coupled.** Guarded cache keys include index generation, so upsert/delete increments invalidate stale snapshots without flushing unrelated historical keys manually.
- **Observability is necessary but not sufficient.** Both systems can emit complete traces (`1.0` observability completeness); the unsafe system remains wrong. Traces make policy/cache failures diagnosable but do not create correctness by themselves.
- **The scale check is deliberately modest.** Linear lexical indexing/query behavior at 100/1000 documents validates the implementation path only; it is not evidence for large-scale vector serving capacity.

Persisted evidence: `labs/10_training_production/results/production_results.json` and `production_results.md`.

### M10.2 Definition of Done

- [x] Freeze production scenarios before implementation tuning.
- [x] Implement cache, incremental-index, observability, ACL, freshness, and adversarial controls.
- [x] Measure quality, latency, invalidation correctness, stale exposure, unauthorized exposure, and regression behavior separately.
- [x] Add deterministic scale sanity checks.
- [x] Persist machine-readable and human-readable production results.

## Completion guardrail

Training and production evidence are complete. M10 becomes `DONE` only after the final source-of-truth gate passes; `ROADMAP.md` is updated only after that gate.
