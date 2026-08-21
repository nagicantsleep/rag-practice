# M10 — Training and Production RAG

Status: **IN PROGRESS — M10.1 TRAINING EVIDENCE RECORDED**

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

With M10.1 evidence now recorded, the next phase freezes deterministic serving scenarios before production implementation. Production metrics remain independent from offline retrieval metrics: a fast cache hit that serves stale or unauthorized evidence is a failure.

### M10.2 Definition of Done

- [ ] Freeze production scenarios before implementation tuning.
- [ ] Implement cache, incremental-index, observability, ACL, freshness, and adversarial controls.
- [ ] Measure quality, latency, invalidation correctness, stale exposure, unauthorized exposure, and regression behavior separately.
- [ ] Add deterministic scale sanity checks.
- [ ] Persist machine-readable and human-readable production results.

## Completion guardrail

M10 is not `DONE` until both training and production evidence are recorded, representative failures are retained, a final source-of-truth gate passes, and `ROADMAP.md` is updated only after that gate.
