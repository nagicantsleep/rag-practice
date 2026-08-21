# M10 — Training and Production RAG

Status: **IN PROGRESS — M10.1 RETRIEVER EVIDENCE RECORDED**

M10 intentionally separates model-training evidence from production-system evidence. Offline retrieval gains do not imply serving readiness, and serving mechanics do not imply retrieval quality.

## M10.1 — Retriever training and hard negatives

The benchmark is frozen at `benchmarks/m10_training/` before fine-tuned result inspection. Freeze commit: `2c8c060c17420d8ec82ad12b916601416a5fc532`.

Systems:

1. pinned pretrained MiniLM baseline;
2. pair-only fine-tune from the same pinned checkpoint;
3. hard-negative fine-tune from the same checkpoint, optimizer, seed, epochs, and batch size, adding exactly one baseline-mined TRAIN negative per query.

Evaluation keeps held-out Recall@1/3, MRR, score margin, class-level errors, mined-negative identity/rank, training loss, model/index footprint, and CPU timing separate. Dev is diagnostic only. A negative or zero fine-tuning delta is retained rather than tuned away.

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
- **This is a tiny synthetic domain-adaptation control.** The result does not imply that two epochs, pair-only training, or these hard negatives generalize to production corpora.

Persisted evidence: `labs/10_training_production/results/training_results.json` and `training_results.md`. All eight hard negatives are recorded with baseline rank/score; no dev/test document participates in mining.

### M10.1 Definition of Done

- [x] Freeze train/dev/test entities, qrels, terminology mismatches, model revision, mining rule, and optimizer hyperparameters before fine-tuned inspection.
- [x] Implement baseline, pair-only, and explicit hard-negative training controls.
- [x] Add rank/margin/class-level evaluation and unit tests.
- [x] Pass full repository regression + pinned M10.1 retriever training evaluator.
- [x] Persist results and inspect representative errors before changing scope.
- [ ] Freeze and evaluate an explicit learned-reranker training control without changing the retrieval split.

## M10.1b — Learned reranker

This is a post-retriever-evidence training control on the **unchanged** M10.1 split, not fresh held-out benchmark design. Its architecture/hyperparameters must be frozen before its first result is inspected. Candidate recall is measured before reranking so a reranker cannot receive credit for a missing positive.

## M10.2 — Production contracts

After M10.1 evidence is recorded, freeze deterministic serving scenarios covering:

- query/result caching and version-aware invalidation;
- incremental upsert/delete indexing;
- traceable latency/cache/index-generation observability;
- ACL filtering before evidence exposure;
- source freshness/staleness policy;
- adversarial/prompt-injection retrieval exposure;
- scaling and regression checks.

Production metrics will be reported independently from offline retrieval quality. A cache hit that returns stale or unauthorized evidence is a failure even if latency improves.

### M10.2 Definition of Done

- [ ] Freeze production scenarios before implementation tuning.
- [ ] Implement cache, incremental-index, observability, ACL, freshness, and adversarial controls.
- [ ] Measure quality, latency, invalidation correctness, stale exposure, unauthorized exposure, and regression behavior separately.
- [ ] Add deterministic scale sanity checks.
- [ ] Persist machine-readable and human-readable production results.

## Completion guardrail

M10 is not `DONE` until both training and production evidence are recorded, representative failures are retained, a final source-of-truth gate passes, and `ROADMAP.md` is updated only after that gate.
