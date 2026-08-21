# M10 — Training and Production RAG

Status: **IN PROGRESS — M10.1 FROZEN / TRAINING GATE PENDING**

M10 intentionally separates model-training evidence from production-system evidence. Offline retrieval gains do not imply serving readiness, and serving mechanics do not imply retrieval quality.

## M10.1 — Retriever training and hard negatives

The benchmark is frozen at `benchmarks/m10_training/` before fine-tuned result inspection.

Systems:

1. pinned pretrained MiniLM baseline;
2. pair-only fine-tune from the same pinned checkpoint;
3. hard-negative fine-tune from the same checkpoint, optimizer, seed, epochs, and batch size, adding exactly one baseline-mined TRAIN negative per query.

Evaluation keeps held-out Recall@1/3, MRR, score margin, class-level errors, mined-negative identity/rank, training loss, model/index footprint, and CPU timing separate. Dev is diagnostic only. A negative or zero fine-tuning delta is retained rather than tuned away.

### M10.1 Definition of Done

- [x] Freeze train/dev/test entities, qrels, terminology mismatches, model revision, mining rule, and optimizer hyperparameters before fine-tuned inspection.
- [x] Implement baseline, pair-only, and explicit hard-negative training controls.
- [x] Add rank/margin/class-level evaluation and unit tests.
- [ ] Pass full repository regression + pinned M10.1 training evaluator.
- [ ] Persist results and inspect representative test errors before changing scope.
- [ ] Add an explicit learned-reranker training control without changing the frozen retrieval test split.

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
