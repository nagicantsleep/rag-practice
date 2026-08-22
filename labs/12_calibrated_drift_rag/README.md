# M12 — Calibrated & Drift-Aware RAG

Status: **DONE**

M12 studies a gap left intentionally open by M00–M11: a RAG system can expose retrieval scores, token probabilities, support checks, or heuristic confidence and still have no principled answer to **when it should answer and when it should abstain**. M12 treats confidence calibration and selective prediction as first-class RAG mechanisms rather than presentation metadata.

## Frozen learning objective

Build and evaluate a RAG decision layer that converts only runtime-observable retrieval/generation/evidence signals into a calibrated probability that the final answer is correct and sufficiently supported, then uses a threshold chosen without test leakage to decide `ANSWER` versus `ABSTAIN`.

The milestone must distinguish four questions:

1. **ranking quality** — did retrieval surface the necessary evidence?;
2. **answer quality** — is the produced answer correct and evidence-supported?;
3. **calibration** — does a confidence value correspond to empirical correctness frequency?;
4. **selective prediction** — as coverage is reduced, does answered-query risk actually fall?

A confidence score is not credited merely for separating obvious examples. Calibration quality and answer/abstention utility must be measured explicitly.

## Frozen hypotheses

### H1 — raw similarity is not calibrated confidence

Maximum retrieval similarity and top-1/top-2 margin will be useful ranking signals but will be overconfident on at least some retrieval-miss, no-evidence, conflict, stale, or out-of-distribution cases.

### H2 — multi-signal calibration improves selective risk

A calibrator fit only on frozen training/calibration data using runtime-visible evidence signals should reduce calibration error and/or area under the risk–coverage curve relative to raw-score confidence without reducing full-coverage underlying answer accuracy.

### H3 — in-distribution calibration can degrade under shift

A threshold/calibrator selected on in-distribution calibration data may become miscalibrated under lexical/entity/style shift. Drift metrics therefore remain separate from in-distribution metrics; M12 must not silently retune on the shifted test slice.

### H4 — abstention has asymmetric utility

Reducing unsupported or wrong answered queries can be worthwhile even when overall coverage falls. M12 must report both risk and coverage instead of optimizing accuracy on answered examples alone.

## Frozen phase boundaries

### M12.0 — Dataset, splits, and evaluation contract

Construct and freeze a deterministic benchmark before implementing or inspecting calibration methods. The benchmark must contain:

- answerable in-distribution cases;
- lexical/paraphrase variation;
- retrieval distractors and near-miss evidence;
- no-evidence cases;
- conflicting-evidence cases;
- stale/invalid evidence cases;
- answerable cases where retrieved evidence is complete but the fixed generator/control answer is wrong or unsupported;
- shifted/OOD cases with entity, wording, or evidence-distribution changes;
- separate train/calibration/test-ID/test-OOD splits or an equivalent leakage-safe partition.

The benchmark must freeze correctness/support labels, required evidence, answerability, shift class, and any source validity metadata before calibrator implementation.

### M12.1 — Uncalibrated baselines

Evaluate simple confidence baselines under identical frozen traces/signals:

1. constant-confidence / always-answer reference;
2. maximum retrieval score;
3. retrieval top-1 minus top-2 margin;
4. a transparent hand-composed evidence-confidence baseline using only frozen runtime-visible signals.

Thresholds for selective answering must be selected on the calibration split only.

### M12.2 — Learned calibration and selective policy

Implement transparent calibrators before introducing any opaque judge:

- Platt/logistic calibration over frozen runtime-visible signals;
- isotonic regression if the calibration split is large enough to support it without degenerate fitting;
- optional conformal-style selective thresholding/control if its finite-sample contract is stated precisely.

No calibrator may consume qrels, expected answers, answerability labels, shift labels, or evaluator-only correctness at inference time.

### M12.3 — Drift gate and production interpretation

Evaluate the frozen calibrator/threshold unchanged on shifted test cases. Record calibration drift, selective-risk drift, false-answer risk, abstention behavior, and operational trade-offs. Any post-shift recalibration must be a separately frozen experiment and cannot overwrite the original test evidence.

## Frozen runtime-visible signal families

The benchmark/system may expose signals such as:

- top retrieval score;
- top-1/top-2 retrieval margin;
- count/fraction of retrieved items passing freshness/trust filters;
- evidence-source agreement or conflict indicator;
- lexical or deterministic answer-support overlap;
- retrieved-evidence completeness proxy that does not use qrels;
- generator confidence when a pinned generator exposes it;
- number of retrieval attempts/actions and empty-result indicator.

Exact signal definitions must be frozen before learned calibration. Evaluator-only labels must never be included in feature construction.

## Frozen evaluation axes

### Base answer/retrieval quality

- full-coverage answer correctness;
- evidence recall/completeness where retrieval is involved;
- grounded/support correctness;
- no-evidence and conflict behavior.

### Calibration

- Brier score;
- expected calibration error (ECE) with bin edges frozen before test inspection;
- reliability table/diagram data;
- negative log loss where probabilities are strictly bounded away from 0/1 by a declared epsilon;
- mean confidence on correct versus incorrect answers.

### Selective prediction

- coverage;
- selective risk = error rate among answered queries;
- risk at frozen target coverages;
- coverage at frozen maximum-risk targets where feasible;
- area under the risk–coverage curve (AURC);
- false-answer rate over all queries;
- abstention accuracy on truly unanswerable cases;
- false-abstention rate on answerable cases.

### Drift

- ID versus OOD answer accuracy;
- ID versus OOD Brier/ECE;
- ID versus OOD AURC;
- confidence shift and error-rate shift;
- frozen-threshold coverage/risk delta.

### Cost/observability

- feature/calibrator latency;
- model calls if any;
- persisted per-query confidence, decision, signals, evidence IDs, answer correctness label, and shift class in evaluator artifacts;
- implementation sanity only, not production throughput claims.

## Frozen anti-leakage rules

1. The final test-ID and test-OOD labels are evaluator-only.
2. Thresholds and calibrator hyperparameters are selected using train/calibration data only.
3. Test cases may be inspected only after the benchmark/split contract and baseline/calibrator definitions are frozen.
4. Once first valid test evidence is observed, benchmark labels/splits and metric definitions are immutable except documented serialization or correctness defects that do not depend on model outcomes.
5. A method that improves ECE but worsens selective risk, or vice versa, must report both; no single aggregate hides the trade-off.
6. OOD results are not used to retune the original calibrator unless a new post-shift experiment is frozen separately.

## Definition of Done

- [x] new learning objective written after M11 completion;
- [x] hypotheses frozen before benchmark construction;
- [x] calibration, selective-risk, drift, and anti-leakage evaluation axes frozen;
- [x] construct and freeze M12 benchmark instances and split assignments;
- [x] freeze runtime-visible feature definitions and confidence baseline formulas;
- [x] implement uncalibrated baselines;
- [x] implement transparent learned calibration;
- [x] choose selective thresholds on calibration data only;
- [x] evaluate unchanged methods on test-ID and test-OOD;
- [x] persist reliability/risk–coverage artifacts and per-query traces;
- [x] inspect overconfidence, false-answer, and false-abstention failures;
- [x] pass final source-of-truth CI gate and document trade-offs.

## Completion

M12 is complete. The frozen benchmark and controls, first held-out calibration/selective/drift evidence, retained overconfidence and feature-collision failures, full reliability/risk–coverage traces, and final source-of-truth gate are recorded in this lab and its artifacts.
