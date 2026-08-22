# M12 — Calibration, selective prediction, and drift findings

Status: **EVIDENCE COMPLETE / FINAL SOURCE-OF-TRUTH GATE PENDING**

M12 was started only after M11 merged. The learning charter was frozen at `9d9f4bcfd39d53cfdc4beba23fb7c4c25576f887`, the benchmark/evaluation construction contract at `4531122296982a65e8d9a70cee985f9fcb6dc2b6`, the exact 40-instance benchmark at `b2bd91f99dd52aa27b656b6ef306fb7fb51bd1eb`, and the evaluator-control details at `ab304d6cf06e32b8400a21b6b203f282ff6a7644` before the first valid held-out result.

The benchmark and feature/calibrator contracts are unchanged after first test inspection.

## First valid held-out gate

PR run `32559624970` / job `96999119450` passed **172 repository tests** and the frozen M12 evaluator on implementation/workflow head `c1b7e8024f328651760ec23820c8d07cf9294158`.

All four confidence baselines and the learned logistic control were defined before this run. Thresholds were selected only on the frozen calibration split and then applied unchanged to both test slices.

### Test-ID

The underlying frozen RAG answer correctness is `0.300` at full coverage for every confidence method; calibration changes only confidence/selection, not the base answer.

| Method | Brier | ECE | AURC | Threshold | Coverage | Selective risk | False-answer rate | Abstention accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 0.250 | 0.200 | 0.371 | 0.50 | 1.000 | 0.700 | 0.700 | 0.000 |
| raw top-1 score | 0.547 | 0.593 | 0.371 | 1.00 | 0.800 | 0.625 | 0.500 | 1.000 |
| raw margin | 0.178 | **0.067** | 0.371 | 0.30 | 0.700 | 0.571 | 0.400 | 1.000 |
| hand-composed | 0.287 | 0.365 | 0.371 | 0.65 | 0.700 | 0.571 | 0.400 | 1.000 |
| logistic | **0.158** | 0.219 | 0.430 | 0.35 | 0.600 | **0.500** | **0.300** | 1.000 |

The learned calibrator therefore improves Brier score and the frozen-threshold false-answer/selective-risk result, but **does not dominate**. The simple margin baseline has substantially better ID ECE, and logistic AURC is worse (`0.430` versus `0.371`). These losses are retained rather than tuned away.

### Test-OOD

The OOD slice has higher raw answer correctness (`0.500`) than the ID slice (`0.300`) on this small synthetic benchmark, so M12 does **not** claim distribution shift universally makes the task harder.

| Method | Brier | ECE | AURC | Threshold | Coverage | Selective risk | False-answer rate | Abstention accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 0.250 | 0.000 | 0.183 | 0.50 | 1.000 | 0.500 | 0.500 | 0.000 |
| raw top-1 score | 0.411 | 0.463 | 0.260 | 1.00 | 0.750 | 0.500 | 0.375 | 1.000 |
| raw margin | 0.240 | 0.229 | 0.239 | 0.30 | 0.625 | 0.400 | 0.250 | 1.000 |
| hand-composed | 0.166 | 0.236 | 0.239 | 0.65 | 0.500 | 0.250 | **0.125** | 1.000 |
| logistic | **0.112** | 0.319 | **0.183** | 0.35 | 0.625 | **0.200** | **0.125** | 1.000 |

For logistic, OOD minus ID ECE is `+0.100`: calibration becomes worse by this metric even though OOD Brier, AURC, and selective risk improve because the frozen OOD cases happen to be easier and more separable. This is a useful example of why “drift” and “difficulty” are not synonyms.

## Hypothesis evaluation

### H1 — raw similarity is not calibrated confidence: supported

Raw top-1 similarity is highly overconfident on test-ID: Brier `0.547`, ECE `0.593`, mean top-1 confidence about `0.893` while raw answer accuracy is only `0.300`. High lexical similarity is not a probability of answer correctness.

### H2 — multi-signal calibration improves selective risk: partially supported

The logistic model reaches the best ID Brier (`0.158`) and lowers frozen-threshold false-answer rate from the always-answer reference `0.700` to `0.300`. However, margin has better ID ECE (`0.067`) and logistic has worse ID AURC (`0.430`). Calibration and ranking-for-abstention are separate objectives.

### H3 — ID calibration can degrade under shift: partially supported

The unchanged logistic calibrator's ECE worsens by `+0.100` on OOD and one stale OOD case remains above the ID-selected threshold. But other OOD metrics improve because this particular shifted slice is easier. The evidence supports metric-specific calibration drift, not a universal “OOD is worse” claim.

### H4 — abstention has asymmetric utility: supported

At the frozen ID threshold the logistic policy answers 60% of queries and cuts false-answer rate to 30%, compared with 70% for always-answer. On OOD it answers 62.5% with a 12.5% false-answer rate. Coverage loss and risk reduction are reported together rather than hiding abstention behind answered-only accuracy.

## Representative retained failures

### Observable-feature collision: correct direct vs wrong near miss

`q23`/`q24` are correct direct cases while `q26`/`q27` are wrong near-miss cases. Under the frozen feature contract they share essentially the same vector: top-1 `1.0`, top-2 `0.667`, margin `0.333`, valid top-1, all retrieved evidence valid, answer present/supported, no conflict, and three retrieved documents.

The logistic model therefore assigns the same confidence, about `0.633`, to both correct and wrong cases. A calibrator cannot separate errors that its input representation does not encode. Adding a semantic judge after seeing these held-out failures would violate the experiment, so the collision is retained. This explains why Brier can improve while AURC does not.

### Conflict and generator-format failures are visible

The ID conflict case `q30` produces the wrong raw answer but has `conflict_signal=1`; logistic confidence is about `0.198`, below threshold `0.35`. The generator-format case `q31` has strong valid evidence but no extractable `ANSWER=` marker, producing `UNKNOWN`; `answer_present=0` and logistic confidence is about `0.270`, also below threshold.

These demonstrate that runtime evidence-state and generation-state signals can identify some failure classes without evaluator labels.

### OOD stale evidence remains an overconfidence failure

`q39` retrieves an inactive stale rank-1 document. The frozen `top1_valid=0` signal lowers confidence, but logistic still emits about `0.375`, just above the calibration-selected threshold `0.35`, so the selective policy answers incorrectly. This failure is retained and contributes to the OOD ECE degradation.

### OOD near-miss happens to be easier to reject

`q37` is a shifted near-miss, but its top-3 set creates `conflict_signal=1`, making it more distinguishable from correct direct cases than the ID near misses. Logistic confidence falls to about `0.232` and the policy abstains. This is one reason the OOD slice has lower selective risk despite being distribution-shifted.

## Artifact completeness

The persisted JSON records every query's answer, evaluator correctness, answerability/shift label, retrieved evidence IDs, document validity metadata, exact 10-feature vector, and every method's confidence. It also persists fixed-bin reliability data, Brier/ECE/log loss, frozen-threshold selective metrics, AURC and risk at target coverages.

Commits `c31a11f00a91a2e97b04c2defb86b74840d6a2d6` and `582c53031f880dbd2c053fa23675e92f40940761` add only diagnostics already required by the frozen evaluation axes: every discrete risk–coverage curve point plus feature/calibrator implementation-sanity timing and regression assertions. They do not alter the benchmark, features, confidence formulas, logistic weights/hyperparameters, threshold policy, correctness rule, ECE bins, AURC definition, or any held-out decision.

Isotonic calibration is not added: the calibration split contains only 10 rows and the charter made it conditional on enough data to avoid degenerate fitting. Conformal-style control was optional and no post-result contract is introduced. No OOD recalibration is performed.

## M12 completion interpretation

M12 demonstrates that:

- retrieval similarity can be badly miscalibrated even when it looks intuitively confident;
- a transparent multi-signal calibrator can reduce probabilistic error and frozen-threshold false answers without improving the base RAG answer itself;
- lower Brier does not imply better ECE or better risk ordering;
- selective prediction is constrained by the information in its observable signals;
- drift must be reported per metric, because a shifted slice can be easier overall while still becoming worse calibrated;
- abstention policy quality requires explicit coverage/risk accounting.

The next accepted evidence is the final source-of-truth CI gate on the completed code, diagnostic artifacts, findings, and completion workflow. No post-hoc benchmark or calibrator tuning is planned.
