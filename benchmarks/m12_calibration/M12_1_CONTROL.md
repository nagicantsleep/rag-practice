# M12.1 calibration evaluator control

Frozen after the M12.0 instance freeze (`b2bd91f99dd52aa27b656b6ef306fb7fb51bd1eb`) and before M12 confidence implementation or held-out result inspection.

This file resolves metric/inference details left implicit in the construction contract without changing any benchmark instance, label, split, feature, coefficient, or hyperparameter.

## Runtime boundary

At inference time the base RAG and confidence methods receive only:

- query ID for trace identity;
- query text;
- entity ID as ordinary structured request metadata;
- generated corpus documents and their runtime metadata (`active`, `trusted`, `updated_generation`);
- the frozen feature vector derived from those runtime objects.

They do **not** receive split, class, shift class, answerability, expected answer, required evidence IDs, or forbidden evidence IDs.

## Correctness target

The binary logistic target on `train` is the frozen raw-answer correctness label defined in the benchmark README. It is computed by the evaluator after the runtime trace is produced.

## Threshold decision

A row is `ANSWER` iff `confidence >= threshold`; otherwise it is `ABSTAIN`.

Calibration threshold selection enumerates exactly `0.00, 0.05, ..., 1.00` using decimal values rounded to two digits before comparison.

## Risk/coverage curve

For a split of `n` rows:

1. sort by confidence descending;
2. tie-break by query ID ascending;
3. for prefix `k=1..n`, coverage is `k/n` and risk is errors in prefix / `k`;
4. AURC is the arithmetic mean of the `n` prefix risks, equivalent to a right-endpoint rectangle sum over equal `1/n` coverage increments from zero to one.

Risk at target coverage `c` uses the smallest prefix `k` with `k/n >= c`.

## Selective denominators

- thresholded coverage: answered / all rows;
- selective risk: wrong answered / answered, or `0` when none are answered;
- false-answer rate: wrong answered / all rows;
- false-abstention rate: abstained answerable / answerable rows;
- abstention accuracy: abstained unanswerable / unanswerable rows, or `1` if a split contains no unanswerable rows.

## ECE

Use the fixed five bins from the construction contract. A bin contributes:

`(bin_count / n) * abs(mean_confidence - empirical_correctness)`.

`0.2`, `0.4`, `0.6`, and `0.8` belong to the higher bin; `1.0` belongs to the last bin.

## Logistic optimization

Initialize all 10 weights and the intercept to `0`. For each of 400 epochs compute full-batch binary-cross-entropy gradients over all train rows, add `0.01 * weight` to each weight gradient, do not regularize intercept, then update by learning rate `0.10`.

Inference probability is the ordinary sigmoid of `intercept + dot(weights, features)`. Only log-loss calculation clips probabilities to `[1e-6, 1-1e-6]`.

## Acceptance boundary

M12.1 does not require the learned calibrator to win. The first valid held-out result is retained even if raw score baselines are better. Implementation defects may be repaired only when they violate this control or the already-frozen M12.0 contract.
