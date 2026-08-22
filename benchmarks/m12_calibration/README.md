# M12 calibration benchmark construction contract

This file freezes M12.0 construction and evaluation semantics **before** any M12 calibrator implementation or held-out test inspection.

## Scope

M12 evaluates confidence calibration and selective answering around one deliberately simple, deterministic RAG mechanism. It is not a retrieval leaderboard. The base mechanism is intentionally imperfect so calibration must distinguish confident-correct, uncertain-correct, overconfident-wrong, no-evidence, stale/conflicting, and shifted cases.

## Frozen split sizes

`benchmark.json` will contain exactly **40 query instances** with fixed IDs and non-overlapping entity families:

- `train`: 12
- `calibration`: 10
- `test_id`: 10
- `test_ood`: 8

Entity IDs may not cross splits. `test_ood` differs from training/calibration in at least one declared shift dimension: vocabulary alias, query style, entity naming pattern, or evidence/distractor distribution.

## Frozen task strata

Across the 40 instances the benchmark must include all of:

- direct answerable evidence;
- paraphrase/alias answerable evidence;
- close distractor / retrieval near miss;
- missing evidence / genuinely unanswerable;
- stale evidence that must not support an answer;
- conflicting trusted evidence;
- answerable evidence where the fixed extractive generator is expected to fail because the answer token/string is not extractable under its frozen rule;
- shifted/OOD answerable and unanswerable cases.

Each query freezes evaluator-only fields:

- `answerable`;
- `expected_answer` or `null`;
- `required_evidence_ids`;
- `forbidden_evidence_ids`;
- `class`;
- `shift_class` (`id`, `lexical_alias`, `query_style`, `entity_pattern`, or `distractor_shift`);
- `split`.

## Frozen corpus record schema

Each document contains:

- `id`;
- `entity_id`;
- `text`;
- `updated_generation`;
- `trusted`;
- `active`.

Only `active=true` and `trusted=true` documents are valid evidence. Invalid documents remain indexable to create calibration failures; the base retriever is intentionally unaware of validity, while validity is exposed as an observable post-retrieval signal.

## Frozen base RAG mechanism

### Tokenization

Lowercase ASCII word tokens matching `[a-z0-9]+`.

Frozen stopword set:

`the,a,an,is,are,was,were,what,which,for,of,to,in,on,and,or,does,do,with,current,please,tell,me`

Content tokens are unique query/document tokens after stopword removal.

### Retrieval

For each query, score every document by:

`score(q,d) = |content(q) ∩ content(d)| / max(1, |content(q)|)`

Rank descending by `(score, document_id ascending)` and retain top 3.

No validity/trust/active filtering occurs before ranking. This is deliberate: the calibration layer must see when a high lexical score points to invalid evidence.

### Fixed answer generator

The generator receives only top-1 text.

1. If top-1 score is `0`, emit `UNKNOWN`.
2. Otherwise search top-1 text for the first exact marker `ANSWER=<value>` where `<value>` is the maximal contiguous `[A-Z0-9_-]+` token immediately after `ANSWER=`.
3. If found, emit that value.
4. Otherwise emit `UNKNOWN`.

This keeps answer behavior inspectable and creates controlled generator-format failures when evidence is semantically present but not encoded with the exact marker.

## Frozen runtime-visible feature vector

No feature may use expected answers, qrels, answerability, class, split, or shift labels.

For every query compute exactly:

1. `top1_score` — frozen lexical score of rank 1.
2. `top2_score` — score of rank 2, or `0` if absent.
3. `margin` — `top1_score - top2_score`.
4. `top1_valid` — `1` iff rank-1 document has both `active=true` and `trusted=true`, else `0`.
5. `top3_valid_fraction` — valid document count among retrieved top 3 divided by retrieved count.
6. `top3_entity_agreement` — maximum fraction of top-3 retrieved documents sharing one `entity_id`.
7. `answer_present` — `1` iff generator output is not `UNKNOWN`.
8. `answer_support` — `1` iff generator output string occurs literally in the top-1 document text, else `0`.
9. `conflict_signal` — `1` iff at least two valid top-3 documents share the query entity and contain distinct `ANSWER=` values, else `0`.
10. `retrieved_count` — number of returned top-k documents, frozen at `min(3, corpus_size)` but retained for trace completeness.

Feature computation is deterministic and frozen before labels are inspected by any learned method.

## Frozen confidence baselines

M12.1 must implement exactly these before learned calibration:

- `constant`: confidence `0.5` for every query;
- `top1`: confidence clipped to `[0,1]` from `top1_score`;
- `margin`: confidence clipped to `[0,1]` from `margin`;
- `hand_composed`: `clip(0.45*top1_score + 0.25*margin + 0.15*top1_valid + 0.10*answer_present + 0.05*answer_support - 0.25*conflict_signal, 0, 1)`.

No coefficients may be altered after held-out evidence is observed.

## Frozen learned calibrator contract

Primary learned method: logistic/Platt-style calibration with binary correctness target and the exact 10-feature vector above.

- fit rows: `train` only;
- standardization: none;
- intercept: yes;
- optimizer: deterministic batch gradient descent;
- seed: 61;
- epochs: 400;
- learning rate: 0.10;
- L2 coefficient: 0.01 applied to weights, not intercept;
- sigmoid output clipped to `[1e-6, 1-1e-6]` for log-loss only.

The implementation may add isotonic calibration only as a secondary method after the primary contract has produced evidence; it cannot replace the frozen logistic result.

## Frozen selective threshold policy

For every confidence method, choose a single threshold using **calibration split only**.

Target: minimize selective risk subject to coverage `>= 0.60` on calibration. Ties break by:

1. higher coverage;
2. higher threshold;
3. numeric threshold ascending as final deterministic tie-break.

Candidate thresholds are `{0.00, 0.05, ..., 1.00}`.

The selected threshold is then applied unchanged to both `test_id` and `test_ood`.

## Frozen correctness label

A query is correct iff all are true:

- it is answerable;
- generator output exactly equals `expected_answer`;
- every `required_evidence_id` is present in retrieved top 3;
- no `forbidden_evidence_id` is used as rank-1 answer evidence;
- if `conflict_signal=1`, the query is not counted correct unless its expected answer is explicitly `CONFLICT` and the generator emits `CONFLICT`.

For genuinely unanswerable queries, correctness of the raw generator is `0`; abstention quality is measured separately. This prevents an `UNKNOWN` string from being treated as a correct factual answer.

## Frozen calibration metrics

Evaluate separately on `test_id` and `test_ood`:

- Brier score;
- binary log loss with epsilon `1e-6`;
- ECE with fixed bins `[0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]`;
- mean confidence on correct and incorrect rows.

ECE bins are fixed here and cannot be changed after test inspection.

## Frozen selective metrics

For each method and split:

- selected threshold from calibration;
- coverage;
- selective risk among answered rows;
- false-answer rate over all rows;
- false-abstention rate over answerable rows;
- abstention accuracy over unanswerable rows;
- risk at coverage targets `0.50`, `0.70`, `0.90` using descending-confidence prefixes;
- AURC from the discrete descending-confidence risk–coverage curve, with deterministic tie-break by query ID.

## Frozen drift report

For every method record OOD minus ID deltas for:

- full-coverage base correctness;
- Brier;
- ECE;
- AURC;
- thresholded coverage;
- thresholded selective risk;
- mean confidence.

A calibrator that improves ID calibration but degrades OOD selective risk must retain both results without retuning.

## Integrity rules

1. Benchmark instances/splits must be frozen in a separate commit before M12 calibration implementation.
2. Runtime feature code never reads evaluator-only labels.
3. Train correctness labels may be used only for fitting the frozen logistic calibrator.
4. Calibration correctness labels may be used only for threshold selection and diagnostic metrics, not logistic fitting.
5. Test-ID/OOD labels are evaluator-only and may not affect features, coefficients, threshold selection, or hyperparameters.
6. First valid held-out result ends the benchmark-edit window except documented label/serialization defects independent of model outcomes.
7. Representative overconfidence and false-abstention failures must be retained.
8. Timing numbers are educational implementation sanity only.

## Next step

Create `benchmark.json` with the exact 40 frozen instances satisfying this contract. Do not implement M12 confidence methods until that instance freeze commit exists.
