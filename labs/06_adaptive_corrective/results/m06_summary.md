# M06 — Multi-hop, Active, Adaptive, and Self-correcting RAG

## Status

`DONE` candidate pending the final documentation-tree CI/merge gate.

M06 is closed as a mixed-result milestone, not as a claim that every adaptive or reflective mechanism improves quality. The strongest result is the routing/iterative/corrective control layer. The simple active-retrieval + lexical-reflection layer reduces some unsupported answers but adds cost and introduces a false refusal.

## Hypothesis

Routing no-retrieval, single-hop, and iterative questions before retrieval should reduce unnecessary calls and improve multi-hop evidence completeness. A qrel-blind retrieval-quality gate should recover deliberately stale primary-source questions through a fallback source. Once generation is added, confidence-triggered retrieval plus relevance/support/utility reflection should reduce unsupported answers, but may increase latency or over-refuse.

## Controlled setup

- Held-out benchmark: `benchmarks/m06_adaptive/queries.jsonl`.
- Router training data is separate: `benchmarks/m06_adaptive/route_train.jsonl`.
- Runtime never receives qrels, answer references, or `answerable` labels.
- Primary retrieval is BM25 top-1 per control step.
- Iterative retrieval is capped at two controller steps.
- The fallback corpus is a separate controlled source used to model CRAG-style correction without live network access.
- The oracle route is a diagnostic ceiling only.
- Generation uses `google/flan-t5-small` at revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`.
- Generation confidence is the geometric mean of greedy selected-token probabilities. It is an inspectable trigger signal, **not** calibrated factual confidence.
- Active retrieval is capped at one extra call and uses `original question + first draft answer`.
- Reflection uses transparent lexical relevance/support/utility signals. It is not the trained reflection-token model from Self-RAG.
- Thresholds were fixed before the held-out generation run; there is no post-hoc threshold tuning on this test set.

## Phase 1 — routing, iterative retrieval, and correction

| System | Route acc | Evidence recall | Evidence complete | Mean calls | Unnecessary retrieval | Iterative under-route | Correction P | Correction R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_single | 0.583 | 0.812 | 0.625 | 1.33 | 1.000 | 1.000 | 0.500 | 1.000 |
| keyword_router | 1.000 | 1.000 | 1.000 | 1.25 | 0.000 | 0.000 | 1.000 | 1.000 |
| naive_bayes_router | 1.000 | 1.000 | 1.000 | 1.25 | 0.000 | 0.000 | 1.000 | 1.000 |
| oracle_route_ceiling | 1.000 | 1.000 | 1.000 | 1.25 | 0.000 | 0.000 | 1.000 | 1.000 |

The learned Naive Bayes router and the transparent keyword router match the oracle route on this deliberately small/template-like held-out set. This is useful mechanism evidence, but **not** evidence that either router generalizes to broad production traffic.

The three iterative traces demonstrate the intended bridge mechanism:

- Atlas → Vega → Raft (`d2 → d3`)
- Project Ember → Oslo → Norway (`d1 → d4`)
- Quartz scripts → Python → Guido van Rossum (`d5 → d6`)

The two stale-source questions are corrected from stale primary documents to the current fallback facts with correction precision/recall both `1.0` for the adaptive systems.

## Phase 2 — generation, active retrieval, and reflection

| System | Answer F1 | Contains ref | Grounded | Evidence complete | Unsupported answer | Answerable refusal | Unanswerable refusal recall | Mean retrieval calls | Active calls | Attempts | E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_single_rag | 0.517 | 0.600 | 0.875 | 0.625 | 1.000 | 0.000 | 0.000 | 1.33 | 0.00 | 1.00 | 93.90 |
| adaptive_control | 0.717 | 0.800 | 0.875 | 1.000 | 1.000 | 0.000 | 0.000 | 1.25 | 0.00 | 1.00 | 100.43 |
| adaptive_active_reflect | 0.717 | 0.800 | 0.875 | 1.000 | 0.500 | 0.100 | 0.500 | 1.42 | 0.17 | 1.17 | 122.63 |

Timings are hosted-CPU sanity measurements and vary between runs. Quality metrics are deterministic for the pinned model/configuration in the recorded run.

### What improved

- Adaptive control raises final evidence completeness from `0.625` to `1.000`.
- Answer F1 rises from `0.517` to `0.717`; reference containment rises from `0.600` to `0.800`.
- No-retrieval routing removes the baseline's unnecessary retrieval on self-contained queries.
- Iterative retrieval supplies complete two-hop evidence for all three controlled multi-hop cases.
- Corrective fallback recovers the current Aurora badge (`green`) and Nova API port (`7443`) from deliberately stale primary evidence.
- Active/reflection reduces unsupported-answer rate on the two deliberately unanswerable questions from `1.0` to `0.5` by correctly refusing `u1`.

### What did not improve

- Active/reflection does **not** improve aggregate answer correctness over `adaptive_control` (`0.717` F1 for both).
- It increases mean retrieval calls from `1.25` to `1.42`, generation attempts from `1.00` to `1.17`, and recorded end-to-end latency from about `100` to `123 ms/query`.
- It introduces a `0.10` answerable false-refusal rate.

## Error analysis

### 1. Always-retrieve can actively damage no-RAG tasks

For `n1` (`Return the word READY exactly.`), the always-single baseline retrieves unrelated evidence, triggers fallback, and FLAN outputs `f1`. The adaptive router makes zero retrieval calls and returns `READY`.

For `n2` (`2 + 2`), adaptive routing correctly selects no retrieval, but FLAN still outputs `0`. This separates **routing correctness from generator competence**: removing harmful retrieval cannot make a weak generator solve every self-contained task.

### 2. Evidence completeness is not answer correctness

For multi-hop `m1`, adaptive control retrieves both required documents (`d2`, `d3`) and therefore reaches complete evidence. FLAN nevertheless outputs the context label `[d3]` instead of `Raft`, with token-probability confidence around `0.97`.

This is a generator/prompt-formatting failure, not a retrieval miss. It also demonstrates that raw sequence probability is not factual correctness confidence.

### 3. Reflection can correctly detect unsupported output yet choose the wrong recovery action

On `m1`, lexical support marks `[d3]` unsupported even though retrieval is already complete. The active system therefore performs another search, adds unrelated `d6`, receives `[d3]` again, and finally refuses. The refusal is safer than emitting an unsupported label but is **wrong for an answerable query**.

This is the milestone's clearest counterexample to “more retrieval after a bad answer is always helpful.” The missing capability is failure attribution: the controller should distinguish insufficient evidence from generation-format/reasoning failure before retrieving again.

### 4. Reflection successfully catches one unanswerable case

For `u1` (asking for Project Ember's encryption algorithm, which is absent), the first draft is `d1` with low confidence and low lexical support. One additional retrieval does not produce support, so the system refuses with `I do not know.`. This is the one successful unanswerable refusal.

### 5. Lexical support is not semantic answer relevance

For `u2` (asking which package manager Quartz scripts use), FLAN answers `Python`. `Python` is literally present in the retrieved Quartz document, so lexical support marks it supported with high utility even though it does not answer the requested relation. The unsupported answer therefore escapes reflection.

A stronger critic needs relation-aware/entailment-style answer support, not merely token grounding.

## Findings

1. **Routing is a first-class RAG decision.** A no-RAG path can be better than retrieving irrelevant context, and iterative routing can be necessary to make the required evidence searchable.
2. **Candidate/evidence completeness remains a ceiling, but not a guarantee.** Complete retrieval can still produce a wrong answer because generation can fail independently.
3. **Correction is only as meaningful as the retrieval-quality signal.** The controlled `STALE:` marker demonstrates the control mechanism cleanly, but production correction needs a trustworthy freshness/relevance evaluator.
4. **Confidence and reflection need calibration and failure attribution.** High model probability can accompany a clearly wrong label-only output, while a supported token can be semantically non-responsive.
5. **Active retrieval has a cost and can be the wrong action.** On this benchmark it reduces one unsupported answer but causes one false refusal and adds retrieval/generation latency without improving aggregate correctness.
6. **Negative/mixed results are valuable.** M06 does not tune the held-out thresholds to erase these failures; they become explicit regression targets for later agentic control/evaluator work.

## Limitations

- Only 12 held-out queries; route classes are intentionally templated/easy.
- The router's perfect accuracy should not be extrapolated beyond this toy split.
- Multi-hop chains are only two hops and use explicit capitalized bridge entities.
- Primary retrieval is BM25 only.
- The `STALE:` marker and fallback corpus are synthetic controls, not a real freshness/search system.
- FLAN-T5-small is a small instruction model; generator failures are model/prompt-specific.
- Token-probability confidence is uncalibrated.
- Reflection is lexical, not semantic entailment or a trained Self-RAG critic.
- Word counts are token-cost proxies; no paid API cost is incurred.
- CPU latency is a reproducibility/system sanity metric, not a serving benchmark.

## Evaluation evidence

- M06 PR workflow run `32416264406` completed successfully on the complete mechanism/evaluation tree.
- Full repository suite: **70 tests passed**.
- The same run completed both phase-1 routing/correction evaluation and phase-2 pinned FLAN generation/active/reflection evaluation successfully.
- Machine-readable and human-readable results are persisted in `labs/06_adaptive_corrective/results/`.
- Per-query traces include routes, retrieval steps, sources, document IDs, scores, judge assessments, generation confidence, reflection signals, retries/refusals, latency, and prompt/output word proxies.
- The next merge gate is a final CI run after completion documentation/ROADMAP updates.
