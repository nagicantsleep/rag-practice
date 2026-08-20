# Lab 06 — Multi-hop, Active, Adaptive, and Self-correcting RAG

Status: `DONE` pending final branch CI/merge verification.

M06 introduces control flow only after M05 showed that unconditional query transformation can add large cost without retrieval gains. The milestone asks two separate questions: **when should the system retrieve?** and **what should it do when retrieved evidence is insufficient, stale, or the generated answer is unsupported?**

## Research mechanisms mapped to inspectable primitives

- **Adaptive-RAG concept:** route among no retrieval, one retrieval step, and iterative retrieval based on question complexity.
- **Multi-hop retrieval:** use newly discovered bridge entities plus unresolved query terms to plan the next retrieval step.
- **CRAG concept:** judge retrieved evidence before generation and trigger a corrective fallback source when primary evidence is stale/low-quality.
- **FLARE concept:** use an explicit generation-confidence signal to decide whether another retrieval step may be useful.
- **Self-RAG concept:** expose separate retrieve, relevance, support, and utility reflection signals and use them to retry or refuse.

These are mechanism implementations for learning and evaluation, not claims of reproducing the trained systems from the papers.

## Benchmark and controls

Benchmark: `benchmarks/m06_adaptive/`.

Held-out regimes:

- `no_retrieval` — self-contained tasks where retrieval is wasteful or harmful;
- `single` — one relevant fact is enough;
- `iterative` — a bridge fact is needed before the second evidence document can be searched effectively;
- `single + needs_correction` — the primary corpus contains an explicitly stale fact and a controlled fallback contains the current one;
- unanswerable retrieval questions — evidence is deliberately absent so refusal/unsupported-answer behavior can be measured.

The learned router is a from-scratch multinomial Naive Bayes classifier over unigram/bigram features, trained on `route_train.jsonl` and evaluated on a separate held-out query file. Runtime retrieval/generation never receives qrels, answer references, or answerability labels.

Generation uses pinned `google/flan-t5-small` revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`. The active trigger uses geometric-mean selected-token probability as an inspectable but uncalibrated confidence signal. Reflection uses transparent lexical relevance/support/utility checks.

## Phase 1 — routing, iterative retrieval, correction

| System | Route acc | Evidence recall | Evidence complete | Mean calls | Unnecessary retrieval | Iterative under-route | Correction P | Correction R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_single | 0.583 | 0.812 | 0.625 | 1.33 | 1.000 | 1.000 | 0.500 | 1.000 |
| keyword_router | 1.000 | 1.000 | 1.000 | 1.25 | 0.000 | 0.000 | 1.000 | 1.000 |
| naive_bayes_router | 1.000 | 1.000 | 1.000 | 1.25 | 0.000 | 0.000 | 1.000 | 1.000 |
| oracle_route_ceiling | 1.000 | 1.000 | 1.000 | 1.25 | 0.000 | 0.000 | 1.000 | 1.000 |

The three held-out iterative chains are recovered as `Atlas → Vega → Raft`, `Ember → Oslo → Norway`, and `Quartz → Python → Guido van Rossum`. The two explicitly stale primary facts are corrected to the fallback source.

Perfect held-out routing is **not** treated as broad generalization evidence: the benchmark is deliberately tiny and template-like.

## Phase 2 — generation, active retrieval, reflection

| System | Answer F1 | Contains ref | Grounded | Evidence complete | Unsupported answer | Answerable refusal | Unanswerable refusal recall | Mean retrieval calls | Active calls | Attempts | E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_single_rag | 0.517 | 0.600 | 0.875 | 0.625 | 1.000 | 0.000 | 0.000 | 1.33 | 0.00 | 1.00 | 93.90 |
| adaptive_control | 0.717 | 0.800 | 0.875 | 1.000 | 1.000 | 0.000 | 0.000 | 1.25 | 0.00 | 1.00 | 100.43 |
| adaptive_active_reflect | 0.717 | 0.800 | 0.875 | 1.000 | 0.500 | 0.100 | 0.500 | 1.42 | 0.17 | 1.17 | 122.63 |

Adaptive routing/correction is the clear positive result: it improves evidence completeness and answer quality while removing unnecessary retrieval on no-RAG tasks. Active/reflection is a mixed result: it catches one unsupported unanswerable answer, but does not improve aggregate F1, adds retrieval/generation cost, and falsely refuses one answerable multi-hop query.

## Representative failures retained

- **Correct route, weak generator:** the no-RAG arithmetic query is routed correctly but FLAN still answers incorrectly. Routing cannot repair generator competence.
- **Complete evidence, wrong generation:** for the Atlas → Vega → Raft query, both evidence documents are retrieved but FLAN emits `[d3]` with high sequence confidence.
- **Wrong recovery action:** reflection notices `[d3]` is unsupported, but another retrieval adds unrelated context and the system ultimately refuses an answerable query.
- **Lexical support false positive:** for an unanswerable package-manager question, FLAN answers `Python`. Because `Python` appears in context, the lexical support critic accepts a semantically non-responsive answer.

Full analysis is saved in `results/m06_summary.md`.

## Evaluation artifacts

- `results/control.json` / `control.md` — routing, iterative retrieval, correction, per-step traces.
- `results/generation.json` / `generation.md` — end-to-end generation, confidence, reflection, retries/refusals, latency, prompt/output word proxies.
- `results/m06_summary.md` — hypothesis, controlled comparison, error analysis, findings, limitations, and evidence.
- `.github/workflows/m06-adaptive-corrective.yml` — full regression + both evaluation phases.

## Definition of Done

- [x] learning objective written
- [x] control/generation mechanisms implemented without orchestration frameworks
- [x] core behaviors covered by automated tests
- [x] always-single baseline and oracle diagnostic ceiling defined
- [x] separate router-train and held-out benchmark data defined
- [x] retrieval/control evaluation independent from generation
- [x] answer correctness and groundedness evaluated
- [x] unsupported/refusal behavior evaluated on deliberately unanswerable questions
- [x] latency, retrieval-call, generation-attempt, and word-count cost proxies recorded
- [x] pinned model/configuration recorded
- [x] JSON and Markdown results persisted
- [x] representative failures inspected and retained
- [x] findings/trade-offs written down
- [x] full repository test suite and both M06 evaluation phases passed on the mechanism tree (`32416264406`, 70 tests)
- [ ] final completion-documentation tree passes the same CI gate

M06 is not merged until the final unchecked gate passes.
