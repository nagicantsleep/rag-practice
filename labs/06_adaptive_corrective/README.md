# Lab 06 — Multi-hop, Active, Adaptive, and Self-correcting RAG

Status: `IN PROGRESS`.

M06 introduces control flow only after M05 showed that unconditional query transformation can add large cost without retrieval gains. The milestone therefore asks two separate questions: **when should the system retrieve?** and **what should it do when retrieved evidence is insufficient or wrong?**

## Research mechanisms mapped to inspectable primitives

- **Adaptive-RAG concept:** route among no retrieval, one retrieval step, and iterative retrieval based on question complexity.
- **Multi-hop retrieval:** use newly discovered bridge entities plus unresolved query terms to plan the next retrieval step.
- **CRAG concept:** judge retrieved evidence before generation and trigger a corrective fallback source when primary evidence is stale/low-quality.
- **FLARE concept:** expose an active-retrieval policy that can trigger retrieval from an explicit generation-confidence signal.
- **Self-RAG concept:** expose separate relevance, support, retrieval, and utility reflection signals before later wiring them into answer selection/retry.

These are mechanism implementations for learning and evaluation, not claims of reproducing the trained models from the papers.

## Phase 1 — routing, iterative retrieval, correction

Benchmark: `benchmarks/m06_adaptive/`.

Held-out query regimes:

- `no_retrieval` — the task is self-contained and retrieval is wasteful
- `single` — one relevant fact is enough
- `iterative` — a bridge fact is needed before the second evidence document can be searched effectively
- `single + needs_correction` — the primary corpus contains an explicitly stale fact and the controlled fallback corpus contains the current one

The learned router is a from-scratch multinomial Naive Bayes classifier over unigram/bigram features, trained on `route_train.jsonl` and evaluated on a separate held-out query file.

Baselines/ceilings:

- always-single retrieval
- transparent keyword router
- learned Naive Bayes complexity router
- oracle route as a **diagnostic ceiling only**; it is never presented as a deployable method

Runtime retrieval never receives qrels. The corrective judge sees only the query, retrieved text, and explicit source-quality marker in the controlled corpus.

### Phase-1 metrics

- route accuracy
- evidence recall
- evidence completeness
- mean retrieval calls
- unnecessary retrieval rate on no-retrieval tasks
- iterative under-routing rate
- correction precision/recall
- control latency
- per-step queries, source, document, score, and judge decision

## Phase 2 — active generation and reflection

Planned next:

1. use a pinned instruction model to produce no-context and context-conditioned answers;
2. surface a reproducible confidence signal for FLARE-style active-retrieval triggering;
3. evaluate retrieval-on-demand against always-retrieve and never-retrieve baselines;
4. use relevance/support/utility reflection signals to accept, retry, or refuse;
5. report answer correctness, groundedness, unsupported/refusal behavior, extra retrieval steps, latency, and generated tokens separately from retrieval metrics.

## Completion gate

M06 cannot be `DONE` until both control/retrieval and generation/reflection phases have automated tests, persisted benchmark artifacts, baselines, latency/cost evidence, error analysis, and written findings.
