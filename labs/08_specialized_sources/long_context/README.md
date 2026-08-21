# M08.7 — Long-context vs retrieval routing

Status: **DONE**

## Goal

Measure when direct full-context reading, retrieval-first context selection, or an explicit router should be preferred while keeping route quality, evidence completeness, answer correctness, grounding, abstention, latency, retrieval calls, and context footprint separate.

The benchmark was frozen in commit `b018a52b112f113ad18447bfc8ab862b5ccded98` before any pretrained reader result was inspected.

## Systems

1. `always_direct` — gives the reader every section in the query's frozen context bundle.
2. `always_retrieve` — BM25 over section text with the frozen top-2 budget.
3. `explicit_router` — reads the whole bundle only when it is small (`<=100` tokenizer words) or the query contains a frozen global marker; otherwise it retrieves.
4. `HuggingFaceTB/SmolLM2-135M-Instruct` pinned to revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, CPU/float32 greedy generation over exactly the context selected by each route policy.

The deterministic reader/router are qrel-blind mechanism controls. The pretrained reader prompt receives only the question plus selected context; qrels, expected answers, preferred routes, and answerability labels are excluded.

## Deterministic mechanism results

| System | Route acc | Evidence recall | Evidence complete | Answer acc | Grounded | Abstention | Context words | Context fraction | Retrieval calls | Unnecessary retrieval | Unnecessary full context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always direct | 0.583 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 490.5 | 1.000 | **0.00** | **0.000** | 1.000 |
| always retrieve | 0.417 | 0.858 | 0.700 | 0.750 | **1.000** | **1.000** | **100.2** | **0.322** | 1.00 | 1.000 | **0.000** |
| explicit router | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 275.5 | 0.667 | 0.42 | **0.000** | **0.000** |

The top-2 retrieval failures are frozen and regression-tested: Atlas global check counting recovers only `1/4` relevant sections, Atlas contact listing `2/3`, and Orion release-code listing `2/3`.

## Pinned pretrained reader results

| System | Route acc | Evidence complete | Strict raw answer acc | Grounded | Abstention | Context words | Prompt tokens | Retrieval calls | CPU generation ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolLM2 always direct | 0.583 | **1.000** | 0.000 | 0.167 | 0.000 | 490.5 | 725.4 | **0.00** | ~2458 |
| SmolLM2 always retrieve | 0.417 | 0.700 | 0.000 | 0.000 | 0.000 | **100.2** | **216.1** | 1.00 | ~1229 |
| SmolLM2 explicit router | **1.000** | **1.000** | 0.000 | 0.083 | 0.000 | 275.5 | 446.2 | 0.42 | ~1666 |

Latest persisted runtime metadata records `134,515,008` parameters / `538,060,032` logical parameter bytes and tokenizer model max length `8192`. Timings are GitHub Actions CPU sanity measurements and vary between runs.

## Findings

- **Full-context reading is a quality ceiling with a context cost, not a universal policy.** The deterministic direct control has complete evidence and answers every frozen query, but consumes the full bundle on every long sparse-fact query and therefore scores unnecessary-full-context rate `1.0` on retrieval-preferred tasks.
- **Retrieval can save context and still destroy global evidence completeness.** Frozen BM25 top-2 cuts mean selected context from `490.5` to `100.2` words, but Evidence Complete falls to `0.7` and deterministic answer accuracy to `0.75` because global count/list questions need more sections than the fixed retrieval window can hold.
- **The explicit router demonstrates the declared mechanism boundary, not learned generalization.** Using only bundle size and frozen query-language markers, it reaches route/evidence/answer accuracy `1.0` while reducing mean selected context to `275.5` words and retrieval calls to `0.42/query`.
- **Correct routing and complete evidence do not imply reader competence.** The pinned SmolLM2 explicit route has route accuracy and Evidence Complete `1.0`, yet strict raw answer accuracy is `0.0`.
- **The strict raw-answer metric deliberately includes format adherence.** Several SmolLM2 fact outputs contain the requested fact but ignore the frozen “final answer only” contract—for example returning `The Cedar service desk hotline is 555-0142.` instead of `555-0142`. No expected-answer-aware parser or post-hoc cleanup is added after observing this behavior.
- **Some failures are semantic, not merely formatting.** With both Orion reserve sections retrieved, SmolLM2 answers `North` instead of `South`; with complete global context it fails Atlas/Lumen counts and Atlas/Orion lists by copying distractor prose or returning only one item.
- **No-evidence handling remains a separate reader failure.** SmolLM2 abstention accuracy is `0.0` under all three route policies: it hallucinates a Cedar cafeteria code and an Atlas paint-color answer instead of returning `ABSTAIN`.
- **Routing changes model cost even when it does not rescue answer quality.** In this CPU run, always-direct averages roughly `725` prompt tokens / `2.46 s` generation, retrieval roughly `216` / `1.23 s`, and the explicit router roughly `446` / `1.67 s` while preserving deterministic evidence completeness.
- **This is controlled evidence only.** The corpus is tiny and synthetic, the explicit router is rule-based, and SmolLM2-135M is one small pinned reader. Neither the mechanism win nor the pretrained negative result is a general long-context leaderboard claim.

## Evidence

- Initial candidate gate `32481053058` / job `96767291207` failed during full-suite collection because the new workflow installed only `.[dev]`; the deterministic evaluator did not run and no result from that gate was accepted.
- Repaired deterministic gate `32481131658` / job `96767529712`: **121 tests passed** and deterministic long-context routing evaluation passed.
- Deterministic JSON/Markdown evidence was persisted in bot commit `184a86e2bc844feb06a5611ec29fd7aa2115e9ad`.
- Full pinned pretrained gate `32481464647` / job `96768559380`: **124 tests passed**, deterministic evaluation passed, and pinned SmolLM2 evaluation passed.
- Deterministic and pretrained JSON/Markdown evidence was persisted in bot commit `51ac1e9a8b29d0c9bae7dacd02e13c34d32ac87c`.
- Final source-of-truth push gate run `32483779972` passed on head `4d79e69ee7eee22ba243e4706c03ed7477112455`: full tests, deterministic evaluation, and pinned SmolLM2 evaluation all succeeded before this automated `[skip ci]` completion update.

## Definition of Done

- [x] Freeze bundle templates/text, section boundaries, queries, qrels, expected answers, preferred routes, retrieval depth, size threshold, and global markers before pretrained inspection.
- [x] Implement direct, retrieval-first, and explicit routing controls.
- [x] Evaluate route accuracy, evidence recall/completeness, answer correctness, grounding, abstention, context footprint, retrieval calls, and latency separately.
- [x] Pass full-regression + deterministic evaluation CI.
- [x] Add a pinned pretrained reader on the unchanged frozen benchmark and retain failures.
- [x] Persist deterministic + pretrained JSON/Markdown results.
- [x] Record representative error analysis without tuning the frozen benchmark or adding post-hoc answer parsing.
- [x] Pass the final source-of-truth full-regression + deterministic + pretrained evaluation gate on the findings head.
- [x] Mark M08.7 and M08 complete in ROADMAP.

## Guardrails

The frozen benchmark is tiny and synthetic. Perfect deterministic route decisions are a mechanism demonstration, not a learned router or production long-context claim. Retrieval/evidence completeness, reader answer quality, strict format adherence, abstention, and context cost remain separate evidence contracts.
