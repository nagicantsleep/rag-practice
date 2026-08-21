# M09 — Agentic RAG

Status: **IN PROGRESS — PHASE 1 MECHANISM GATE PENDING**

## Goal

Make agentic control observable before introducing framework or model-driven planners. Phase 1 compares a docs-only one-shot baseline, a one-tool static router, and a bounded single-agent loop with explicit state, source/tool routing, evidence checks, recovery, stop policy, and cost accounting.

The benchmark was frozen in commit `de6f978ab7f14ea1a792a591aa468795b13f92d9` before the agent implementation and before any pretrained agent-policy result is inspected.

## Frozen phase-1 systems

1. `docs_only` — exactly one `docs_search(question)` call, no tool routing or recovery.
2. `static_router` — chooses exactly one tool from the question, then stops.
3. `agent_loop` — planner → tool → observation/evidence state → next-action/stop loop, bounded to four actions and one recovery transition.

All policies use the same deterministic tool implementations and frozen source data. The planner/reader receive only question + recorded tool state; evaluation labels are loaded separately.

## Metrics

Task success, grounded answer rate, exact action-sequence accuracy, tool precision/recall, unnecessary-action rate, evidence recall/completeness, abstention accuracy, recovery success, steps, synthetic tool cost, and latency are persisted separately.

## Frozen hypotheses

- One-shot document retrieval should work for direct document facts but fail when the answer lives behind a structured/status/calculator tool.
- A static one-tool router should improve direct structured/calculation tasks but still fail cross-tool composition and recovery.
- A bounded agent loop should recover the two declared miss cases and compose cross-source evidence without calling every tool.
- Perfect deterministic behavior, if achieved, is a mechanism control on this tiny synthetic benchmark, not a learned-agent claim.

## Definition of Done

- [x] Freeze tasks, tool corpora, tool costs, expected evidence/actions, action budget, retry budget, and abstention contract before implementation.
- [x] Implement docs-only and one-tool routing baselines.
- [x] Implement explicit planner/tool/evidence/recovery/stop state loop.
- [x] Persist per-action traces and separate quality/action/cost metrics.
- [ ] Pass full repository regression + deterministic M09 evaluator gate.
- [ ] Record phase-1 findings and representative baseline/recovery failures.
- [ ] Add a pinned model-driven single-agent planner/control on the unchanged benchmark.
- [ ] Compare a multi-agent variant only after the single-agent model control is recorded.
- [ ] Pass final source-of-truth gate and mark M09 `DONE`.

## Guardrails

Expected answers, expected action sequences, expected evidence, recovery labels, and no-evidence labels are evaluator-only. Runtime code must not use them. Failed calls in recovery tasks remain in traces. Agentic quality is not inferred from task success alone: unnecessary actions, tool precision, grounding, recovery, latency, and cost remain independent contracts.
