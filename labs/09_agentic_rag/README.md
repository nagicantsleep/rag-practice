# M09 — Agentic RAG

Status: **IN PROGRESS — PHASE 1 RECORDED / MODEL CONTROL NEXT**

## Goal

Make agentic control observable before introducing framework or model-driven planners. Phase 1 compares a docs-only one-shot baseline, a one-tool static router, and a bounded single-agent loop with explicit state, source/tool routing, evidence checks, recovery, stop policy, and cost accounting.

The benchmark was frozen in commit `de6f978ab7f14ea1a792a591aa468795b13f92d9` before the agent implementation and before any pretrained agent-policy result is inspected.

## Frozen phase-1 systems

1. `docs_only` — exactly one `docs_search(question)` call, no tool routing or recovery.
2. `static_router` — chooses exactly one tool from the question, then stops.
3. `agent_loop` — planner → tool → observation/evidence state → next-action/stop loop, bounded to four actions and one recovery transition.

All policies use the same deterministic tool implementations and frozen source data. The planner/reader receive only question + recorded tool state; evaluation labels are loaded separately.

## Phase-1 deterministic results

| System | Task success | Grounded | Plan exact | Tool precision | Evidence complete | Abstention | Recovery | Steps | Cost units |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| docs-only | 0.250 | 0.250 | 0.167 | 0.750 | 0.250 | **1.000** | 0.000 | **1.00** | 2.00 |
| static router | 0.417 | 0.417 | 0.333 | **1.000** | 0.417 | **1.000** | 0.000 | **1.00** | **1.54** |
| deterministic agent loop | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 1.92 | 2.58 |

Latency is persisted in JSON as a GitHub Actions CPU sanity measurement; the table uses the frozen synthetic tool-cost units.

## Phase-1 findings

- **Source routing and agentic composition are different capabilities.** The static router improves direct structured/calculator tasks over docs-only, but one correct first tool cannot finish document → inventory/status joins, multi-tool arithmetic, or comparison tasks.
- **A loop earns cost only when later observations matter.** The deterministic agent uses `1.92` tool steps and `2.58` cost units/query versus one step for both baselines, but that extra work raises frozen task/evidence completeness from `0.25/0.25` and `0.417/0.417` to `1.0/1.0`.
- **Recovery must retain the failed action.** `a9` intentionally begins with `inventory_lookup("Atlas field kit") → NOT_FOUND`, then searches documents, discovers `SKU-A17`, and retries inventory. `a10` intentionally misses `falcon-backup`, searches documents, finds no Falcon mapping, and abstains. Both failed calls remain in the persisted trace.
- **Stopping is part of agent quality.** The loop does not fan out to every tool; frozen action-sequence accuracy and tool precision are `1.0`, unnecessary-action rate is `0.0`, and both no-evidence tasks return `ABSTAIN`.
- **Perfect deterministic scores are not a learned-agent result.** The policy is rule-based and the 12-task corpus is tiny/synthetic. It is a mechanism control establishing what planner/tool/evidence/recovery wiring can do when the routing boundary is hand-coded.
- **Task success alone would hide policy quality.** The benchmark therefore persists exact actions, evidence ids, failed calls, grounding, abstention, steps, latency, and cost separately.

## Evidence

- Benchmark freeze: `de6f978ab7f14ea1a792a591aa468795b13f92d9`.
- Phase-1 implementation: `f7ce3c4fbd41a6e9d71b56af1a6167b30543d1d8`.
- PR mechanism gate `32485342984` / job `96780479396`: full repository test suite and deterministic M09 evaluator both passed.
- Deterministic JSON/Markdown evidence was persisted by `github-actions[bot]` in commit `b9c1373428322675f23e8f2b7291031fb100670b`.

## Definition of Done

- [x] Freeze tasks, tool corpora, tool costs, expected evidence/actions, action budget, retry budget, and abstention contract before implementation.
- [x] Implement docs-only and one-tool routing baselines.
- [x] Implement explicit planner/tool/evidence/recovery/stop state loop.
- [x] Persist per-action traces and separate quality/action/cost metrics.
- [x] Pass full repository regression + deterministic M09 evaluator gate.
- [x] Record phase-1 findings and representative baseline/recovery failures.
- [ ] Add a pinned model-driven single-agent planner/control on the unchanged benchmark.
- [ ] Compare a multi-agent variant only after the single-agent model control is recorded.
- [ ] Pass final source-of-truth gate and mark M09 `DONE`.

## Guardrails

Expected answers, expected action sequences, expected evidence, recovery labels, and no-evidence labels are evaluator-only. Runtime code must not use them. Failed calls in recovery tasks remain in traces. Agentic quality is not inferred from task success alone: unnecessary actions, tool precision, grounding, recovery, latency, and cost remain independent contracts.
