# M09 — Agentic RAG

Status: **IN PROGRESS — EVIDENCE RECORDED / FINAL GATE PENDING**

## Goal

Make agentic control observable before introducing framework abstractions. Phase 1 compares a docs-only one-shot baseline, a one-tool static router, and a bounded deterministic single-agent loop with explicit state, source/tool routing, evidence checks, recovery, stop policy, and cost accounting. Phase 2 holds that benchmark fixed and tests a pinned pretrained model as the tool planner only. Phase 3 adds a shared-checkpoint proposer/critic role split only after the single-agent result is recorded.

The benchmark was frozen in commit `de6f978ab7f14ea1a792a591aa468795b13f92d9` before the agent implementation and before any pretrained agent-policy result was inspected.

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

## Pinned single-agent model planner

`HuggingFaceTB/SmolLM2-135M-Instruct` is pinned to revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` and used only for next-tool selection. It must emit exactly one strict line: `tool|argument` or `STOP`. Final answers still use the same qrel-blind deterministic evidence reader, so this control isolates planner/tool-selection behavior rather than mixing it with free-form answer generation.

| Task success | Grounded | Plan exact | Tool precision | Evidence complete | Abstention | Recovery | Steps | Tool cost | Planner calls | Valid decisions | Prompt tokens | Planner generation ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.167 | 0.167 | 0.000 | 0.000 | 0.167 | **1.000** | 0.000 | 0.00 | 0.00 | 1.00 | 0.000 | 153.2 | ~571.7 |

### Single-agent model findings

- **The pinned planner is a retained negative result.** It produces no valid DSL action on any frozen task, so zero tools execute and the deterministic evidence reader abstains on every query.
- **The apparent `0.167` task success is not agent success.** It comes only from the two frozen no-evidence tasks (`a8`, `a10`) where zero evidence correctly maps to `ABSTAIN`; plan-sequence accuracy remains `0.0` and recovery remains `0.0`.
- **Most failures are format/control failures, not tool-runtime failures.** Ten tasks emit `"(none)"` or similarly invalid output instead of a legal action. The direct calculator task emits prose rather than `calculator|17 + 25`. Recovery task `a9` emits `Answer: STOP` plus an explanation instead of a legal first action.
- **No expected-answer-aware parser or post-hoc action repair is added.** The strict parser, prompt, model revision, benchmark, action budget, and source data remain unchanged after first pretrained inspection.
- **A small instruction model can know task semantics yet fail an agent protocol.** Tool-use reliability therefore needs its own evaluation; treating any fluent text as an action would hide exactly this failure.
- **Planner latency is measurable even when no tools run.** The first decision averages roughly `572 ms` CPU generation and `153` prompt tokens/query, while tool cost stays `0` because every emitted action is invalid.

## Exploratory shared-checkpoint role split

Only after the single-agent result was recorded, the unchanged proposer was paired with a new verifier/corrector role using the same pinned checkpoint. Each cycle runs proposer → critic; the coordinator uses a valid critic action, falls back to a valid proposer action only if the critic is invalid, or stops when neither role yields a valid action. Neither role receives evaluator labels.

| Task success | Grounded | Plan exact | Tool precision | Evidence complete | Abstention | Recovery | Steps | Tool cost | Model-role calls | Proposer valid | Critic valid | Proposer ms | Critic ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.167 | 0.167 | 0.000 | 0.000 | 0.167 | **1.000** | 0.000 | 0.00 | 0.00 | 2.00 | 0.000 | 0.000 | ~325.9 | ~792.5 |

### Role-split findings

- **Adding another role does not automatically create tool-use reliability.** The critic also produces no valid DSL action on any frozen task, so the role split executes zero tools and leaves quality unchanged from the single-agent negative result.
- **The second role increases model cost without improving evidence.** Mean model-role calls rise from `1.0` to `2.0`; the critic adds roughly `793 ms` generation/query on top of proposer generation while task success/evidence completeness stay `0.167`.
- **Shared weights are not independent agents.** This is a role-separation/control experiment using one pinned checkpoint, not evidence that heterogeneous or independently trained agents behave the same way.
- **The result is post-hoc on the same test benchmark.** The critic architecture was introduced only after inspecting the single-agent result, so it is explicitly exploratory and not fresh held-out generalization evidence.
- **Raw role outputs remain visible.** Representative proposer and critic outputs are both `"(none)"`; no coordinator heuristic converts malformed text into a tool action.
- **More agent calls can make a system worse on efficiency while leaving correctness unchanged.** Multi-agent comparisons therefore need task quality, action quality, token/model-call cost, and latency together rather than success alone.

## Evidence

- Benchmark freeze: `de6f978ab7f14ea1a792a591aa468795b13f92d9`.
- Phase-1 implementation: `f7ce3c4fbd41a6e9d71b56af1a6167b30543d1d8`.
- PR mechanism gate `32485342984` / job `96780479396`: full repository test suite and deterministic M09 evaluator both passed.
- Deterministic JSON/Markdown evidence was persisted by `github-actions[bot]` in commit `b9c1373428322675f23e8f2b7291031fb100670b`.
- Phase-1 findings were recorded in `46fe55ce00cf17767f821e9599acb9351dff7a6e`; subsequent bot evidence refreshes changed timings only.
- Pinned single-agent planner implementation: `55bd2fa6ce1234f75ca9b505c62cde71a30d91ed`.
- Full pinned-planner PR gate `32485883446` / job `96782153068`: full repository tests, deterministic evaluation, and pinned SmolLM2 tool-planner evaluation all passed.
- Deterministic + pretrained JSON/Markdown evidence was persisted in bot commit `f0ddfee3cad69d4493651a1b30add35147e79b35`.
- Exploratory role-split implementation: `1e74eea55ec0fbcd71edd947dc125c90f361f5f7`.
- Role-split PR gate `32486359470` / job `96783661841`: full tests, deterministic evaluation, pinned single-agent evaluation, and exploratory role-split evaluation all passed.
- Deterministic + single-agent + role-split evidence was persisted by `github-actions[bot]` in commit `bc8b184bc38b6ed22f7589ae0273be376ecfc40c`.
- A final source-of-truth full-regression + deterministic + pinned single-agent + role-split gate must pass on the findings/completion head before M09 is marked `DONE`.

## Definition of Done

- [x] Freeze tasks, tool corpora, tool costs, expected evidence/actions, action budget, retry budget, and abstention contract before implementation.
- [x] Implement docs-only and one-tool routing baselines.
- [x] Implement explicit planner/tool/evidence/recovery/stop state loop.
- [x] Persist per-action traces and separate quality/action/cost metrics.
- [x] Pass full repository regression + deterministic M09 evaluator gate.
- [x] Record phase-1 findings and representative baseline/recovery failures.
- [x] Add a pinned model-driven single-agent planner/control on the unchanged benchmark and retain its failures.
- [x] Compare a shared-checkpoint role-split multi-agent variant only after the single-agent model control was recorded, with post-hoc caveat.
- [ ] Pass final source-of-truth full-regression + deterministic + pinned single-agent + role-split gate.
- [ ] Mark M09 complete in ROADMAP and point the immediate next step to M10.

## Guardrails

Expected answers, expected action sequences, expected evidence, recovery labels, and no-evidence labels are evaluator-only. Runtime code must not use them. Failed calls in recovery tasks remain in traces. Agentic quality is not inferred from task success alone: unnecessary actions, tool precision, grounding, recovery, latency, model calls/tokens, and synthetic tool cost remain independent contracts. The role-split experiment is post-single-agent exploratory evidence on this same frozen benchmark, not fresh held-out generalization.
