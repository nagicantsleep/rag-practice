from __future__ import annotations

import json
from pathlib import Path

from rag_practice.evaluation.agentic_multi import evaluate_role_split_agent

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "m09_agentic" / "benchmark.json"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

payload = evaluate_role_split_agent(BENCHMARK)
(RESULTS / "multi_agent_results.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
aggregate = payload["aggregate"]
model = payload["model"]
lines = [
    "# M09 exploratory role-split multi-agent results",
    "",
    f"Shared model: `{model['model_id']}` @ `{model['revision']}`",
    "",
    "This is a post-single-agent exploratory control on the same frozen benchmark, not fresh held-out generalization evidence.",
    "",
    "| Task success | Grounded | Plan exact | Tool precision | Evidence complete | Abstention | Recovery | Steps | Tool cost | Role calls | Proposer valid | Critic valid | Proposer ms | Critic ms |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    (
        "| {task_success:.3f} | {grounded_answer_rate:.3f} | {plan_sequence_accuracy:.3f} | "
        "{mean_tool_precision:.3f} | {evidence_complete_rate:.3f} | {abstention_accuracy:.3f} | "
        "{recovery_success:.3f} | {mean_steps:.2f} | {mean_cost_units:.2f} | "
        "{mean_model_role_calls:.2f} | {proposer_valid_decision_rate:.3f} | "
        "{critic_valid_decision_rate:.3f} | {mean_proposer_generation_ms:.1f} | "
        "{mean_critic_generation_ms:.1f} |"
    ).format(**aggregate),
    "",
    "The proposer is unchanged from the recorded single-agent control. The critic/corrector is a new role sharing the same pinned weights; both raw outputs are persisted.",
    "No evaluator labels are available to either role or the coordinator.",
    "",
]
(RESULTS / "multi_agent_results.md").write_text("\n".join(lines))
print("\n".join(lines))
