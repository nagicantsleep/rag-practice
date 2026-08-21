from __future__ import annotations

import json
from pathlib import Path

from rag_practice.evaluation.agentic_pretrained import evaluate_smollm_agent

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "m09_agentic" / "benchmark.json"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

payload = evaluate_smollm_agent(BENCHMARK)
(RESULTS / "smollm_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

aggregate = payload["aggregate"]
model = payload["model"]
lines = [
    "# M09 pinned SmolLM2 tool-planner results",
    "",
    f"Model: `{model['model_id']}` @ `{model['revision']}`",
    "",
    "| Task success | Grounded | Plan exact | Tool precision | Evidence complete | Abstention | Recovery | Steps | Tool cost | Planner calls | Valid decisions | Prompt tokens | Planner generation ms |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    (
        "| {task_success:.3f} | {grounded_answer_rate:.3f} | {plan_sequence_accuracy:.3f} | "
        "{mean_tool_precision:.3f} | {evidence_complete_rate:.3f} | {abstention_accuracy:.3f} | "
        "{recovery_success:.3f} | {mean_steps:.2f} | {mean_cost_units:.2f} | "
        "{mean_planner_calls:.2f} | {planner_valid_decision_rate:.3f} | "
        "{mean_planner_prompt_tokens:.1f} | {mean_planner_generation_ms:.1f} |"
    ).format(**aggregate),
    "",
    "The model selects tools only. Final answers are produced by the same qrel-blind deterministic evidence reader used by phase 1, isolating planner/tool-selection behavior.",
    "Raw planner outputs are persisted without expected-answer-aware repair or post-hoc action cleanup.",
    "",
]
(RESULTS / "smollm_results.md").write_text("\n".join(lines))
print("\n".join(lines))
