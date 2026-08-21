from __future__ import annotations

import json
from pathlib import Path

from rag_practice.evaluation.agentic import evaluate_all

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "m09_agentic" / "benchmark.json"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

payload = evaluate_all(BENCHMARK)
(RESULTS / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

lines = [
    "# M09 agentic RAG deterministic results",
    "",
    "| System | Task success | Grounded | Plan exact | Tool precision | Evidence complete | Abstention | Recovery | Steps | Cost units |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for key in ("docs_only", "static_router", "agent_loop"):
    aggregate = payload["systems"][key]["aggregate"]
    lines.append(
        "| {name} | {task_success:.3f} | {grounded_answer_rate:.3f} | "
        "{plan_sequence_accuracy:.3f} | {mean_tool_precision:.3f} | "
        "{evidence_complete_rate:.3f} | {abstention_accuracy:.3f} | "
        "{recovery_success:.3f} | {mean_steps:.2f} | {mean_cost_units:.2f} |".format(
            name=key,
            **aggregate,
        )
    )

lines.extend(
    [
        "",
        "Latency is a local/GitHub Actions CPU sanity measurement; tool costs are the frozen synthetic cost units.",
        "Per-task JSON retains action arguments, observations, evidence ids, recovery count, latency, and cost.",
        "",
    ]
)
(RESULTS / "results.md").write_text("\n".join(lines))
print("\n".join(lines))
