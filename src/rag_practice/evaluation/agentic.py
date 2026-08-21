"""Evaluation for the frozen M09 agentic RAG benchmark."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from statistics import mean

from rag_practice.agentic import AgentOutcome, RuntimeTask, ToolEnvironment, load_runtime_tasks

Runner = Callable[[RuntimeTask, ToolEnvironment], AgentOutcome]


def _action_match_counts(expected: list[str], actual: list[str]) -> int:
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    return sum(min(expected_counts[name], actual_counts[name]) for name in expected_counts)


def evaluate_policy(name: str, runner: Runner, benchmark_path: str | Path) -> dict:
    payload = json.loads(Path(benchmark_path).read_text())
    runtime, runtime_tasks = load_runtime_tasks(benchmark_path)
    task_by_id = {task.id: task for task in runtime_tasks}
    environment = ToolEnvironment(runtime)

    rows: list[dict] = []
    for frozen in payload["tasks"]:
        task = task_by_id[frozen["id"]]
        outcome = runner(task, environment)
        actual_actions = [action.tool for action in outcome.actions]
        expected_actions = list(frozen["expected_actions"])
        matched_actions = _action_match_counts(expected_actions, actual_actions)

        actual_evidence = {
            evidence_id
            for action in outcome.actions
            for evidence_id in action.evidence_ids
        }
        expected_evidence = set(frozen["expected_evidence"])
        evidence_matches = len(actual_evidence & expected_evidence)
        evidence_recall = evidence_matches / len(expected_evidence) if expected_evidence else 1.0
        evidence_complete = expected_evidence.issubset(actual_evidence)
        answer_correct = outcome.answer == frozen["expected_answer"]
        grounded = answer_correct and (evidence_complete or frozen["no_evidence"])

        tool_precision = matched_actions / len(actual_actions) if actual_actions else 0.0
        tool_recall = matched_actions / len(expected_actions) if expected_actions else 1.0
        unnecessary_action_rate = (
            (len(actual_actions) - matched_actions) / len(actual_actions)
            if actual_actions
            else 0.0
        )

        rows.append(
            {
                "id": frozen["id"],
                "question": frozen["question"],
                "task_class": frozen["task_class"],
                "expected_answer": frozen["expected_answer"],
                "answer": outcome.answer,
                "answer_correct": answer_correct,
                "grounded": grounded,
                "no_evidence": frozen["no_evidence"],
                "abstention_correct": answer_correct if frozen["no_evidence"] else None,
                "requires_recovery": frozen["requires_recovery"],
                "recovery_success": (
                    answer_correct and outcome.recoveries > 0
                    if frozen["requires_recovery"]
                    else None
                ),
                "expected_actions": expected_actions,
                "actual_actions": actual_actions,
                "plan_sequence_exact": actual_actions == expected_actions,
                "tool_precision": tool_precision,
                "tool_recall": tool_recall,
                "unnecessary_action_rate": unnecessary_action_rate,
                "expected_evidence": sorted(expected_evidence),
                "actual_evidence": sorted(actual_evidence),
                "evidence_recall": evidence_recall,
                "evidence_complete": evidence_complete,
                "recoveries": outcome.recoveries,
                "steps": len(outcome.actions),
                "cost_units": outcome.total_cost,
                "latency_ms": outcome.latency_ms,
                "trace": [
                    {
                        "tool": action.tool,
                        "argument": action.argument,
                        "observation": action.observation,
                        "evidence_ids": list(action.evidence_ids),
                        "success": action.success,
                        "latency_ms": action.latency_ms,
                        "cost": action.cost,
                    }
                    for action in outcome.actions
                ],
            }
        )

    no_evidence_rows = [row for row in rows if row["no_evidence"]]
    recovery_rows = [row for row in rows if row["requires_recovery"]]
    return {
        "policy": name,
        "aggregate": {
            "task_success": mean(row["answer_correct"] for row in rows),
            "grounded_answer_rate": mean(row["grounded"] for row in rows),
            "plan_sequence_accuracy": mean(row["plan_sequence_exact"] for row in rows),
            "mean_tool_precision": mean(row["tool_precision"] for row in rows),
            "mean_tool_recall": mean(row["tool_recall"] for row in rows),
            "mean_unnecessary_action_rate": mean(row["unnecessary_action_rate"] for row in rows),
            "mean_evidence_recall": mean(row["evidence_recall"] for row in rows),
            "evidence_complete_rate": mean(row["evidence_complete"] for row in rows),
            "abstention_accuracy": mean(row["abstention_correct"] for row in no_evidence_rows),
            "recovery_success": mean(row["recovery_success"] for row in recovery_rows),
            "mean_steps": mean(row["steps"] for row in rows),
            "mean_cost_units": mean(row["cost_units"] for row in rows),
            "mean_latency_ms": mean(row["latency_ms"] for row in rows),
        },
        "tasks": rows,
    }


def evaluate_all(benchmark_path: str | Path) -> dict:
    from rag_practice.agentic import run_agent_loop, run_docs_only, run_static_router

    return {
        "benchmark": str(benchmark_path),
        "systems": {
            "docs_only": evaluate_policy("docs_only", run_docs_only, benchmark_path),
            "static_router": evaluate_policy("static_router", run_static_router, benchmark_path),
            "agent_loop": evaluate_policy("agent_loop", run_agent_loop, benchmark_path),
        },
    }
