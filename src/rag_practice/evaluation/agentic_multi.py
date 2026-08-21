"""Exploratory role-split multi-agent evaluation on the frozen M09 benchmark."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean

from rag_practice.agentic import ToolEnvironment, load_runtime_tasks
from rag_practice.agentic.multi_agent import SmolLM2ActionCritic, run_role_split_agent
from rag_practice.agentic.smollm import SmolLM2ToolPlanner


def _action_match_counts(expected: list[str], actual: list[str]) -> int:
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    return sum(min(expected_counts[name], actual_counts[name]) for name in expected_counts)


def evaluate_role_split_agent(benchmark_path: str | Path) -> dict:
    payload = json.loads(Path(benchmark_path).read_text())
    runtime, tasks = load_runtime_tasks(benchmark_path)
    task_by_id = {task.id: task for task in tasks}
    environment = ToolEnvironment(runtime)
    proposer = SmolLM2ToolPlanner()
    critic = SmolLM2ActionCritic(proposer)

    rows: list[dict] = []
    for frozen in payload["tasks"]:
        task = task_by_id[frozen["id"]]
        multi = run_role_split_agent(task, environment, proposer, critic)
        outcome = multi.outcome
        actual_actions = [action.tool for action in outcome.actions]
        expected_actions = list(frozen["expected_actions"])
        matched_actions = _action_match_counts(expected_actions, actual_actions)
        actual_evidence = {
            evidence_id
            for action in outcome.actions
            for evidence_id in action.evidence_ids
        }
        expected_evidence = set(frozen["expected_evidence"])
        evidence_recall = (
            len(actual_evidence & expected_evidence) / len(expected_evidence)
            if expected_evidence
            else 1.0
        )
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
        proposer_traces = [cycle.proposer for cycle in multi.cycles]
        critic_traces = [cycle.critic for cycle in multi.cycles]

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
                "role_cycles": len(multi.cycles),
                "model_role_calls": 2 * len(multi.cycles),
                "proposer_valid_decisions": sum(trace.valid for trace in proposer_traces),
                "critic_valid_decisions": sum(trace.valid for trace in critic_traces),
                "proposer_prompt_tokens": sum(trace.prompt_tokens for trace in proposer_traces),
                "critic_prompt_tokens": sum(trace.prompt_tokens for trace in critic_traces),
                "proposer_output_tokens": sum(trace.output_tokens for trace in proposer_traces),
                "critic_output_tokens": sum(trace.output_tokens for trace in critic_traces),
                "proposer_generation_ms": sum(trace.generation_ms for trace in proposer_traces),
                "critic_generation_ms": sum(trace.generation_ms for trace in critic_traces),
                "cycles": [
                    {
                        "selected_source": cycle.selected_source,
                        "selected_tool": cycle.selected_tool,
                        "selected_argument": cycle.selected_argument,
                        "proposer": {
                            "raw_output": cycle.proposer.raw_output,
                            "parsed_tool": cycle.proposer.parsed_tool,
                            "parsed_argument": cycle.proposer.parsed_argument,
                            "valid": cycle.proposer.valid,
                            "stopped": cycle.proposer.stopped,
                            "prompt_tokens": cycle.proposer.prompt_tokens,
                            "output_tokens": cycle.proposer.output_tokens,
                            "generation_ms": cycle.proposer.generation_ms,
                        },
                        "critic": {
                            "raw_output": cycle.critic.raw_output,
                            "parsed_tool": cycle.critic.parsed_tool,
                            "parsed_argument": cycle.critic.parsed_argument,
                            "valid": cycle.critic.valid,
                            "stopped": cycle.critic.stopped,
                            "prompt_tokens": cycle.critic.prompt_tokens,
                            "output_tokens": cycle.critic.output_tokens,
                            "generation_ms": cycle.critic.generation_ms,
                        },
                    }
                    for cycle in multi.cycles
                ],
                "tool_trace": [
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
    total_cycles = sum(row["role_cycles"] for row in rows)
    total_proposer_valid = sum(row["proposer_valid_decisions"] for row in rows)
    total_critic_valid = sum(row["critic_valid_decisions"] for row in rows)
    return {
        "model": proposer.metadata(),
        "experiment_class": "post-single-agent exploratory shared-checkpoint role split",
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
            "mean_role_cycles": mean(row["role_cycles"] for row in rows),
            "mean_model_role_calls": mean(row["model_role_calls"] for row in rows),
            "proposer_valid_decision_rate": total_proposer_valid / total_cycles,
            "critic_valid_decision_rate": total_critic_valid / total_cycles,
            "mean_proposer_prompt_tokens": mean(row["proposer_prompt_tokens"] for row in rows),
            "mean_critic_prompt_tokens": mean(row["critic_prompt_tokens"] for row in rows),
            "mean_proposer_generation_ms": mean(row["proposer_generation_ms"] for row in rows),
            "mean_critic_generation_ms": mean(row["critic_generation_ms"] for row in rows),
        },
        "tasks": rows,
    }
