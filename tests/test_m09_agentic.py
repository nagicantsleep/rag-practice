from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_practice.agentic import ToolEnvironment, load_runtime_tasks, run_agent_loop
from rag_practice.evaluation.agentic import evaluate_all

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "m09_agentic" / "benchmark.json"


def test_frozen_agentic_benchmark_contract() -> None:
    payload = json.loads(BENCHMARK.read_text())
    assert payload["version"] == 1
    assert len(payload["tasks"]) == 12
    assert payload["frozen_contract"]["docs_top_k"] == 2
    assert payload["frozen_contract"]["max_actions"] == 4
    assert payload["frozen_contract"]["max_recoveries"] == 1
    assert payload["frozen_contract"]["abstain_token"] == "ABSTAIN"
    assert payload["frozen_contract"]["tool_costs"] == {
        "docs_search": 2.0,
        "inventory_lookup": 1.0,
        "status_lookup": 1.0,
        "calculator": 0.5,
    }


def test_runtime_tasks_do_not_expose_evaluation_labels() -> None:
    runtime, tasks = load_runtime_tasks(BENCHMARK)
    assert set(vars(tasks[0])) == {"id", "question"}
    assert "expected_answer" not in runtime
    assert "tasks" not in runtime


def test_frozen_tools_keep_source_boundaries_explicit() -> None:
    runtime, _ = load_runtime_tasks(BENCHMARK)
    env = ToolEnvironment(runtime)

    multi_source = env.call(
        "docs_search",
        "Report the Orion backup status and the Cedar rollback phrase.",
    )
    assert multi_source.evidence_ids == ("d2", "d3")
    assert "degraded" not in multi_source.observation

    missing_alias = env.call("inventory_lookup", "Atlas field kit")
    assert not missing_alias.success
    assert missing_alias.observation == "NOT_FOUND"
    assert missing_alias.evidence_ids == ()

    assert env.call("inventory_lookup", "SKU-A17").observation == "14"
    assert env.call("status_lookup", "svc-orion-backup").observation == "degraded"
    assert env.call("calculator", "17 + 25").observation == "42"
    assert not env.call("calculator", "17 * 25").success


def test_agent_loop_matches_frozen_action_contract_without_labels() -> None:
    payload = json.loads(BENCHMARK.read_text())
    runtime, tasks = load_runtime_tasks(BENCHMARK)
    env = ToolEnvironment(runtime)
    expected = {item["id"]: item for item in payload["tasks"]}

    for task in tasks:
        outcome = run_agent_loop(task, env)
        frozen = expected[task.id]
        assert outcome.answer == frozen["expected_answer"]
        assert [action.tool for action in outcome.actions] == frozen["expected_actions"]
        assert len(outcome.actions) <= payload["frozen_contract"]["max_actions"]


def test_agent_loop_retains_declared_recovery_misses() -> None:
    runtime, tasks = load_runtime_tasks(BENCHMARK)
    env = ToolEnvironment(runtime)
    task_by_id = {task.id: task for task in tasks}

    inventory_recovery = run_agent_loop(task_by_id["a9"], env)
    assert inventory_recovery.recoveries == 1
    assert [action.tool for action in inventory_recovery.actions] == [
        "inventory_lookup",
        "docs_search",
        "inventory_lookup",
    ]
    assert inventory_recovery.actions[0].observation == "NOT_FOUND"
    assert inventory_recovery.actions[-1].observation == "14"

    status_recovery = run_agent_loop(task_by_id["a10"], env)
    assert status_recovery.recoveries == 1
    assert [action.tool for action in status_recovery.actions] == [
        "status_lookup",
        "docs_search",
    ]
    assert status_recovery.actions[0].observation == "NOT_FOUND"
    assert status_recovery.answer == "ABSTAIN"


def test_phase1_metrics_separate_task_quality_actions_and_cost() -> None:
    results = evaluate_all(BENCHMARK)
    docs = results["systems"]["docs_only"]["aggregate"]
    static = results["systems"]["static_router"]["aggregate"]
    agent = results["systems"]["agent_loop"]["aggregate"]

    assert docs["task_success"] == pytest.approx(0.25)
    assert docs["evidence_complete_rate"] == pytest.approx(0.25)
    assert docs["mean_steps"] == pytest.approx(1.0)
    assert docs["mean_cost_units"] == pytest.approx(2.0)

    assert static["task_success"] == pytest.approx(5 / 12)
    assert static["plan_sequence_accuracy"] == pytest.approx(4 / 12)
    assert static["recovery_success"] == pytest.approx(0.0)
    assert static["mean_steps"] == pytest.approx(1.0)

    assert agent["task_success"] == pytest.approx(1.0)
    assert agent["grounded_answer_rate"] == pytest.approx(1.0)
    assert agent["plan_sequence_accuracy"] == pytest.approx(1.0)
    assert agent["mean_tool_precision"] == pytest.approx(1.0)
    assert agent["mean_tool_recall"] == pytest.approx(1.0)
    assert agent["mean_unnecessary_action_rate"] == pytest.approx(0.0)
    assert agent["evidence_complete_rate"] == pytest.approx(1.0)
    assert agent["abstention_accuracy"] == pytest.approx(1.0)
    assert agent["recovery_success"] == pytest.approx(1.0)
    assert agent["mean_steps"] == pytest.approx(23 / 12)
    assert agent["mean_cost_units"] == pytest.approx(31 / 12)
