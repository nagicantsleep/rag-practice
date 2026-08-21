from __future__ import annotations

from pathlib import Path

from rag_practice.agentic import AgentState, ToolEnvironment, load_runtime_tasks
from rag_practice.agentic.smollm import (
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    PlannerTrace,
    build_messages,
    parse_decision,
    run_model_agent,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "m09_agentic" / "benchmark.json"


def test_pinned_smollm_agent_contract() -> None:
    assert MODEL_ID == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert MODEL_REVISION == "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    assert MAX_NEW_TOKENS == 48


def test_strict_model_action_parser_keeps_raw_failures_visible() -> None:
    assert parse_decision("inventory_lookup|SKU-A17") == (
        ("inventory_lookup", "SKU-A17"),
        True,
        False,
    )
    assert parse_decision("STOP") == (None, True, True)
    assert parse_decision("I think docs_search|Atlas") == (None, False, False)
    assert parse_decision("docs_search|Atlas\nbecause it is useful") == (None, False, False)
    assert parse_decision("unknown_tool|x") == (None, False, False)


def test_model_prompt_contains_only_question_tool_contract_and_recorded_state() -> None:
    runtime, tasks = load_runtime_tasks(BENCHMARK)
    env = ToolEnvironment(runtime)
    state = AgentState(task=tasks[0])
    state.actions.append(env.call("docs_search", tasks[0].question))
    messages = build_messages(tasks[0], state)
    rendered = "\n".join(message["content"] for message in messages)

    assert tasks[0].question in rendered
    assert "docs_search" in rendered
    assert "quiet harbor" in rendered
    assert "expected_answer" not in rendered
    assert "expected_actions" not in rendered
    assert "requires_recovery" not in rendered


class _FakePlanner:
    def __init__(self, decisions: list[tuple[str, str] | None]) -> None:
        self.decisions = list(decisions)

    def decide(self, task, state):
        action = self.decisions.pop(0)
        if action is None:
            return None, PlannerTrace(
                raw_output="STOP",
                parsed_tool=None,
                parsed_argument=None,
                valid=True,
                stopped=True,
                prompt_tokens=10,
                output_tokens=1,
                generation_ms=1.0,
            )
        return action, PlannerTrace(
            raw_output=f"{action[0]}|{action[1]}",
            parsed_tool=action[0],
            parsed_argument=action[1],
            valid=True,
            stopped=False,
            prompt_tokens=10,
            output_tokens=4,
            generation_ms=1.0,
        )


def test_model_agent_runtime_retains_failed_call_and_recovery_transition() -> None:
    runtime, tasks = load_runtime_tasks(BENCHMARK)
    env = ToolEnvironment(runtime)
    task = next(task for task in tasks if task.id == "a9")
    planner = _FakePlanner(
        [
            ("inventory_lookup", "Atlas field kit"),
            ("docs_search", task.question),
            ("inventory_lookup", "SKU-A17"),
            None,
        ]
    )

    result = run_model_agent(task, env, planner)
    assert result.outcome.answer == "14"
    assert result.outcome.recoveries == 1
    assert [action.tool for action in result.outcome.actions] == [
        "inventory_lookup",
        "docs_search",
        "inventory_lookup",
    ]
    assert result.outcome.actions[0].observation == "NOT_FOUND"
    assert len(result.planner_traces) == 4
