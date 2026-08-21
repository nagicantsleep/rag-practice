from __future__ import annotations

from pathlib import Path

from rag_practice.agentic import AgentState, ToolEnvironment, load_runtime_tasks
from rag_practice.agentic.multi_agent import build_critic_messages, run_role_split_agent
from rag_practice.agentic.smollm import PlannerTrace

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "m09_agentic" / "benchmark.json"


def _trace(raw: str, action: tuple[str, str] | None, *, valid: bool, stopped: bool = False) -> PlannerTrace:
    return PlannerTrace(
        raw_output=raw,
        parsed_tool=action[0] if action else None,
        parsed_argument=action[1] if action else None,
        valid=valid,
        stopped=stopped,
        prompt_tokens=10,
        output_tokens=4,
        generation_ms=1.0,
    )


def test_critic_prompt_uses_runtime_state_not_evaluator_labels() -> None:
    runtime, tasks = load_runtime_tasks(BENCHMARK)
    env = ToolEnvironment(runtime)
    task = next(task for task in tasks if task.id == "a2")
    state = AgentState(task=task)
    state.actions.append(env.call("docs_search", task.question))
    messages = build_critic_messages(task, state, "(none)")
    rendered = "\n".join(message["content"] for message in messages)

    assert task.question in rendered
    assert "SKU-A17" in rendered
    assert "(none)" in rendered
    assert "expected_answer" not in rendered
    assert "expected_actions" not in rendered
    assert "expected_evidence" not in rendered
    assert "requires_recovery" not in rendered


class _FakeProposer:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, task, state):
        self.calls += 1
        return None, _trace("(none)", None, valid=False)


class _FakeCritic:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def decide(self, task, state, proposer_output):
        action, raw, valid, stopped = self.decisions.pop(0)
        return action, _trace(raw, action, valid=valid, stopped=stopped)


def test_role_split_critic_can_supply_action_without_evaluator_labels() -> None:
    runtime, tasks = load_runtime_tasks(BENCHMARK)
    env = ToolEnvironment(runtime)
    task = next(task for task in tasks if task.id == "a6")
    proposer = _FakeProposer()
    critic = _FakeCritic(
        [
            (("inventory_lookup", "SKU-C05"), "inventory_lookup|SKU-C05", True, False),
            (None, "STOP", True, True),
        ]
    )

    result = run_role_split_agent(task, env, proposer, critic)
    assert result.outcome.answer == "31"
    assert [action.tool for action in result.outcome.actions] == ["inventory_lookup"]
    assert len(result.cycles) == 2
    assert result.cycles[0].selected_source == "critic"
    assert result.cycles[1].selected_source is None


def test_coordinator_falls_back_to_valid_proposer_when_critic_is_invalid() -> None:
    runtime, tasks = load_runtime_tasks(BENCHMARK)
    env = ToolEnvironment(runtime)
    task = next(task for task in tasks if task.id == "a7")

    class Proposer:
        def __init__(self):
            self.calls = 0

        def decide(self, task, state):
            self.calls += 1
            if self.calls == 1:
                action = ("calculator", "17 + 25")
                return action, _trace("calculator|17 + 25", action, valid=True)
            return None, _trace("STOP", None, valid=True, stopped=True)

    critic = _FakeCritic(
        [
            (None, "I agree", False, False),
            (None, "I agree", False, False),
        ]
    )
    result = run_role_split_agent(task, env, Proposer(), critic)
    assert result.outcome.answer == "42"
    assert result.cycles[0].selected_source == "proposer_fallback"
    assert result.cycles[1].selected_source is None
