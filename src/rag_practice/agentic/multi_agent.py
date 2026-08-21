"""Exploratory shared-checkpoint role-split agent control for M09."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .core import AgentOutcome, AgentState, RuntimeTask, ToolEnvironment, derive_answer
from .smollm import PlannerTrace, SmolLM2ToolPlanner, parse_decision


@dataclass(frozen=True)
class RoleCycleTrace:
    proposer: PlannerTrace
    critic: PlannerTrace
    selected_source: str | None
    selected_tool: str | None
    selected_argument: str | None


@dataclass(frozen=True)
class MultiAgentOutcome:
    outcome: AgentOutcome
    cycles: tuple[RoleCycleTrace, ...]


def _history_text(state: AgentState) -> str:
    if not state.actions:
        return "(none)"
    return "\n".join(
        f"{index}. {action.tool}|{action.argument} => "
        f"{action.observation} [success={str(action.success).lower()}]"
        for index, action in enumerate(state.actions, start=1)
    )


def build_critic_messages(
    task: RuntimeTask,
    state: AgentState,
    proposer_output: str,
) -> list[dict[str, str]]:
    """Build the post-single-agent verifier/corrector role prompt."""

    system = (
        "You are the verifier/corrector in a role-split tool-using agent. "
        "A proposer produced a candidate next action that may be wrong or malformed. "
        "Independently choose the best next action using only the question, recorded tool "
        "history, and the proposer text. Output exactly one line and nothing else. "
        "Allowed outputs are: docs_search|<query>, inventory_lookup|<exact key>, "
        "status_lookup|<exact key>, calculator|<integer + integer>, or STOP. "
        "docs_search discovers document evidence and aliases. inventory_lookup and "
        "status_lookup require exact keys. calculator only adds two integers. "
        "If evidence is sufficient or no useful action remains, output STOP. "
        "Do not explain, number, quote, or wrap the action."
    )
    user = (
        f"Question:\n{task.question}\n\n"
        f"Tool history:\n{_history_text(state)}\n\n"
        f"Proposer output:\n{proposer_output}\n\n"
        "Return the verified/corrected next action."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class SmolLM2ActionCritic:
    """Second role sharing the exact pinned model weights with the proposer."""

    def __init__(self, proposer: SmolLM2ToolPlanner) -> None:
        self.proposer = proposer
        self.torch = proposer.torch
        self.tokenizer = proposer.tokenizer
        self.model = proposer.model
        self.max_new_tokens = proposer.max_new_tokens

    def decide(
        self,
        task: RuntimeTask,
        state: AgentState,
        proposer_output: str,
    ) -> tuple[tuple[str, str] | None, PlannerTrace]:
        messages = build_critic_messages(task, state, proposer_output)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {name: value.to("cpu") for name, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        if prompt_tokens + self.max_new_tokens > int(self.tokenizer.model_max_length):
            raise ValueError(
                f"critic prompt exceeds pinned context window: {prompt_tokens} + "
                f"{self.max_new_tokens} > {self.tokenizer.model_max_length}"
            )

        started = perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generation_ms = (perf_counter() - started) * 1000.0
        generated = outputs[0, prompt_tokens:]
        output_tokens = int(generated.shape[-1])
        raw_output = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        action, valid, stopped = parse_decision(raw_output)
        return action, PlannerTrace(
            raw_output=raw_output,
            parsed_tool=action[0] if action else None,
            parsed_argument=action[1] if action else None,
            valid=valid,
            stopped=stopped,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            generation_ms=generation_ms,
        )


def run_role_split_agent(
    task: RuntimeTask,
    env: ToolEnvironment,
    proposer: SmolLM2ToolPlanner,
    critic: SmolLM2ActionCritic,
) -> MultiAgentOutcome:
    """Run proposer -> critic/coordinator cycles under the original action budget."""

    started = perf_counter()
    state = AgentState(task=task)
    cycles: list[RoleCycleTrace] = []

    while len(state.actions) < env.contract["max_actions"]:
        proposed_action, proposer_trace = proposer.decide(task, state)
        critic_action, critic_trace = critic.decide(task, state, proposer_trace.raw_output)

        selected_action: tuple[str, str] | None = None
        selected_source: str | None = None
        if critic_trace.valid:
            if not critic_trace.stopped:
                selected_action = critic_action
                selected_source = "critic"
        elif proposer_trace.valid and not proposer_trace.stopped:
            selected_action = proposed_action
            selected_source = "proposer_fallback"

        cycles.append(
            RoleCycleTrace(
                proposer=proposer_trace,
                critic=critic_trace,
                selected_source=selected_source,
                selected_tool=selected_action[0] if selected_action else None,
                selected_argument=selected_action[1] if selected_action else None,
            )
        )
        if selected_action is None:
            break

        previous = state.actions[-1] if state.actions else None
        result = env.call(*selected_action)
        state.actions.append(result)
        if (
            previous is not None
            and not previous.success
            and result.tool == "docs_search"
            and state.recoveries < env.contract["max_recoveries"]
        ):
            state.recoveries += 1

    answer = derive_answer(state, env.contract["abstain_token"])
    return MultiAgentOutcome(
        outcome=AgentOutcome(
            task_id=task.id,
            answer=answer,
            actions=tuple(state.actions),
            recoveries=state.recoveries,
            latency_ms=(perf_counter() - started) * 1000.0,
        ),
        cycles=tuple(cycles),
    )
