"""Pinned SmolLM2 tool-planning control for M09."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .core import AgentOutcome, AgentState, RuntimeTask, ToolEnvironment, derive_answer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
MAX_NEW_TOKENS = 48
_ALLOWED_TOOLS = {"docs_search", "inventory_lookup", "status_lookup", "calculator"}


@dataclass(frozen=True)
class PlannerTrace:
    raw_output: str
    parsed_tool: str | None
    parsed_argument: str | None
    valid: bool
    stopped: bool
    prompt_tokens: int
    output_tokens: int
    generation_ms: float


@dataclass(frozen=True)
class ModelAgentOutcome:
    outcome: AgentOutcome
    planner_traces: tuple[PlannerTrace, ...]


def build_messages(task: RuntimeTask, state: AgentState) -> list[dict[str, str]]:
    history = []
    for index, action in enumerate(state.actions, start=1):
        history.append(
            f"{index}. {action.tool}|{action.argument} => "
            f"{action.observation} [success={str(action.success).lower()}]"
        )
    history_text = "\n".join(history) if history else "(none)"
    system = (
        "You are a tool planner. Output exactly one line and nothing else. "
        "Allowed outputs are: docs_search|<query>, inventory_lookup|<exact key>, "
        "status_lookup|<exact key>, calculator|<integer + integer>, or STOP. "
        "docs_search finds document text and aliases. inventory_lookup and "
        "status_lookup require exact keys. calculator only adds two integers. "
        "Use the question and recorded tool observations only. Never invent a tool result. "
        "If evidence is sufficient or no useful action remains, output STOP."
    )
    user = (
        f"Question:\n{task.question}\n\n"
        f"Tool history:\n{history_text}\n\n"
        "Choose the next action."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_decision(raw_output: str) -> tuple[tuple[str, str] | None, bool, bool]:
    """Strict one-line parser: returns (action, valid, stopped)."""

    stripped = raw_output.strip()
    if stripped == "STOP":
        return None, True, True
    if not stripped or "\n" in stripped or "|" not in stripped:
        return None, False, False
    tool, argument = stripped.split("|", 1)
    tool = tool.strip()
    argument = argument.strip()
    if tool not in _ALLOWED_TOOLS or not argument:
        return None, False, False
    return (tool, argument), True, False


class SmolLM2ToolPlanner:
    """CPU/float32 greedy next-action policy over explicit agent state."""

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.transformers_version = transformers.__version__
        self.torch_version = torch.__version__
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens

        started = perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            use_safetensors=True,
            dtype=torch.float32,
        )
        self.model.to("cpu")
        self.model.eval()
        self.model_load_ms = (perf_counter() - started) * 1000.0
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.parameter_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in self.model.parameters()
        )

    def decide(self, task: RuntimeTask, state: AgentState) -> tuple[tuple[str, str] | None, PlannerTrace]:
        messages = build_messages(task, state)
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
                f"planner prompt exceeds pinned context window: {prompt_tokens} + "
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
        trace = PlannerTrace(
            raw_output=raw_output,
            parsed_tool=action[0] if action else None,
            parsed_argument=action[1] if action else None,
            valid=valid,
            stopped=stopped,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            generation_ms=generation_ms,
        )
        return action, trace

    def metadata(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "device": "cpu",
            "dtype": "float32",
            "max_new_tokens": self.max_new_tokens,
            "model_max_length": int(self.tokenizer.model_max_length),
            "model_load_ms": self.model_load_ms,
            "parameter_count": self.parameter_count,
            "parameter_bytes": self.parameter_bytes,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
        }


def run_model_agent(
    task: RuntimeTask,
    env: ToolEnvironment,
    planner: SmolLM2ToolPlanner,
) -> ModelAgentOutcome:
    """Execute model-selected tools under the frozen action budget."""

    started = perf_counter()
    state = AgentState(task=task)
    traces: list[PlannerTrace] = []

    while len(state.actions) < env.contract["max_actions"]:
        action, trace = planner.decide(task, state)
        traces.append(trace)
        if not trace.valid or trace.stopped or action is None:
            break

        previous = state.actions[-1] if state.actions else None
        result = env.call(*action)
        state.actions.append(result)
        if (
            previous is not None
            and not previous.success
            and result.tool == "docs_search"
            and state.recoveries < env.contract["max_recoveries"]
        ):
            state.recoveries += 1

    answer = derive_answer(state, env.contract["abstain_token"])
    outcome = AgentOutcome(
        task_id=task.id,
        answer=answer,
        actions=tuple(state.actions),
        recoveries=state.recoveries,
        latency_ms=(perf_counter() - started) * 1000.0,
    )
    return ModelAgentOutcome(outcome=outcome, planner_traces=tuple(traces))
