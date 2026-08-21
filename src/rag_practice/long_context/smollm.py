"""Pinned SmolLM2 reader control for M08.7."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
MAX_NEW_TOKENS = 32


@dataclass(frozen=True)
class ReaderTrace:
    prompt_tokens: int
    output_tokens: int
    generation_ms: float


def build_messages(question: str, texts: Sequence[str]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[Section {index}]\n{text}" for index, text in enumerate(texts, start=1)
    )
    system = (
        "Answer the question using only the supplied context. "
        "If the context does not contain enough evidence to answer, output exactly ABSTAIN. "
        "Return only the final answer, with no explanation. "
        "For list questions, separate items with semicolons."
    )
    user = f"Question:\n{question}\n\nContext:\n{context}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class SmolLM2ContextReader:
    """CPU/float32 deterministic reader over already-selected context."""

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

        start = perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            use_safetensors=True,
            dtype=torch.float32,
        )
        self.model.to("cpu")
        self.model.eval()
        self.model_load_ms = (perf_counter() - start) * 1000.0
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.parameter_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in self.model.parameters()
        )
        self.last_trace: ReaderTrace | None = None

    def answer(self, question: str, texts: Sequence[str]) -> str:
        messages = build_messages(question, texts)
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
                f"prompt exceeds pinned context window: {prompt_tokens} + "
                f"{self.max_new_tokens} > {self.tokenizer.model_max_length}"
            )

        start = perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generation_ms = (perf_counter() - start) * 1000.0
        generated = outputs[0, prompt_tokens:]
        output_tokens = int(generated.shape[-1])
        answer = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        self.last_trace = ReaderTrace(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            generation_ms=generation_ms,
        )
        return answer

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
