from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class GenerationWithConfidence:
    text: str
    confidence: float


class FlanT5Backend:
    """Small text-to-text backend shared by M04+ generation and LLM scoring."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        start = perf_counter()
        if tokenizer is None or model is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise RuntimeError("Install the 'pretrained' extra to use FlanT5Backend") from exc
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)

        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        self.revision = revision
        self.device = device
        if hasattr(self.model, "to"):
            self.model.to(device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.model_load_ms = (perf_counter() - start) * 1000

    def _encode(self, text: str):
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }

    def generate(self, prompt: str, *, max_new_tokens: int = 64) -> str:
        encoded = self._encode(prompt)
        output = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def generate_with_confidence(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
    ) -> GenerationWithConfidence:
        """Greedy generation plus geometric-mean selected-token probability.

        The confidence is intentionally simple and inspectable. It is not calibrated
        probability of factual correctness; M06 uses it only as a FLARE-style signal
        for deciding whether another retrieval step may be warranted.
        """

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("PyTorch is required for generation confidence") from exc

        encoded = self._encode(prompt)
        output = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        sequence = output.sequences[0]
        text = self.tokenizer.decode(sequence, skip_special_tokens=True).strip()
        scores = list(output.scores or ())
        if not scores:
            return GenerationWithConfidence(text=text, confidence=1.0)

        generated_ids = sequence[-len(scores):]
        log_probabilities: list[float] = []
        for logits, token_id in zip(scores, generated_ids):
            probability = torch.softmax(logits[0], dim=-1)[int(token_id)].item()
            log_probabilities.append(math.log(max(float(probability), 1e-12)))
        confidence = math.exp(sum(log_probabilities) / len(log_probabilities))
        return GenerationWithConfidence(text=text, confidence=max(0.0, min(1.0, confidence)))

    def score_yes_no(self, prompt: str) -> float:
        """Return a continuous yes-vs-no relevance logit margin."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("PyTorch is required for FLAN relevance scoring") from exc

        encoded = self._encode(prompt)
        start_id = getattr(self.model.config, "decoder_start_token_id", None)
        if start_id is None:
            start_id = getattr(self.tokenizer, "pad_token_id", 0)
        decoder_input_ids = torch.tensor([[start_id]], device=self.device)
        with torch.no_grad():
            logits = self.model(
                **encoded,
                decoder_input_ids=decoder_input_ids,
            ).logits[0, 0]

        yes_ids = self.tokenizer.encode("yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode("no", add_special_tokens=False)
        if not yes_ids or not no_ids:
            raise RuntimeError("Tokenizer must encode yes and no")
        return float((logits[yes_ids[0]] - logits[no_ids[0]]).item())
