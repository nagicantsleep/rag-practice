from __future__ import annotations

from time import perf_counter
from typing import Any


class FlanT5Backend:
    """Small text-to-text backend shared by M04 generation and LLM reranking."""

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
