from __future__ import annotations

from time import perf_counter
from typing import Any

from .selection import RankedCandidate


class CrossEncoderReranker:
    """Thin Sentence Transformers CrossEncoder wrapper for frozen candidates."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        start = perf_counter()
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise RuntimeError(
                    "Install the 'pretrained' extra to use CrossEncoderReranker"
                ) from exc
            model = CrossEncoder(model_name, revision=revision, device=device)
        self.model = model
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.model_load_ms = (perf_counter() - start) * 1000

    def score(
        self,
        query: str,
        candidates: list[RankedCandidate],
    ) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        values = self.model.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(value) for value in values]

    def rerank(
        self,
        query: str,
        candidates: list[RankedCandidate],
    ) -> list[RankedCandidate]:
        scores = self.score(query, candidates)
        rescored = [
            RankedCandidate(
                id=candidate.id,
                document_id=candidate.document_id,
                text=candidate.text,
                first_stage_score=candidate.first_stage_score,
                start_word=candidate.start_word,
                end_word=candidate.end_word,
                rerank_score=score,
            )
            for candidate, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda item: (-item.effective_score, item.id))
        return rescored
