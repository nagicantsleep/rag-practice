from __future__ import annotations

from typing import Protocol

from .selection import RankedCandidate, rerank_candidates


class YesNoScorer(Protocol):
    def score_yes_no(self, prompt: str) -> float: ...


def relevance_prompt(query: str, passage: str) -> str:
    return (
        "Decide whether the passage contains evidence useful for answering the question. "
        "Answer yes or no.\n"
        f"Question: {query}\n"
        f"Passage: {passage}\n"
        "Relevant:"
    )


class PointwiseLLMReranker:
    """Pointwise instruction reranker over an already-frozen candidate set."""

    def __init__(self, scorer: YesNoScorer) -> None:
        self.scorer = scorer

    def rerank(self, query: str, candidates: list[RankedCandidate]) -> list[RankedCandidate]:
        return rerank_candidates(
            candidates,
            lambda candidate: self.scorer.score_yes_no(
                relevance_prompt(query, candidate.text)
            ),
        )
