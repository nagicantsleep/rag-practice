from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from rag_practice.ir.text import tokenize


@dataclass(frozen=True)
class ReflectionSignals:
    retrieve: bool
    relevant: bool
    supported: bool
    utility: float


class ActiveRetrievalPolicy:
    """FLARE-style trigger primitive over an explicit model confidence signal."""

    def __init__(self, *, confidence_threshold: float = 0.6) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.confidence_threshold = confidence_threshold

    def should_retrieve(self, confidence: float) -> bool:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return confidence < self.confidence_threshold


class ReflectionCritic:
    """Inspectable Self-RAG-style relevance/support/utility critic.

    This is a mechanism implementation, not the trained reflection-token model from
    Self-RAG. It exposes the same control questions with deterministic lexical
    semantics so retrieval/control failures can be unit-tested independently.
    """

    def __init__(
        self,
        *,
        relevance_threshold: float = 0.15,
        support_threshold: float = 0.8,
        active_policy: ActiveRetrievalPolicy | None = None,
    ) -> None:
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be in [0, 1]")
        if not 0.0 <= support_threshold <= 1.0:
            raise ValueError("support_threshold must be in [0, 1]")
        self.relevance_threshold = relevance_threshold
        self.support_threshold = support_threshold
        self.active_policy = active_policy or ActiveRetrievalPolicy()

    def relevance_score(self, question: str, context: str) -> float:
        query_terms = set(tokenize(question))
        context_terms = set(tokenize(context))
        if not query_terms:
            return 0.0
        return len(query_terms & context_terms) / len(query_terms)

    def support_score(self, answer: str, contexts: list[str]) -> float:
        answer_terms = tokenize(answer)
        if not answer_terms:
            return 0.0
        context_terms = set(tokenize(" ".join(contexts)))
        return sum(term in context_terms for term in answer_terms) / len(answer_terms)

    def reflect(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        generation_confidence: float,
    ) -> ReflectionSignals:
        relevant_scores = [self.relevance_score(question, context) for context in contexts]
        mean_relevance = fmean(relevant_scores) if relevant_scores else 0.0
        support = self.support_score(answer, contexts)
        relevant = mean_relevance >= self.relevance_threshold
        supported = support >= self.support_threshold
        utility = (float(relevant) + float(supported) + support) / 3.0
        return ReflectionSignals(
            retrieve=self.active_policy.should_retrieve(generation_confidence),
            relevant=relevant,
            supported=supported,
            utility=utility,
        )
