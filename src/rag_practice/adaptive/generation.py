from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag_practice.adaptive.control import AdaptiveRAGController, ControlTrace
from rag_practice.adaptive.reflection import ReflectionCritic, ReflectionSignals
from rag_practice.adaptive.router import Route
from rag_practice.models.flan_t5 import GenerationWithConfidence


class ConfidenceGenerator(Protocol):
    def generate_with_confidence(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
    ) -> GenerationWithConfidence: ...


@dataclass(frozen=True)
class GenerationAttempt:
    answer: str
    confidence: float
    context_ids: tuple[str, ...]
    reflection: ReflectionSignals | None
    trigger_reason: str


@dataclass(frozen=True)
class AdaptiveAnswerTrace:
    question: str
    control: ControlTrace
    attempts: tuple[GenerationAttempt, ...]
    final_answer: str
    final_context_ids: tuple[str, ...]
    active_retrieval_calls: int
    refused: bool

    @property
    def total_retrieval_calls(self) -> int:
        return self.control.retrieval_calls + self.active_retrieval_calls


def _rag_prompt(question: str, contexts: list[tuple[str, str]]) -> str:
    joined = "\n\n".join(f"[{document_id}] {text}" for document_id, text in contexts)
    return (
        "Answer the question using only the supplied context. "
        "If the context does not contain the answer, say exactly: I do not know. "
        "Keep the answer short.\n"
        f"Question: {question}\n"
        f"Context:\n{joined}\n"
        "Answer:"
    )


def _direct_prompt(question: str) -> str:
    return (
        "Answer the user request directly and obey any requested output format.\n"
        f"Question: {question}\n"
        "Answer:"
    )


class AdaptiveGenerationPipeline:
    """Generation layer for adaptive/active/corrective RAG.

    Routing and CRAG-style correction happen in ``AdaptiveRAGController``. This
    layer adds a FLARE-style confidence-triggered extra retrieval attempt and a
    Self-RAG-style relevance/support/utility reflection before accepting output.
    It never receives qrels or answer references.
    """

    def __init__(
        self,
        *,
        controller: AdaptiveRAGController,
        generator: ConfidenceGenerator,
        critic: ReflectionCritic | None = None,
        utility_threshold: float = 0.65,
        max_new_tokens: int = 32,
    ) -> None:
        if not 0.0 <= utility_threshold <= 1.0:
            raise ValueError("utility_threshold must be in [0, 1]")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.controller = controller
        self.generator = generator
        self.critic = critic or ReflectionCritic()
        self.utility_threshold = utility_threshold
        self.max_new_tokens = max_new_tokens

    def _document(self, document_id: str) -> str:
        if document_id in self.controller.primary_documents:
            return self.controller.primary_documents[document_id]
        return self.controller.fallback_documents[document_id]

    def _contexts(self, document_ids: list[str]) -> list[tuple[str, str]]:
        return [(document_id, self._document(document_id)) for document_id in document_ids]

    def _active_search(self, query: str, excluded: set[str]) -> str | None:
        # Search deeper than top-1 so the active step can find a new passage rather
        # than simply returning the already selected context.
        depth = min(len(self.controller.primary_documents), max(3, len(excluded) + 2))
        for document_id, _ in self.controller.primary.search(query, k=depth):
            if document_id not in excluded:
                return document_id
        if self.controller.fallback is not None:
            depth = min(len(self.controller.fallback_documents), max(2, len(excluded) + 1))
            for document_id, _ in self.controller.fallback.search(query, k=depth):
                if document_id not in excluded:
                    return document_id
        return None

    def _generate_with_context(
        self,
        question: str,
        context_ids: list[str],
        *,
        trigger_reason: str,
    ) -> GenerationAttempt:
        contexts = self._contexts(context_ids)
        generated = self.generator.generate_with_confidence(
            _rag_prompt(question, contexts),
            max_new_tokens=self.max_new_tokens,
        )
        reflection = self.critic.reflect(
            question=question,
            answer=generated.text,
            contexts=[text for _, text in contexts],
            generation_confidence=generated.confidence,
        )
        return GenerationAttempt(
            answer=generated.text,
            confidence=generated.confidence,
            context_ids=tuple(context_ids),
            reflection=reflection,
            trigger_reason=trigger_reason,
        )

    def run(self, question: str, *, enable_active_reflection: bool = True) -> AdaptiveAnswerTrace:
        control = self.controller.run(question)
        if control.route == Route.NO_RETRIEVAL:
            generated = self.generator.generate_with_confidence(
                _direct_prompt(question),
                max_new_tokens=self.max_new_tokens,
            )
            attempt = GenerationAttempt(
                answer=generated.text,
                confidence=generated.confidence,
                context_ids=(),
                reflection=None,
                trigger_reason="direct_no_retrieval",
            )
            return AdaptiveAnswerTrace(
                question=question,
                control=control,
                attempts=(attempt,),
                final_answer=generated.text,
                final_context_ids=(),
                active_retrieval_calls=0,
                refused=False,
            )

        context_ids = list(control.selected_document_ids)
        first = self._generate_with_context(question, context_ids, trigger_reason="initial")
        attempts = [first]
        active_calls = 0
        final = first

        signals = first.reflection
        needs_more = bool(
            enable_active_reflection
            and signals is not None
            and (
                signals.retrieve
                or not signals.relevant
                or not signals.supported
                or signals.utility < self.utility_threshold
            )
        )
        if needs_more:
            active_query = f"{question} {first.answer}".strip()
            extra_id = self._active_search(active_query, set(context_ids))
            if extra_id is not None:
                active_calls += 1
                context_ids.append(extra_id)
                final = self._generate_with_context(
                    question,
                    context_ids,
                    trigger_reason="low_confidence_or_reflection_retry",
                )
                attempts.append(final)

        final_signals = final.reflection
        refused = bool(
            enable_active_reflection
            and final_signals is not None
            and (
                not final_signals.supported
                or final_signals.utility < self.utility_threshold
            )
        )
        final_answer = "I do not know." if refused else final.answer
        return AdaptiveAnswerTrace(
            question=question,
            control=control,
            attempts=tuple(attempts),
            final_answer=final_answer,
            final_context_ids=tuple(context_ids),
            active_retrieval_calls=active_calls,
            refused=refused,
        )
