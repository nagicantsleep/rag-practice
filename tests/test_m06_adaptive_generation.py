from __future__ import annotations

from rag_practice.adaptive.control import AdaptiveRAGController
from rag_practice.adaptive.generation import AdaptiveGenerationPipeline
from rag_practice.adaptive.reflection import ActiveRetrievalPolicy, ReflectionCritic
from rag_practice.adaptive.router import Route
from rag_practice.models.flan_t5 import GenerationWithConfidence


class FixedRouter:
    def __init__(self, route: Route) -> None:
        self.value = route

    def route(self, query: str) -> Route:
        del query
        return self.value


class FakeGenerator:
    def __init__(self, outputs: list[GenerationWithConfidence]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate_with_confidence(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
    ) -> GenerationWithConfidence:
        del max_new_tokens
        self.prompts.append(prompt)
        return self.outputs.pop(0)


def critic(threshold: float = 0.6) -> ReflectionCritic:
    return ReflectionCritic(
        relevance_threshold=0.1,
        support_threshold=0.8,
        active_policy=ActiveRetrievalPolicy(confidence_threshold=threshold),
    )


def test_no_retrieval_route_never_adds_active_retrieval() -> None:
    controller = AdaptiveRAGController(
        router=FixedRouter(Route.NO_RETRIEVAL),
        primary_documents={"d1": "Irrelevant stored evidence."},
    )
    generator = FakeGenerator([GenerationWithConfidence("READY", 0.2)])
    pipeline = AdaptiveGenerationPipeline(controller=controller, generator=generator, critic=critic())

    trace = pipeline.run("Return the word READY exactly.")

    assert trace.final_answer == "READY"
    assert trace.total_retrieval_calls == 0
    assert trace.active_retrieval_calls == 0
    assert len(trace.attempts) == 1
    assert trace.attempts[0].reflection is None


def test_low_confidence_answer_triggers_one_active_retrieval_and_retry() -> None:
    controller = AdaptiveRAGController(
        router=FixedRouter(Route.SINGLE),
        primary_documents={
            "d1": "Project Ember is hosted in Oslo for its production deployment.",
            "d2": "Oslo is a city in Norway and serves as the national capital.",
        },
    )
    generator = FakeGenerator(
        [
            GenerationWithConfidence("Oslo", 0.2),
            GenerationWithConfidence("Oslo", 0.9),
        ]
    )
    pipeline = AdaptiveGenerationPipeline(controller=controller, generator=generator, critic=critic())

    trace = pipeline.run("Which city hosts Project Ember?")

    assert trace.control.selected_document_ids == ("d1",)
    assert trace.active_retrieval_calls == 1
    assert trace.total_retrieval_calls == 2
    assert trace.final_context_ids == ("d1", "d2")
    assert len(trace.attempts) == 2
    assert trace.final_answer == "Oslo"
    assert not trace.refused


def test_unsupported_answer_is_refused_when_no_new_evidence_exists() -> None:
    controller = AdaptiveRAGController(
        router=FixedRouter(Route.SINGLE),
        primary_documents={"d3": "Vega database replicas use the Raft consensus protocol."},
    )
    generator = FakeGenerator([GenerationWithConfidence("Paxos", 0.9)])
    pipeline = AdaptiveGenerationPipeline(controller=controller, generator=generator, critic=critic())

    trace = pipeline.run("What consensus protocol does Vega use?")

    assert trace.total_retrieval_calls == 1
    assert trace.active_retrieval_calls == 0
    assert trace.refused
    assert trace.final_answer == "I do not know."
