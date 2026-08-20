from __future__ import annotations

from rag_practice.adaptive.control import (
    AdaptiveRAGController,
    BridgeQueryPlanner,
    LexicalRetrievalJudge,
    RetrievalAssessment,
)
from rag_practice.adaptive.router import NaiveBayesRouteClassifier, Route


class FixedRouter:
    def __init__(self, route: Route) -> None:
        self.value = route

    def route(self, query: str) -> Route:
        del query
        return self.value


def test_naive_bayes_router_learns_separate_route_classes() -> None:
    router = NaiveBayesRouteClassifier()
    router.fit(
        [
            ("return the word ready", Route.NO_RETRIEVAL),
            ("what is two plus two", Route.NO_RETRIEVAL),
            ("which city hosts cedar", Route.SINGLE),
            ("which database does orion use", Route.SINGLE),
            ("which country contains the city where cedar is hosted", Route.ITERATIVE),
            ("who created the language used by orion scripts", Route.ITERATIVE),
        ]
    )
    assert router.route("return the word done") == Route.NO_RETRIEVAL
    assert router.route("which city hosts ember") == Route.SINGLE
    assert router.route("who created the language used by quartz scripts") == Route.ITERATIVE


def test_retrieval_judge_rejects_stale_source_without_qrels() -> None:
    judge = LexicalRetrievalJudge()
    assert (
        judge.assess(
            "What color is the current Aurora badge?",
            "STALE: Aurora badge was blue before redesign.",
        )
        == RetrievalAssessment.INCORRECT
    )
    assert (
        judge.assess(
            "What color is the current Aurora badge?",
            "CURRENT: Aurora badge is green.",
        )
        == RetrievalAssessment.CORRECT
    )


def test_bridge_planner_uses_new_entity_and_unresolved_terms() -> None:
    planner = BridgeQueryPlanner()
    follow_up = planner.plan(
        "Who created the language used by Quartz automation scripts?",
        "Quartz automation scripts are written in the Python programming language.",
    )
    assert follow_up is not None
    assert follow_up.startswith("Python ")
    assert "created" in follow_up


def test_controller_iterative_retrieval_follows_bridge_entity() -> None:
    documents = {
        "d5": "Quartz automation scripts are written in the Python programming language.",
        "d6": "Python was created by Guido van Rossum and first released in 1991.",
        "d9": "A deployment guide discusses shell automation.",
    }
    controller = AdaptiveRAGController(
        router=FixedRouter(Route.ITERATIVE),
        primary_documents=documents,
        max_iterative_steps=2,
    )
    trace = controller.run("Who created the language used by Quartz automation scripts?")
    assert trace.selected_document_ids == ("d5", "d6")
    assert trace.retrieval_calls == 2
    assert not trace.correction_triggered


def test_controller_corrects_stale_primary_with_fallback_source() -> None:
    controller = AdaptiveRAGController(
        router=FixedRouter(Route.SINGLE),
        primary_documents={"d7": "STALE: The Aurora release badge was blue before redesign."},
        fallback_documents={"f1": "CURRENT: The Aurora release badge is green after redesign."},
    )
    trace = controller.run("What color is the current Aurora release badge?")
    assert trace.correction_triggered
    assert trace.selected_document_ids == ("f1",)
    assert [step.source for step in trace.steps] == ["primary", "fallback"]
