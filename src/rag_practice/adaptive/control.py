from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from rag_practice.adaptive.router import Route
from rag_practice.ir.bm25 import BM25Index
from rag_practice.ir.text import tokenize


class Router(Protocol):
    def route(self, query: str) -> Route: ...


class RetrievalAssessment(str, Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass(frozen=True)
class RetrievalStep:
    query: str
    source: str
    document_id: str
    score: float
    assessment: RetrievalAssessment


@dataclass(frozen=True)
class ControlTrace:
    question: str
    route: Route
    steps: tuple[RetrievalStep, ...]
    selected_document_ids: tuple[str, ...]
    correction_triggered: bool

    @property
    def retrieval_calls(self) -> int:
        return len(self.steps)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "by",
    "does",
    "for",
    "how",
    "in",
    "is",
    "of",
    "the",
    "to",
    "what",
    "where",
    "which",
    "who",
}
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9-]+\b")
_ENTITY_STOP = {"The", "What", "Which", "Who", "Where", "How", "Project", "CURRENT", "STALE"}


class LexicalRetrievalJudge:
    """Transparent CRAG-style retrieval-quality gate.

    `STALE:` is an explicit source-quality signal in the controlled benchmark.
    Otherwise the evaluator uses lexical coverage only; it never receives qrels.
    """

    def __init__(self, *, minimum_overlap: float = 0.12) -> None:
        if not 0.0 <= minimum_overlap <= 1.0:
            raise ValueError("minimum_overlap must be in [0, 1]")
        self.minimum_overlap = minimum_overlap

    def assess(self, query: str, text: str) -> RetrievalAssessment:
        if text.lstrip().upper().startswith("STALE:"):
            return RetrievalAssessment.INCORRECT
        query_terms = {token for token in tokenize(query) if token not in _STOPWORDS}
        text_terms = set(tokenize(text))
        if not query_terms:
            return RetrievalAssessment.AMBIGUOUS
        overlap = len(query_terms & text_terms) / len(query_terms)
        if overlap < self.minimum_overlap:
            return RetrievalAssessment.AMBIGUOUS
        return RetrievalAssessment.CORRECT


class BridgeQueryPlanner:
    """Plan a second-hop query from a newly discovered entity + unresolved terms."""

    def __init__(self, *, max_unresolved_terms: int = 4) -> None:
        if max_unresolved_terms <= 0:
            raise ValueError("max_unresolved_terms must be positive")
        self.max_unresolved_terms = max_unresolved_terms

    def plan(self, question: str, evidence_text: str) -> str | None:
        query_tokens = set(tokenize(question))
        evidence_tokens = set(tokenize(evidence_text))
        entities = [
            entity
            for entity in _ENTITY_RE.findall(evidence_text)
            if entity not in _ENTITY_STOP and entity.lower() not in query_tokens
        ]
        if not entities:
            return None

        unresolved: list[str] = []
        seen: set[str] = set()
        for token in tokenize(question):
            if token in _STOPWORDS or token in evidence_tokens or token in seen:
                continue
            unresolved.append(token)
            seen.add(token)
            if len(unresolved) >= self.max_unresolved_terms:
                break
        suffix = " ".join(unresolved)
        return f"{entities[0]} {suffix}".strip()


class AdaptiveRAGController:
    """Explicit no-RAG/single/iterative routing with CRAG-style correction."""

    def __init__(
        self,
        *,
        router: Router,
        primary_documents: dict[str, str],
        fallback_documents: dict[str, str] | None = None,
        judge: LexicalRetrievalJudge | None = None,
        planner: BridgeQueryPlanner | None = None,
        max_iterative_steps: int = 2,
    ) -> None:
        if max_iterative_steps <= 0:
            raise ValueError("max_iterative_steps must be positive")
        self.router = router
        self.primary_documents = dict(primary_documents)
        self.fallback_documents = dict(fallback_documents or {})
        self.primary = BM25Index(self.primary_documents)
        self.fallback = BM25Index(self.fallback_documents) if self.fallback_documents else None
        self.judge = judge or LexicalRetrievalJudge()
        self.planner = planner or BridgeQueryPlanner()
        self.max_iterative_steps = max_iterative_steps

    def _search(self, query: str, *, source: str) -> RetrievalStep | None:
        if source == "primary":
            index = self.primary
            documents = self.primary_documents
        elif source == "fallback" and self.fallback is not None:
            index = self.fallback
            documents = self.fallback_documents
        else:
            return None
        results = index.search(query, k=1)
        if not results:
            return None
        document_id, score = results[0]
        return RetrievalStep(
            query=query,
            source=source,
            document_id=document_id,
            score=score,
            assessment=self.judge.assess(query, documents[document_id]),
        )

    def _correct_if_needed(self, question: str, step: RetrievalStep) -> tuple[list[RetrievalStep], str]:
        if step.assessment == RetrievalAssessment.CORRECT or self.fallback is None:
            return [step], step.document_id
        fallback_step = self._search(question, source="fallback")
        if fallback_step is None:
            return [step], step.document_id
        return [step, fallback_step], fallback_step.document_id

    def run(self, question: str) -> ControlTrace:
        route = self.router.route(question)
        if route == Route.NO_RETRIEVAL:
            return ControlTrace(question, route, (), (), False)

        first = self._search(question, source="primary")
        if first is None:
            return ControlTrace(question, route, (), (), False)
        steps, first_selected = self._correct_if_needed(question, first)
        correction_triggered = len(steps) > 1
        selected: list[str] = [first_selected]

        if route == Route.ITERATIVE:
            current_id = first_selected
            current_source = steps[-1].source
            for _ in range(1, self.max_iterative_steps):
                documents = self.fallback_documents if current_source == "fallback" else self.primary_documents
                current_text = documents[current_id]
                follow_up = self.planner.plan(question, current_text)
                if not follow_up:
                    break
                next_step = self._search(follow_up, source="primary")
                if next_step is None:
                    break
                next_steps, next_selected = self._correct_if_needed(follow_up, next_step)
                steps.extend(next_steps)
                correction_triggered = correction_triggered or len(next_steps) > 1
                if next_selected not in selected:
                    selected.append(next_selected)
                else:
                    break
                current_id = next_selected
                current_source = next_steps[-1].source

        return ControlTrace(
            question=question,
            route=route,
            steps=tuple(steps),
            selected_document_ids=tuple(selected),
            correction_triggered=correction_triggered,
        )
