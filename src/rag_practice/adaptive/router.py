from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from enum import Enum

from rag_practice.ir.text import tokenize


class Route(str, Enum):
    NO_RETRIEVAL = "no_retrieval"
    SINGLE = "single"
    ITERATIVE = "iterative"


def _features(text: str) -> list[str]:
    tokens = tokenize(text)
    features = list(tokens)
    features.extend(f"{left}__{right}" for left, right in zip(tokens, tokens[1:]))
    return features


class AlwaysSingleRouter:
    def route(self, query: str) -> Route:
        del query
        return Route.SINGLE


class KeywordRouter:
    """Transparent heuristic baseline before the learned complexity router."""

    def route(self, query: str) -> Route:
        lowered = query.lower()
        no_retrieval_markers = (
            "return the word",
            "answer only the number",
            "uppercase",
            "repeat the token",
            " plus ",
            " minus ",
        )
        iterative_markers = (
            "city where",
            "language used by",
            "database behind",
            "data store used by",
            "country where",
            "company that",
            "article linked from",
        )
        if any(marker in lowered for marker in no_retrieval_markers):
            return Route.NO_RETRIEVAL
        if any(marker in lowered for marker in iterative_markers):
            return Route.ITERATIVE
        return Route.SINGLE


class NaiveBayesRouteClassifier:
    """Small from-scratch multinomial NB classifier over uni/bi-gram features.

    M06 uses this instead of a framework classifier so priors, likelihoods, and
    failure modes stay visible. Training examples must come from a separate
    split from the held-out control benchmark.
    """

    def __init__(self, *, alpha: float = 1.0) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self._class_docs: Counter[Route] = Counter()
        self._class_tokens: dict[Route, Counter[str]] = defaultdict(Counter)
        self._class_token_totals: Counter[Route] = Counter()
        self._vocabulary: set[str] = set()
        self._fitted = False

    def fit(self, examples: Iterable[tuple[str, Route | str]]) -> None:
        self._class_docs.clear()
        self._class_tokens.clear()
        self._class_token_totals.clear()
        self._vocabulary.clear()

        count = 0
        for query, raw_route in examples:
            route = raw_route if isinstance(raw_route, Route) else Route(raw_route)
            features = _features(query)
            self._class_docs[route] += 1
            self._class_tokens[route].update(features)
            self._class_token_totals[route] += len(features)
            self._vocabulary.update(features)
            count += 1

        missing = set(Route) - set(self._class_docs)
        if count == 0 or missing:
            raise ValueError(f"training examples must cover every route; missing={sorted(item.value for item in missing)}")
        self._fitted = True

    def log_scores(self, query: str) -> dict[Route, float]:
        if not self._fitted:
            raise RuntimeError("fit must be called before route")
        total_docs = sum(self._class_docs.values())
        vocab_size = max(1, len(self._vocabulary))
        feature_counts = Counter(_features(query))
        scores: dict[Route, float] = {}
        for route in Route:
            prior = math.log(self._class_docs[route] / total_docs)
            denominator = self._class_token_totals[route] + self.alpha * vocab_size
            score = prior
            for feature, frequency in feature_counts.items():
                numerator = self._class_tokens[route][feature] + self.alpha
                score += frequency * math.log(numerator / denominator)
            scores[route] = score
        return scores

    def route(self, query: str) -> Route:
        scores = self.log_scores(query)
        return max(Route, key=lambda route: (scores[route], -list(Route).index(route)))
