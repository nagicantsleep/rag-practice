from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RetrievalQuery:
    id: str
    query: str
    relevant: str
    query_class: str


@dataclass(frozen=True)
class RankedDocument:
    document_id: str
    score: float


def _reciprocal_rank(relevant: str, ranking: Sequence[RankedDocument]) -> float:
    for rank, item in enumerate(ranking, start=1):
        if item.document_id == relevant:
            return 1.0 / rank
    return 0.0


def _score_margin(relevant: str, ranking: Sequence[RankedDocument]) -> float:
    relevant_score: float | None = None
    best_negative: float | None = None
    for item in ranking:
        if item.document_id == relevant:
            relevant_score = item.score
        elif best_negative is None or item.score > best_negative:
            best_negative = item.score
    if relevant_score is None:
        raise ValueError(f"relevant document {relevant!r} is absent from ranking")
    if best_negative is None:
        return 0.0
    return relevant_score - best_negative


def _aggregate(
    queries: Sequence[RetrievalQuery],
    rankings: Mapping[str, Sequence[RankedDocument]],
) -> dict[str, float]:
    if not queries:
        return {
            "recall@1": 0.0,
            "recall@3": 0.0,
            "mrr": 0.0,
            "mean_score_margin": 0.0,
        }

    recall_1 = 0.0
    recall_3 = 0.0
    reciprocal_ranks = 0.0
    margins = 0.0
    for query in queries:
        ranking = rankings[query.id]
        ids = [item.document_id for item in ranking]
        recall_1 += float(query.relevant in ids[:1])
        recall_3 += float(query.relevant in ids[:3])
        reciprocal_ranks += _reciprocal_rank(query.relevant, ranking)
        margins += _score_margin(query.relevant, ranking)

    count = float(len(queries))
    return {
        "recall@1": recall_1 / count,
        "recall@3": recall_3 / count,
        "mrr": reciprocal_ranks / count,
        "mean_score_margin": margins / count,
    }


def evaluate_rankings(
    queries: Sequence[RetrievalQuery],
    rankings: Mapping[str, Sequence[RankedDocument]],
) -> dict[str, object]:
    missing = {query.id for query in queries} - set(rankings)
    if missing:
        raise ValueError(f"missing rankings for queries: {sorted(missing)}")

    classes: dict[str, list[RetrievalQuery]] = defaultdict(list)
    for query in queries:
        classes[query.query_class].append(query)

    return {
        "all": _aggregate(queries, rankings),
        "by_class": {
            query_class: _aggregate(class_queries, rankings)
            for query_class, class_queries in sorted(classes.items())
        },
        "per_query": [
            {
                "id": query.id,
                "class": query.query_class,
                "query": query.query,
                "relevant": query.relevant,
                "rank": next(
                    (
                        rank
                        for rank, item in enumerate(rankings[query.id], start=1)
                        if item.document_id == query.relevant
                    ),
                    None,
                ),
                "score_margin": _score_margin(query.relevant, rankings[query.id]),
                "ranking": [
                    {"document_id": item.document_id, "score": item.score}
                    for item in rankings[query.id]
                ],
            }
            for query in queries
        ],
    }


def select_top_non_positive(
    ranking: Iterable[RankedDocument], *, positive_document_id: str
) -> RankedDocument:
    for item in ranking:
        if item.document_id != positive_document_id:
            return item
    raise ValueError("ranking contains no non-positive document")
