"""Deterministic retrieval metrics used as the evaluation foundation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import fmean


def _relevant_set(relevant: Sequence[str] | set[str]) -> set[str]:
    return set(relevant)


def precision_at_k(
    ranked_document_ids: Sequence[str],
    relevant_document_ids: Sequence[str] | set[str],
    k: int,
) -> float:
    """Fraction of the top-k slots occupied by relevant documents."""

    if k <= 0:
        raise ValueError("k must be positive")
    relevant = _relevant_set(relevant_document_ids)
    hits = sum(document_id in relevant for document_id in ranked_document_ids[:k])
    return hits / k


def recall_at_k(
    ranked_document_ids: Sequence[str],
    relevant_document_ids: Sequence[str] | set[str],
    k: int,
) -> float:
    """Fraction of all relevant documents retrieved in the top k."""

    if k <= 0:
        raise ValueError("k must be positive")
    relevant = _relevant_set(relevant_document_ids)
    if not relevant:
        return 0.0
    hits = len(set(ranked_document_ids[:k]) & relevant)
    return hits / len(relevant)


def hit_rate_at_k(
    ranked_document_ids: Sequence[str],
    relevant_document_ids: Sequence[str] | set[str],
    k: int,
) -> float:
    """Return 1 when at least one relevant document appears in top k."""

    return float(recall_at_k(ranked_document_ids, relevant_document_ids, k) > 0.0)


def reciprocal_rank(
    ranked_document_ids: Sequence[str],
    relevant_document_ids: Sequence[str] | set[str],
) -> float:
    """Reciprocal rank of the first relevant result."""

    relevant = _relevant_set(relevant_document_ids)
    for rank, document_id in enumerate(ranked_document_ids, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(
    ranked_document_ids: Sequence[str],
    relevant_document_ids: Sequence[str] | set[str],
) -> float:
    """Average precision over all relevant documents for one query."""

    relevant = _relevant_set(relevant_document_ids)
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    seen: set[str] = set()
    for rank, document_id in enumerate(ranked_document_ids, start=1):
        if document_id in relevant and document_id not in seen:
            hits += 1
            precision_sum += hits / rank
            seen.add(document_id)
    return precision_sum / len(relevant)


def dcg_at_k(
    ranked_document_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    """Discounted cumulative gain using ``2**rel - 1`` gains."""

    if k <= 0:
        raise ValueError("k must be positive")

    total = 0.0
    for rank, document_id in enumerate(ranked_document_ids[:k], start=1):
        grade = relevance.get(document_id, 0.0)
        if grade <= 0.0:
            continue
        total += (2.0**grade - 1.0) / math.log2(rank + 1)
    return total


def ndcg_at_k(
    ranked_document_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    """Normalized discounted cumulative gain at k."""

    actual = dcg_at_k(ranked_document_ids, relevance, k)
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    ideal = sum(
        (2.0**grade - 1.0) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
        if grade > 0.0
    )
    if ideal == 0.0:
        return 0.0
    return actual / ideal


def mean_average_precision(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Sequence[str] | set[str]],
) -> float:
    """Mean average precision over query IDs present in qrels."""

    if not qrels:
        return 0.0
    return fmean(
        average_precision(rankings.get(query_id, ()), relevant)
        for query_id, relevant in qrels.items()
    )


def mrr(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Sequence[str] | set[str]],
) -> float:
    """Mean reciprocal rank over query IDs present in qrels."""

    if not qrels:
        return 0.0
    return fmean(
        reciprocal_rank(rankings.get(query_id, ()), relevant)
        for query_id, relevant in qrels.items()
    )


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """Aggregate common binary/graded retrieval metrics across queries."""

    if not qrels:
        return {}
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")

    binary_qrels = {
        query_id: {document_id for document_id, grade in grades.items() if grade > 0.0}
        for query_id, grades in qrels.items()
    }

    metrics: dict[str, float] = {
        "mrr": mrr(rankings, binary_qrels),
        "map": mean_average_precision(rankings, binary_qrels),
    }

    for k in ks:
        metrics[f"hit_rate@{k}"] = fmean(
            hit_rate_at_k(rankings.get(query_id, ()), relevant, k)
            for query_id, relevant in binary_qrels.items()
        )
        metrics[f"precision@{k}"] = fmean(
            precision_at_k(rankings.get(query_id, ()), relevant, k)
            for query_id, relevant in binary_qrels.items()
        )
        metrics[f"recall@{k}"] = fmean(
            recall_at_k(rankings.get(query_id, ()), relevant, k)
            for query_id, relevant in binary_qrels.items()
        )
        metrics[f"ndcg@{k}"] = fmean(
            ndcg_at_k(rankings.get(query_id, ()), qrels[query_id], k)
            for query_id in qrels
        )

    return metrics
