from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean

from .retrieval import evaluate_rankings


def complete_recall_at_k(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Sequence[str] | set[str]],
    *,
    k: int,
) -> float:
    """Fraction of queries whose entire relevant set appears in top-k.

    Standard Recall@K is averaged within each query and can hide partial success
    for multi-aspect information needs. This stricter metric returns 1 only when
    every relevant document for a query is present.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if not qrels:
        return 0.0
    return fmean(
        float(set(relevant).issubset(set(rankings.get(query_id, ())[:k])))
        for query_id, relevant in qrels.items()
    )


def evaluate_query_transform_rankings(
    rankings: Mapping[str, Sequence[str]],
    rows: Sequence[Mapping[str, object]],
    *,
    ks: Sequence[int] = (1, 3, 5),
    complete_k: int = 3,
) -> dict[str, float]:
    if not rows:
        return {}
    qrels = {
        str(row["id"]): {
            str(document_id): 1.0
            for document_id in row["relevant_document_ids"]  # type: ignore[index]
        }
        for row in rows
    }
    metrics = evaluate_rankings(rankings, qrels, ks=ks)
    binary = {query_id: set(grades) for query_id, grades in qrels.items()}
    metrics[f"complete_recall@{complete_k}"] = complete_recall_at_k(
        rankings,
        binary,
        k=complete_k,
    )
    return metrics


def query_class_breakdown(
    rankings: Mapping[str, Sequence[str]],
    rows: Sequence[Mapping[str, object]],
    *,
    ks: Sequence[int] = (1, 3, 5),
    complete_k: int = 3,
) -> dict[str, dict[str, float]]:
    """Evaluate all queries and each explicit query class independently."""

    if not rows:
        return {}
    classes = sorted({str(row["class"]) for row in rows})
    result = {
        "all": evaluate_query_transform_rankings(
            rankings,
            rows,
            ks=ks,
            complete_k=complete_k,
        )
    }
    for query_class in classes:
        selected = [row for row in rows if str(row["class"]) == query_class]
        result[query_class] = evaluate_query_transform_rankings(
            rankings,
            selected,
            ks=ks,
            complete_k=complete_k,
        )
    return result
