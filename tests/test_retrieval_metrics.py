import math

import pytest

from rag_practice.evaluation.retrieval import (
    average_precision,
    evaluate_rankings,
    hit_rate_at_k,
    mean_average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_binary_metrics_are_hand_computable() -> None:
    ranking = ["d3", "d2", "d1", "d4"]
    relevant = {"d1", "d3"}

    assert precision_at_k(ranking, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranking, relevant, 3) == pytest.approx(1.0)
    assert hit_rate_at_k(ranking, relevant, 1) == 1.0
    assert reciprocal_rank(ranking, relevant) == pytest.approx(1.0)
    assert average_precision(ranking, relevant) == pytest.approx((1.0 + 2 / 3) / 2)


def test_ndcg_with_graded_relevance() -> None:
    ranking = ["d2", "d1", "d3"]
    relevance = {"d1": 2.0, "d2": 1.0}

    actual = 1.0 + 3.0 / math.log2(3)
    ideal = 3.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(ranking, relevance, 3) == pytest.approx(actual / ideal)


def test_aggregate_metrics() -> None:
    rankings = {"q1": ["d1", "d2"], "q2": ["d3", "d4"]}
    qrels_binary = {"q1": {"d1"}, "q2": {"d4"}}
    qrels_graded = {"q1": {"d1": 2.0}, "q2": {"d4": 1.0}}

    assert mrr(rankings, qrels_binary) == pytest.approx((1.0 + 0.5) / 2)
    assert mean_average_precision(rankings, qrels_binary) == pytest.approx((1.0 + 0.5) / 2)

    metrics = evaluate_rankings(rankings, qrels_graded, ks=(1, 2))
    assert metrics["recall@2"] == pytest.approx(1.0)
    assert metrics["precision@1"] == pytest.approx(0.5)
