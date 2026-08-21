from pathlib import Path

import pytest

from rag_practice.evaluation.training import (
    RankedDocument,
    RetrievalQuery,
    evaluate_rankings,
    select_top_non_positive,
)
from rag_practice.training.retriever_finetune import (
    MODEL_NAME,
    MODEL_REVISION,
    load_training_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "m10_training" / "dataset.json"


def test_m10_training_benchmark_is_disjoint_and_frozen_shape() -> None:
    benchmark = load_training_benchmark(BENCHMARK)
    assert MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"
    assert MODEL_REVISION == "1c82ace116a2629de82404c4be48c0e5d4cf08be"
    assert benchmark.config.seed == 23
    assert benchmark.config.epochs == 2
    assert benchmark.config.batch_size == 4
    assert benchmark.config.learning_rate == pytest.approx(2e-5)
    assert benchmark.config.temperature == pytest.approx(0.05)

    for split in (benchmark.train, benchmark.dev, benchmark.test):
        assert len(split.documents) == 8
        assert len(split.queries) == 8
        assert {query.query_class for query in split.queries} == {
            "rollback_alias",
            "status_alias",
            "shipping_alias",
            "inventory_alias",
        }

    assert set(benchmark.train.documents).isdisjoint(benchmark.dev.documents)
    assert set(benchmark.train.documents).isdisjoint(benchmark.test.documents)
    assert set(benchmark.dev.documents).isdisjoint(benchmark.test.documents)


def test_training_rank_metrics_keep_margin_separate_from_rank() -> None:
    queries = (
        RetrievalQuery("q1", "query one", "d1", "alpha"),
        RetrievalQuery("q2", "query two", "d4", "beta"),
    )
    rankings = {
        "q1": [
            RankedDocument("d1", 0.8),
            RankedDocument("d2", 0.7),
            RankedDocument("d3", 0.1),
        ],
        "q2": [
            RankedDocument("d2", 0.9),
            RankedDocument("d4", 0.85),
            RankedDocument("d1", 0.2),
        ],
    }
    result = evaluate_rankings(queries, rankings)
    assert result["all"]["recall@1"] == pytest.approx(0.5)
    assert result["all"]["recall@3"] == pytest.approx(1.0)
    assert result["all"]["mrr"] == pytest.approx(0.75)
    assert result["all"]["mean_score_margin"] == pytest.approx(0.025)
    assert result["by_class"]["alpha"]["recall@1"] == pytest.approx(1.0)
    assert result["by_class"]["beta"]["recall@1"] == pytest.approx(0.0)


def test_hard_negative_selection_excludes_positive_without_repairing_rank() -> None:
    ranking = [
        RankedDocument("positive", 0.92),
        RankedDocument("confuser", 0.89),
        RankedDocument("easy", 0.20),
    ]
    selected = select_top_non_positive(ranking, positive_document_id="positive")
    assert selected.document_id == "confuser"
    assert selected.score == pytest.approx(0.89)


def test_hard_negative_selection_requires_a_negative() -> None:
    with pytest.raises(ValueError, match="no non-positive"):
        select_top_non_positive(
            [RankedDocument("positive", 1.0)], positive_document_id="positive"
        )
