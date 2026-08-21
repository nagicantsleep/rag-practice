from pathlib import Path

import pytest

from rag_practice.evaluation.training import RankedDocument, RetrievalQuery
from rag_practice.training.linear_reranker import (
    FeatureRow,
    LinearPairwiseReranker,
    build_candidate_features,
    load_reranker_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "benchmarks" / "m10_training" / "reranker_contract.json"


def test_reranker_contract_is_frozen() -> None:
    config = load_reranker_contract(CONTRACT)
    assert config.seed == 29
    assert config.epochs == 80
    assert config.learning_rate == pytest.approx(0.05)
    assert config.weight_decay == 0.0
    assert config.candidate_k == 3


def test_candidate_features_keep_first_stage_rank_explicit() -> None:
    query = RetrievalQuery("q", "which Alpha guide has stock key", "d1", "inventory")
    documents = {
        "d1": "Alpha inventory guide contains the inventory code",
        "d2": "Alpha shipping policy contains the shipping SLA",
        "d3": "Beta inventory guide contains another code",
    }
    ranking = [
        RankedDocument("d2", 0.8),
        RankedDocument("d1", 0.7),
        RankedDocument("d3", 0.4),
    ]
    rows = build_candidate_features(
        query=query,
        documents=documents,
        baseline_ranking=ranking,
        candidate_k=3,
    )
    assert rows["d2"].reciprocal_rank == pytest.approx(1.0)
    assert rows["d1"].reciprocal_rank == pytest.approx(0.5)
    assert rows["d3"].reciprocal_rank == pytest.approx(1 / 3)
    assert rows["d1"].overlap_fraction > 0


def test_linear_reranker_starts_neutral() -> None:
    reranker = LinearPairwiseReranker(load_reranker_contract(CONTRACT))
    rows = [
        FeatureRow(0.9, 2.0, 0.5, 1.0),
        FeatureRow(0.3, 0.0, 0.1, 0.5),
    ]
    assert reranker.score_rows(rows) == pytest.approx([0.0, 0.0])
    payload = reranker.parameters_payload()
    assert payload["parameter_count"] == 5
    assert payload["parameter_bytes_float32"] == 20
