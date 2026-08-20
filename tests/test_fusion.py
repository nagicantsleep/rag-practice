import pytest

from rag_practice.retrieval.fusion import (
    min_max_normalize,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)


def test_rrf_rewards_documents_supported_by_multiple_rankings():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [document_id for document_id, _ in fused[:2]] == ["a", "b"]
    assert fused[0][1] == fused[1][1]


def test_rrf_deduplicates_within_one_ranking():
    fused = reciprocal_rank_fusion([["a", "a", "b"]], k=0)
    scores = dict(fused)
    assert scores["a"] == 1.0
    assert scores["b"] == 1 / 3


def test_min_max_normalize_and_weighted_fusion():
    assert min_max_normalize({"a": 2.0, "b": 4.0}) == {"a": 0.0, "b": 1.0}
    fused = weighted_score_fusion(
        [{"a": 10.0, "b": 0.0}, {"a": 0.0, "b": 5.0}],
        [0.25, 0.75],
    )
    assert fused[0][0] == "b"
    with pytest.raises(ValueError):
        weighted_score_fusion([{"a": 1.0}], [0.5, 0.5])
