import math

import pytest

from rag_practice.ir.vector import cosine_similarity, sparse_cosine_similarity


def test_dense_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1 / math.sqrt(2))


def test_dense_cosine_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])


def test_sparse_cosine_similarity() -> None:
    assert sparse_cosine_similarity({"a": 1.0}, {"a": 2.0}) == pytest.approx(1.0)
    assert sparse_cosine_similarity({"a": 1.0}, {"b": 2.0}) == pytest.approx(0.0)
