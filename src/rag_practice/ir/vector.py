"""Small vector helpers implemented without numerical libraries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

NumericVector = Sequence[float]
SparseVector = Mapping[str, float]


def cosine_similarity(a: NumericVector, b: NumericVector) -> float:
    """Return cosine similarity between two dense vectors.

    Zero vectors have similarity 0.0. A dimension mismatch is an error rather
    than being silently truncated.
    """

    if len(a) != len(b):
        raise ValueError("vectors must have the same dimensionality")

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def sparse_cosine_similarity(a: SparseVector, b: SparseVector) -> float:
    """Return cosine similarity between sparse vectors keyed by term."""

    if len(a) > len(b):
        a, b = b, a

    dot = sum(value * b.get(term, 0.0) for term, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
