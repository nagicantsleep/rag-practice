from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse rankings with the classic RRF score sum(1 / (k + rank))."""

    if k < 0:
        raise ValueError("k must be non-negative")
    if limit is not None and limit <= 0:
        return []

    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, document_id in enumerate(ranking, start=1):
            if document_id in seen:
                continue
            scores[document_id] += 1.0 / (k + rank)
            seen.add(document_id)

    fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return fused if limit is None else fused[:limit]


def min_max_normalize(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high == low:
        return {document_id: 0.0 for document_id in scores}
    return {
        document_id: (score - low) / (high - low)
        for document_id, score in scores.items()
    }


def weighted_score_fusion(
    score_maps: Sequence[Mapping[str, float]],
    weights: Sequence[float],
    *,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Min-max normalize each system then combine with explicit weights."""

    if len(score_maps) != len(weights):
        raise ValueError("score_maps and weights must have equal length")
    if not score_maps:
        return []
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("at least one weight must be positive")
    if limit is not None and limit <= 0:
        return []

    normalized = [min_max_normalize(scores) for scores in score_maps]
    document_ids = set().union(*(scores.keys() for scores in normalized))
    fused = []
    for document_id in document_ids:
        score = sum(
            weight * scores.get(document_id, 0.0)
            for weight, scores in zip(weights, normalized, strict=True)
        ) / total_weight
        fused.append((document_id, score))
    fused.sort(key=lambda item: (-item[1], item[0]))
    return fused if limit is None else fused[:limit]
