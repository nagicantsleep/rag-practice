from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable

from rag_practice.ir.text import tokenize


@dataclass(frozen=True)
class RankedCandidate:
    """One frozen first-stage candidate carried through M04 experiments."""

    id: str
    document_id: str
    text: str
    first_stage_score: float
    start_word: int = 0
    end_word: int = 0
    rerank_score: float | None = None

    @property
    def effective_score(self) -> float:
        return self.first_stage_score if self.rerank_score is None else self.rerank_score

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def rerank_candidates(
    candidates: Iterable[RankedCandidate],
    scorer: Callable[[RankedCandidate], float],
) -> list[RankedCandidate]:
    """Score only the supplied candidates and return a stable descending ranking.

    The function deliberately cannot retrieve new items. This makes the candidate
    set an experimental control: reranking quality is measured independently from
    first-stage recall.
    """

    rescored = [replace(candidate, rerank_score=float(scorer(candidate))) for candidate in candidates]
    rescored.sort(key=lambda item: (-item.effective_score, item.id))
    return rescored


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _normalized_relevance(candidates: list[RankedCandidate]) -> dict[str, float]:
    if not candidates:
        return {}
    scores = [candidate.effective_score for candidate in candidates]
    low = min(scores)
    high = max(scores)
    if high == low:
        return {candidate.id: 1.0 for candidate in candidates}
    return {
        candidate.id: (candidate.effective_score - low) / (high - low)
        for candidate in candidates
    }


def mmr_select(
    candidates: Iterable[RankedCandidate],
    *,
    limit: int,
    relevance_weight: float = 0.7,
    similarity: Callable[[str, str], float] = token_jaccard,
) -> list[RankedCandidate]:
    """Greedy Maximal Marginal Relevance over a frozen candidate list."""

    if limit <= 0:
        return []
    if not 0.0 <= relevance_weight <= 1.0:
        raise ValueError("relevance_weight must be between 0 and 1")

    remaining = list(candidates)
    relevance = _normalized_relevance(remaining)
    selected: list[RankedCandidate] = []

    while remaining and len(selected) < limit:
        def mmr_score(candidate: RankedCandidate) -> tuple[float, float, str]:
            redundancy = max(
                (similarity(candidate.text, prior.text) for prior in selected),
                default=0.0,
            )
            score = (
                relevance_weight * relevance[candidate.id]
                - (1.0 - relevance_weight) * redundancy
            )
            return score, relevance[candidate.id], candidate.id

        best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)

    return selected


def _source_positions(candidate: RankedCandidate) -> set[int]:
    if candidate.end_word <= candidate.start_word:
        return set(range(candidate.word_count))
    return set(range(candidate.start_word, candidate.end_word))


def pack_context(
    candidates: Iterable[RankedCandidate],
    *,
    budget_words: int,
    reject_source_overlap_above: float | None = None,
) -> list[RankedCandidate]:
    """Greedily pack ranked candidates under a word budget.

    Optional overlap rejection is computed from source spans rather than text
    similarity. It therefore exposes the cost of overlapping chunks directly.
    """

    if budget_words < 0:
        raise ValueError("budget_words must be non-negative")
    if reject_source_overlap_above is not None and not 0.0 <= reject_source_overlap_above <= 1.0:
        raise ValueError("reject_source_overlap_above must be between 0 and 1")

    selected: list[RankedCandidate] = []
    used_words = 0
    covered: dict[str, set[int]] = {}

    for candidate in candidates:
        if used_words + candidate.word_count > budget_words:
            continue

        positions = _source_positions(candidate)
        already = covered.get(candidate.document_id, set())
        if positions and reject_source_overlap_above is not None:
            overlap_fraction = len(positions & already) / len(positions)
            if overlap_fraction > reject_source_overlap_above:
                continue

        selected.append(candidate)
        used_words += candidate.word_count
        covered.setdefault(candidate.document_id, set()).update(positions)

    return selected


def context_source_utilization(candidates: Iterable[RankedCandidate]) -> float:
    """Unique source positions divided by words actually placed in context."""

    items = list(candidates)
    context_words = sum(item.word_count for item in items)
    if context_words == 0:
        return 0.0

    unique_positions: set[tuple[str, int]] = set()
    for item in items:
        unique_positions.update((item.document_id, position) for position in _source_positions(item))
    return len(unique_positions) / context_words


def source_order(candidates: Iterable[RankedCandidate]) -> list[RankedCandidate]:
    """Order selected context by source document and source position."""

    return sorted(candidates, key=lambda item: (item.document_id, item.start_word, item.end_word, item.id))


def edge_biased_order(candidates: Iterable[RankedCandidate]) -> list[RankedCandidate]:
    """Place higher-ranked evidence near context edges while preserving the set.

    Rank 1 goes first, rank 2 last, rank 3 second, rank 4 second-last, and so on.
    This is an explicit ordering experiment rather than a claim of universal
    lost-in-the-middle mitigation.
    """

    items = list(candidates)
    if len(items) <= 2:
        return items
    ordered: list[RankedCandidate | None] = [None] * len(items)
    left = 0
    right = len(items) - 1
    for index, item in enumerate(items):
        if index % 2 == 0:
            ordered[left] = item
            left += 1
        else:
            ordered[right] = item
            right -= 1
    return [item for item in ordered if item is not None]
