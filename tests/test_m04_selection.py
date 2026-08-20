from rag_practice.reranking import (
    RankedCandidate,
    context_source_utilization,
    mmr_select,
    pack_context,
    rerank_candidates,
)


def candidate(
    identifier: str,
    text: str,
    score: float,
    *,
    document_id: str = "d1",
    start: int = 0,
    end: int = 0,
) -> RankedCandidate:
    return RankedCandidate(
        id=identifier,
        document_id=document_id,
        text=text,
        first_stage_score=score,
        start_word=start,
        end_word=end,
    )


def test_rerank_only_reorders_frozen_candidates():
    items = [
        candidate("a", "alpha", 3.0),
        candidate("b", "beta", 2.0),
        candidate("c", "gamma", 1.0),
    ]
    reranked = rerank_candidates(items, lambda item: {"a": 0.1, "b": 0.9, "c": 0.2}[item.id])

    assert [item.id for item in reranked] == ["b", "c", "a"]
    assert {item.id for item in reranked} == {item.id for item in items}
    assert [item.first_stage_score for item in items] == [3.0, 2.0, 1.0]


def test_mmr_can_trade_redundancy_for_distinct_evidence():
    items = [
        candidate("a", "reranking latency grows with candidate count", 1.0),
        candidate("b", "reranking latency grows when candidate count increases", 0.95),
        candidate("c", "batching reduces service cost under tight latency", 0.8),
    ]

    selected = mmr_select(items, limit=2, relevance_weight=0.25)

    assert selected[0].id == "a"
    assert selected[1].id == "c"


def test_pack_context_respects_budget_and_source_overlap_threshold():
    items = [
        candidate("a", "one two three four", 1.0, start=0, end=4),
        candidate("b", "three four five six", 0.9, start=2, end=6),
        candidate("c", "seven eight", 0.8, start=6, end=8),
    ]

    selected = pack_context(items, budget_words=8, reject_source_overlap_above=0.4)

    assert [item.id for item in selected] == ["a", "c"]
    assert sum(item.word_count for item in selected) == 6


def test_context_source_utilization_penalizes_overlap():
    items = [
        candidate("a", "one two three four", 1.0, start=0, end=4),
        candidate("b", "three four five six", 0.9, start=2, end=6),
    ]

    assert context_source_utilization(items) == 6 / 8


def test_empty_context_utilization_is_zero():
    assert context_source_utilization([]) == 0.0
