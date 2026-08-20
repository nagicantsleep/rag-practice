from rag_practice.generation.query_extract import QueryAwareExtractiveAnswerer
from rag_practice.reranking.llm import PointwiseLLMReranker, relevance_prompt
from rag_practice.reranking.selection import RankedCandidate, edge_biased_order, source_order


def candidate(identifier: str, text: str, score: float, *, doc: str = "d1", start: int = 0):
    return RankedCandidate(
        id=identifier,
        document_id=doc,
        text=text,
        first_stage_score=score,
        start_word=start,
        end_word=start + len(text.split()),
    )


class FakeScorer:
    def score_yes_no(self, prompt: str) -> float:
        return 1.0 if "atomic swap" in prompt else -1.0


def test_pointwise_llm_reranker_reorders_only_frozen_candidates():
    items = [
        candidate("a", "generic refresh note", 3.0),
        candidate("b", "use an atomic swap after re-encoding", 2.0),
    ]
    ranked = PointwiseLLMReranker(FakeScorer()).rerank("how to refresh?", items)
    assert [item.id for item in ranked] == ["b", "a"]
    assert {item.id for item in ranked} == {item.id for item in items}


def test_relevance_prompt_contains_question_and_passage():
    prompt = relevance_prompt("why rerank?", "reranking improves ordering")
    assert "why rerank?" in prompt
    assert "reranking improves ordering" in prompt
    assert prompt.rstrip().endswith("Relevant:")


def test_query_aware_extractor_is_reference_blind_and_cites_context():
    items = [
        candidate("a", "Unrelated startup note. Network checks happen later.", 1.0),
        candidate("b", "Reranking latency grows with candidate count. Batching lowers cost.", 0.9),
    ]
    result = QueryAwareExtractiveAnswerer(max_sentences=1).answer(
        "Why cap reranker candidates when latency matters?",
        items,
    )
    assert "Reranking latency" in result.text
    assert result.cited_candidate_ids == ("b",)


def test_source_order_uses_document_then_source_position():
    items = [
        candidate("c", "three", 1.0, doc="d2", start=0),
        candidate("b", "two", 1.0, doc="d1", start=20),
        candidate("a", "one", 1.0, doc="d1", start=0),
    ]
    assert [item.id for item in source_order(items)] == ["a", "b", "c"]


def test_edge_biased_order_keeps_set_and_moves_rank2_to_far_edge():
    items = [candidate(str(i), f"item {i}", 5 - i) for i in range(4)]
    ordered = edge_biased_order(items)
    assert [item.id for item in ordered] == ["0", "2", "3", "1"]
    assert {item.id for item in ordered} == {item.id for item in items}
