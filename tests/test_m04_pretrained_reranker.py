from rag_practice.reranking.pretrained import CrossEncoderReranker
from rag_practice.reranking.selection import RankedCandidate


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        assert kwargs["show_progress_bar"] is False
        assert kwargs["convert_to_numpy"] is True
        return [0.1 if "first" in passage else 0.9 for _, passage in pairs]


def test_cross_encoder_reranks_only_supplied_candidates():
    candidates = [
        RankedCandidate("a", "d1", "first passage", 3.0),
        RankedCandidate("b", "d2", "second passage", 2.0),
    ]
    reranker = CrossEncoderReranker("fake", model=FakeCrossEncoder())

    reranked = reranker.rerank("query", candidates)

    assert [item.id for item in reranked] == ["b", "a"]
    assert {item.id for item in reranked} == {"a", "b"}
    assert [item.rerank_score for item in reranked] == [0.9, 0.1]
    assert [item.rerank_score for item in candidates] == [None, None]


def test_cross_encoder_empty_candidates_do_not_call_model():
    reranker = CrossEncoderReranker("fake", model=FakeCrossEncoder())
    assert reranker.score("query", []) == []
