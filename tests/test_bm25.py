import pytest

from rag_practice.ir import BM25Index


def test_bm25_ranks_matching_document_first() -> None:
    index = BM25Index(
        {
            "d1": "probabilistic ranking information retrieval",
            "d2": "british comedy sketches",
            "d3": "vector cosine similarity",
        }
    )

    results = index.search("probabilistic retrieval", k=3)
    assert results[0][0] == "d1"
    assert results[0][1] > 0.0


def test_bm25_validates_parameters() -> None:
    with pytest.raises(ValueError):
        BM25Index({"d1": "text"}, k1=-1)
    with pytest.raises(ValueError):
        BM25Index({"d1": "text"}, b=1.1)
