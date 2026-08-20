from rag_practice.ir import TfidfIndex


def test_tfidf_ranks_matching_document_first() -> None:
    index = TfidfIndex(
        {
            "d1": "python programming language",
            "d2": "british sketch comedy",
            "d3": "database query language",
        }
    )

    results = index.search("python language", k=3)
    assert results[0][0] == "d1"
    assert results[0][1] > 0.0


def test_tfidf_returns_no_results_without_lexical_overlap() -> None:
    index = TfidfIndex({"d1": "semantic embeddings"})
    assert index.search("meaning representation") == []
