from rag_practice.ir import InvertedIndex


def test_inverted_index_statistics() -> None:
    index = InvertedIndex({"d1": "cat cat dog", "d2": "dog bird"})

    assert index.document_count == 2
    assert index.document_frequency("cat") == 1
    assert index.document_frequency("dog") == 2
    assert index.document_lengths == {"d1": 3, "d2": 2}
    assert index.average_document_length == 2.5
    assert [(p.document_id, p.term_frequency) for p in index.postings_for("cat")] == [
        ("d1", 2)
    ]
