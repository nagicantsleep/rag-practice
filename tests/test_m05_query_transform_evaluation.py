from rag_practice.evaluation.query_transform import (
    complete_recall_at_k,
    query_class_breakdown,
)


def test_complete_recall_requires_every_relevant_document():
    rankings = {
        "q1": ["a", "x", "b"],
        "q2": ["c", "x", "y"],
    }
    qrels = {
        "q1": {"a", "b"},
        "q2": {"c", "d"},
    }
    assert complete_recall_at_k(rankings, qrels, k=3) == 0.5


def test_complete_recall_validates_k():
    try:
        complete_recall_at_k({"q": ["a"]}, {"q": {"a"}}, k=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_query_class_breakdown_keeps_multi_aspect_partial_recall_visible():
    rows = [
        {"id": "x", "class": "exact", "relevant_document_ids": ["a"]},
        {"id": "m", "class": "multi_aspect", "relevant_document_ids": ["b", "c"]},
    ]
    rankings = {
        "x": ["a", "z"],
        "m": ["b", "z"],
    }
    metrics = query_class_breakdown(rankings, rows, ks=(1, 3), complete_k=3)

    assert metrics["exact"]["recall@1"] == 1.0
    assert metrics["multi_aspect"]["recall@3"] == 0.5
    assert metrics["multi_aspect"]["complete_recall@3"] == 0.0
