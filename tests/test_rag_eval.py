from rag_practice.evaluation.rag import (
    answer_contains_reference,
    citation_precision,
    citation_recall,
    grounded_token_recall,
    token_f1,
)


def test_generation_metrics_are_hand_checkable():
    assert token_f1("alpha beta gamma", "beta gamma") == 0.8
    assert answer_contains_reference("Guido van Rossum created Python", "Guido van Rossum") == 1.0
    assert grounded_token_recall("alpha beta", ["alpha beta gamma"]) == 1.0
    assert citation_precision(["d1", "d2"], {"d1"}) == 0.5
    assert citation_recall(["d1"], {"d1", "d2"}) == 0.5
