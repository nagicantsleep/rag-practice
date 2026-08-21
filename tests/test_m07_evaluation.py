from rag_practice.evaluation.structured import evidence_complete_at_budget, recall_at_budget, reciprocal_rank

def test_evidence_budget_metrics_are_hand_checkable():
    ranked=["a","x","b"]
    relevant=["a","b"]
    assert recall_at_budget(ranked,relevant,2)==0.5
    assert evidence_complete_at_budget(ranked,relevant,2)==0.0
    assert evidence_complete_at_budget(ranked,relevant,3)==1.0
    assert reciprocal_rank(ranked,relevant)==1.0
