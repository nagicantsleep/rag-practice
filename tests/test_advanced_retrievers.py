import pytest

pytest.importorskip("torch")

from rag_practice.retrieval.late_interaction import LateInteractionRetriever
from rag_practice.retrieval.learned_sparse import LearnedSparseRetriever
from rag_practice.retrieval.neural_dual_encoder import TrainingPair


@pytest.fixture
def toy():
    documents = {"d1": "cat feline pet", "d2": "car motor vehicle"}
    pairs = [
        TrainingPair("kitten feline", "d1"),
        TrainingPair("kitten animal", "d1"),
        TrainingPair("automobile vehicle", "d2"),
        TrainingPair("automobile transport", "d2"),
    ]
    return documents, pairs


def test_learned_sparse_expands_and_retrieves(toy):
    documents, pairs = toy
    model = LearnedSparseRetriever(seed=2)
    losses = model.fit(documents, pairs, epochs=120, learning_rate=0.08)
    assert losses[-1] < losses[0]
    assert model.search("kitten", k=1)[0][0] == "d1"
    assert model.mean_nonzero_dimensions() > 0


def test_late_interaction_learns_token_level_alignment(toy):
    documents, pairs = toy
    model = LateInteractionRetriever(dimensions=8, seed=2)
    losses = model.fit(documents, pairs, epochs=120, learning_rate=0.08)
    assert losses[-1] < losses[0]
    assert model.search("kitten", k=1)[0][0] == "d1"
    assert model.search("automobile", k=1)[0][0] == "d2"
