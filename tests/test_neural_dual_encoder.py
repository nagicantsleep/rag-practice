import pytest

torch = pytest.importorskip("torch")

from rag_practice.retrieval.neural_dual_encoder import TinyNeuralDualEncoder, TrainingPair


def test_dual_encoder_learns_cross_vocabulary_mapping():
    documents = {
        "d1": "feline animal cat",
        "d2": "motor vehicle car",
    }
    pairs = [
        TrainingPair("kitten feline", "d1"),
        TrainingPair("automobile vehicle", "d2"),
        TrainingPair("kitten pet", "d1"),
        TrainingPair("automobile transport", "d2"),
    ]
    retriever = TinyNeuralDualEncoder(dimensions=8, seed=3)
    losses = retriever.fit(documents, pairs, epochs=150, learning_rate=0.08)
    assert losses[-1] < losses[0]
    assert retriever.search("kitten", k=1)[0][0] == "d1"
    assert retriever.search("automobile", k=1)[0][0] == "d2"
