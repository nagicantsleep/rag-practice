from rag_practice.embeddings.hashing import HashingEmbedder


def test_hashing_embedding_is_deterministic_and_normalized():
    embedder = HashingEmbedder(64)
    left = embedder.embed("dense retrieval vectors")
    right = embedder.embed("dense retrieval vectors")
    assert left == right
    assert abs(sum(value * value for value in left) - 1.0) < 1e-9


def test_empty_text_embeds_to_zero_vector():
    assert HashingEmbedder(8).embed("") == [0.0] * 8
