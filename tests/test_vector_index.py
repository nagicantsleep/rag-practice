from rag_practice.core.models import Chunk
from rag_practice.retrieval.vector_index import InMemoryVectorIndex


def test_vector_index_ranks_highest_cosine_first():
    chunks = [Chunk("a", "d1", "a", 0, 1), Chunk("b", "d2", "b", 0, 1)]
    index = InMemoryVectorIndex(2)
    index.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
    results = index.search([0.9, 0.1], k=2)
    assert [result.chunk.id for result in results] == ["a", "b"]
    assert [result.rank for result in results] == [1, 2]
