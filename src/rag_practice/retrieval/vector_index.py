from __future__ import annotations

from rag_practice.core.models import Chunk, RetrievedChunk
from rag_practice.ir.vector import cosine_similarity


class InMemoryVectorIndex:
    def __init__(self, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self._entries: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError("vector dimension mismatch")
        self._entries.extend(zip(chunks, vectors))

    def search(self, query_vector: list[float], k: int = 5) -> list[RetrievedChunk]:
        if len(query_vector) != self.dimensions:
            raise ValueError("query vector dimension mismatch")
        if k <= 0:
            raise ValueError("k must be positive")

        scored = [
            (chunk, cosine_similarity(query_vector, vector))
            for chunk, vector in self._entries
        ]
        scored.sort(key=lambda item: (-item[1], item[0].id))
        return [
            RetrievedChunk(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(scored[:k], start=1)
        ]
