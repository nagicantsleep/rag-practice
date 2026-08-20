from __future__ import annotations

from rag_practice.core.models import Chunk, Document


class FixedSizeChunker:
    def __init__(self, chunk_size_words: int = 80, overlap_words: int = 10) -> None:
        if chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be positive")
        if overlap_words < 0:
            raise ValueError("overlap_words must be non-negative")
        if overlap_words >= chunk_size_words:
            raise ValueError("overlap_words must be smaller than chunk_size_words")
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def chunk(self, document: Document) -> list[Chunk]:
        words = document.text.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size_words - self.overlap_words
        for index, start in enumerate(range(0, len(words), step)):
            end = min(start + self.chunk_size_words, len(words))
            chunks.append(
                Chunk(
                    id=f"{document.id}::chunk-{index}",
                    document_id=document.id,
                    text=" ".join(words[start:end]),
                    start_word=start,
                    end_word=end,
                    metadata=dict(document.metadata),
                )
            )
            if end == len(words):
                break
        return chunks

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        return [chunk for document in documents for chunk in self.chunk(document)]
