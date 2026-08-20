from __future__ import annotations

from time import perf_counter

from rag_practice.core.chunking import FixedSizeChunker
from rag_practice.core.models import Document, RAGTrace
from rag_practice.embeddings.base import Embedder
from rag_practice.generation.base import Generator
from rag_practice.retrieval.vector_index import InMemoryVectorIndex


def count_whitespace_tokens(text: str) -> int:
    return len(text.split())


def build_context(retrieved) -> str:
    return "\n\n".join(
        f"[{item.chunk.id}] {item.chunk.text}" for item in retrieved
    )


def build_prompt(question: str, context: str) -> str:
    return (
        "Answer only from the provided context. Cite the supporting chunk IDs.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


class NaiveRAGPipeline:
    def __init__(
        self,
        *,
        embedder: Embedder,
        generator: Generator,
        chunker: FixedSizeChunker | None = None,
    ) -> None:
        self.embedder = embedder
        self.generator = generator
        self.chunker = chunker or FixedSizeChunker()
        self.index = InMemoryVectorIndex(embedder.dimensions)
        self._indexed = False

    def index_documents(self, documents: list[Document]) -> int:
        chunks = self.chunker.chunk_many(documents)
        vectors = self.embedder.embed_many([chunk.text for chunk in chunks])
        self.index = InMemoryVectorIndex(self.embedder.dimensions)
        self.index.add(chunks, vectors)
        self._indexed = True
        return len(chunks)

    def answer(self, question: str, *, top_k: int = 3) -> RAGTrace:
        if not self._indexed:
            raise RuntimeError("index_documents must be called before answer")

        start = perf_counter()
        query_vector = self.embedder.embed(question)
        after_embedding = perf_counter()
        retrieved = self.index.search(query_vector, k=top_k)
        after_retrieval = perf_counter()
        context = build_context(retrieved)
        prompt = build_prompt(question, context)
        answer = self.generator.generate(
            question=question,
            prompt=prompt,
            retrieved=retrieved,
        )
        after_generation = perf_counter()

        return RAGTrace(
            question=question,
            retrieved=tuple(retrieved),
            context=context,
            prompt=prompt,
            answer=answer,
            timings_ms={
                "embedding": (after_embedding - start) * 1000,
                "retrieval": (after_retrieval - after_embedding) * 1000,
                "generation": (after_generation - after_retrieval) * 1000,
                "end_to_end": (after_generation - start) * 1000,
            },
            prompt_tokens=count_whitespace_tokens(prompt),
            output_tokens=count_whitespace_tokens(answer.text),
        )
