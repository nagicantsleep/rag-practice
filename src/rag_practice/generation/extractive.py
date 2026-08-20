from __future__ import annotations

from rag_practice.core.models import GeneratedAnswer, RetrievedChunk


class TopChunkExtractiveGenerator:
    """Deterministic offline generator used to evaluate the RAG mechanics.

    It returns the highest-ranked chunk verbatim and cites that chunk. It is not
    an LLM. Because generation is extractive, hallucination is minimized and
    retrieval failures remain visible instead of being hidden by parametric
    model knowledge.
    """

    refusal = "I don't know based on the provided context."

    def generate(
        self,
        *,
        question: str,
        prompt: str,
        retrieved: list[RetrievedChunk],
    ) -> GeneratedAnswer:
        if not retrieved:
            return GeneratedAnswer(text=self.refusal, cited_chunk_ids=())
        top = retrieved[0]
        return GeneratedAnswer(text=top.chunk.text, cited_chunk_ids=(top.chunk.id,))
