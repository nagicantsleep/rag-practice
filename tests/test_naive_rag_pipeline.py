import pytest

from rag_practice.core.chunking import FixedSizeChunker
from rag_practice.core.models import Document
from rag_practice.embeddings.hashing import HashingEmbedder
from rag_practice.generation.extractive import TopChunkExtractiveGenerator
from rag_practice.rag.pipeline import NaiveRAGPipeline


def test_pipeline_requires_indexing():
    pipeline = NaiveRAGPipeline(
        embedder=HashingEmbedder(64), generator=TopChunkExtractiveGenerator()
    )
    with pytest.raises(RuntimeError):
        pipeline.answer("question")


def test_pipeline_returns_trace_context_and_citation():
    pipeline = NaiveRAGPipeline(
        embedder=HashingEmbedder(128),
        generator=TopChunkExtractiveGenerator(),
        chunker=FixedSizeChunker(50, 0),
    )
    pipeline.index_documents(
        [
            Document("d1", "Python was created by Guido van Rossum."),
            Document("d2", "Cosine similarity compares vectors."),
        ]
    )
    trace = pipeline.answer("Who created Python?", top_k=1)
    assert trace.retrieved[0].chunk.document_id == "d1"
    assert trace.answer.cited_chunk_ids == ("d1::chunk-0",)
    assert "[d1::chunk-0]" in trace.context
    assert "Who created Python?" in trace.prompt
    assert trace.timings_ms["end_to_end"] >= 0
    assert trace.prompt_tokens > 0
