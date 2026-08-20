from rag_practice.core.chunking import FixedSizeChunker
from rag_practice.core.models import Document
from rag_practice.evaluation.chunking import source_token_utilization
from rag_practice.indexing.chunking import (
    MetadataEnrichedChunker,
    ParagraphChunker,
    SemanticChunker,
    SentenceChunker,
)


def test_sentence_chunker_preserves_sentence_boundaries():
    document = Document("d", "Alpha beta gamma. Delta epsilon zeta. Eta theta.")
    chunks = SentenceChunker(max_words=6).chunk(document)
    assert [chunk.text for chunk in chunks] == [
        "Alpha beta gamma. Delta epsilon zeta.",
        "Eta theta.",
    ]


def test_paragraph_chunker_keeps_small_paragraphs_intact():
    document = Document("d", "First paragraph stays together.\n\nSecond paragraph also stays together.")
    chunks = ParagraphChunker(max_words=6).chunk(document)
    assert len(chunks) == 2
    assert chunks[0].text == "First paragraph stays together."
    assert chunks[1].text == "Second paragraph also stays together."


def test_semantic_chunker_splits_clear_topic_shift():
    document = Document(
        "d",
        "Cats purr and chase mice. Kittens chase toys and purr. "
        "Database indexes store postings. Search engines rank documents with BM25.",
    )
    chunks = SemanticChunker(max_words=50, similarity_threshold=0.05).chunk(document)
    assert len(chunks) >= 2
    assert "Database" not in chunks[0].text


def test_metadata_enrichment_keeps_source_span_and_adds_prefix():
    document = Document(
        "d",
        "A retrieval health probe runs before traffic.",
        metadata={"title": "Arctic Deployment", "region": "arctic"},
    )
    chunk = MetadataEnrichedChunker(
        SentenceChunker(max_words=20), fields=("title", "region")
    ).chunk(document)[0]
    assert chunk.text.startswith("title: Arctic Deployment | region: arctic")
    assert (chunk.start_word, chunk.end_word) == (0, 7)


def test_overlap_reduces_source_token_utilization():
    document = Document("d", "one two three four five six seven eight nine ten")
    plain = FixedSizeChunker(chunk_size_words=6, overlap_words=0).chunk(document)
    overlap = FixedSizeChunker(chunk_size_words=6, overlap_words=2).chunk(document)
    assert source_token_utilization(overlap) < source_token_utilization(plain)
