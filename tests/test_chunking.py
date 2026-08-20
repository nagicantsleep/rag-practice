import pytest

from rag_practice.core.chunking import FixedSizeChunker
from rag_practice.core.models import Document


def test_fixed_size_chunker_preserves_overlap():
    chunks = FixedSizeChunker(chunk_size_words=4, overlap_words=1).chunk(
        Document("d", "one two three four five six seven")
    )
    assert [chunk.text for chunk in chunks] == ["one two three four", "four five six seven"]
    assert [(chunk.start_word, chunk.end_word) for chunk in chunks] == [(0, 4), (3, 7)]


def test_chunker_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size_words=4, overlap_words=4)
