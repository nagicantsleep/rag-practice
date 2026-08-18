from dataclasses import dataclass

from .documents import Document


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    index: int


def chunk_document(
    document: Document,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    """Character-based chunking kept intentionally simple for experimentation."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text = " ".join(document.text.split())
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(
            Chunk(text=text[start:end], source=document.source, index=index)
        )
        if end == len(text):
            break
        start += step
        index += 1

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size=chunk_size, overlap=overlap)
    ]
