from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from rag_practice.core.models import Chunk, Document
from rag_practice.embeddings.hashing import HashingEmbedder

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...

    def chunk_many(self, documents: list[Document]) -> list[Chunk]: ...


@dataclass(frozen=True)
class _Unit:
    text: str
    start_word: int
    end_word: int


def _units(parts: list[str]) -> list[_Unit]:
    result: list[_Unit] = []
    cursor = 0
    for part in parts:
        clean = " ".join(part.split())
        if not clean:
            continue
        words = clean.split()
        result.append(_Unit(clean, cursor, cursor + len(words)))
        cursor += len(words)
    return result


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return [piece.strip() for piece in _SENTENCE_SPLIT_RE.split(normalized) if piece.strip()]


def split_paragraphs(text: str) -> list[str]:
    return [" ".join(piece.split()) for piece in _PARAGRAPH_SPLIT_RE.split(text) if piece.strip()]


def _pack_units(
    document: Document,
    units: list[_Unit],
    *,
    max_words: int,
    overlap_units: int,
    id_prefix: str,
) -> list[Chunk]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_units < 0:
        raise ValueError("overlap_units must be non-negative")
    if not units:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(units):
        selected: list[_Unit] = []
        words = 0
        cursor = start
        while cursor < len(units):
            unit = units[cursor]
            unit_words = unit.end_word - unit.start_word
            if selected and words + unit_words > max_words:
                break
            selected.append(unit)
            words += unit_words
            cursor += 1
            if words >= max_words:
                break

        first = selected[0]
        last = selected[-1]
        chunks.append(
            Chunk(
                id=f"{document.id}::{id_prefix}-{len(chunks)}",
                document_id=document.id,
                text=" ".join(unit.text for unit in selected),
                start_word=first.start_word,
                end_word=last.end_word,
                metadata=dict(document.metadata),
            )
        )
        if cursor >= len(units):
            break
        next_start = max(start + 1, cursor - overlap_units)
        start = next_start
    return chunks


class SentenceChunker:
    def __init__(self, max_words: int = 80, overlap_sentences: int = 0) -> None:
        self.max_words = max_words
        self.overlap_sentences = overlap_sentences

    def chunk(self, document: Document) -> list[Chunk]:
        return _pack_units(
            document,
            _units(split_sentences(document.text)),
            max_words=self.max_words,
            overlap_units=self.overlap_sentences,
            id_prefix="sentence",
        )

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        return [chunk for document in documents for chunk in self.chunk(document)]


class ParagraphChunker:
    def __init__(self, max_words: int = 160, overlap_paragraphs: int = 0) -> None:
        self.max_words = max_words
        self.overlap_paragraphs = overlap_paragraphs

    def chunk(self, document: Document) -> list[Chunk]:
        return _pack_units(
            document,
            _units(split_paragraphs(document.text)),
            max_words=self.max_words,
            overlap_units=self.overlap_paragraphs,
            id_prefix="paragraph",
        )

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        return [chunk for document in documents for chunk in self.chunk(document)]


class SemanticChunker:
    """Sentence-level semantic-boundary chunker using deterministic hashed features.

    This keeps the chunking mechanism inspectable and dependency-free. It is a
    chunk-boundary experiment, not a claim that feature hashing is a semantic
    embedding model.
    """

    def __init__(
        self,
        max_words: int = 100,
        similarity_threshold: float = 0.08,
        dimensions: int = 256,
    ) -> None:
        if max_words <= 0:
            raise ValueError("max_words must be positive")
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1")
        self.max_words = max_words
        self.similarity_threshold = similarity_threshold
        self.embedder = HashingEmbedder(dimensions)

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def chunk(self, document: Document) -> list[Chunk]:
        sentence_units = _units(split_sentences(document.text))
        if not sentence_units:
            return []
        vectors = self.embedder.embed_many([unit.text for unit in sentence_units])
        groups: list[list[_Unit]] = []
        current: list[_Unit] = [sentence_units[0]]
        current_words = sentence_units[0].end_word - sentence_units[0].start_word

        for index in range(1, len(sentence_units)):
            unit = sentence_units[index]
            unit_words = unit.end_word - unit.start_word
            similarity = self._dot(vectors[index - 1], vectors[index])
            should_split = (
                current_words + unit_words > self.max_words
                or similarity < self.similarity_threshold
            )
            if should_split:
                groups.append(current)
                current = [unit]
                current_words = unit_words
            else:
                current.append(unit)
                current_words += unit_words
        groups.append(current)

        chunks: list[Chunk] = []
        for index, group in enumerate(groups):
            chunks.append(
                Chunk(
                    id=f"{document.id}::semantic-{index}",
                    document_id=document.id,
                    text=" ".join(unit.text for unit in group),
                    start_word=group[0].start_word,
                    end_word=group[-1].end_word,
                    metadata=dict(document.metadata),
                )
            )
        return chunks

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        return [chunk for document in documents for chunk in self.chunk(document)]


class MetadataEnrichedChunker:
    def __init__(self, base: Chunker, fields: tuple[str, ...] = ("title", "section", "tags")) -> None:
        self.base = base
        self.fields = fields

    @staticmethod
    def _format_value(value) -> str:
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value)
        return str(value)

    def chunk(self, document: Document) -> list[Chunk]:
        prefixes = [
            f"{field}: {self._format_value(document.metadata[field])}"
            for field in self.fields
            if field in document.metadata and document.metadata[field] not in (None, "", [])
        ]
        prefix = " | ".join(prefixes)
        enriched = []
        for chunk in self.base.chunk(document):
            text = f"{prefix}\n{chunk.text}" if prefix else chunk.text
            enriched.append(
                Chunk(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    text=text,
                    start_word=chunk.start_word,
                    end_word=chunk.end_word,
                    metadata=dict(chunk.metadata),
                )
            )
        return enriched

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        return [chunk for document in documents for chunk in self.chunk(document)]
