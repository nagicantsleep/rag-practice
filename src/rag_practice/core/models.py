from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    start_word: int
    end_word: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    cited_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class RAGTrace:
    question: str
    retrieved: tuple[RetrievedChunk, ...]
    context: str
    prompt: str
    answer: GeneratedAnswer
    timings_ms: dict[str, float]
    prompt_tokens: int
    output_tokens: int
