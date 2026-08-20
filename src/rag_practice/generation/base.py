from __future__ import annotations

from typing import Protocol

from rag_practice.core.models import GeneratedAnswer, RetrievedChunk


class Generator(Protocol):
    def generate(
        self,
        *,
        question: str,
        prompt: str,
        retrieved: list[RetrievedChunk],
    ) -> GeneratedAnswer: ...
