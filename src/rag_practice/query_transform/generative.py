from __future__ import annotations

import re
from typing import Protocol


class TextGenerator(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int = 64) -> str: ...


_NUMBERED_PREFIX = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")


def _clean_lines(text: str, *, limit: int) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        clean = _NUMBERED_PREFIX.sub("", raw).strip().strip('"')
        if not clean or clean in lines:
            continue
        lines.append(clean)
        if len(lines) >= limit:
            break
    return lines


class GenerativeQueryTransformer:
    """Prompt-visible query transformations backed by a text-to-text model."""

    def __init__(self, generator: TextGenerator) -> None:
        self.generator = generator

    def rewrite(self, query: str) -> str:
        prompt = (
            "Rewrite the search query using precise information-retrieval keywords. "
            "Preserve the intent and return only one rewritten query.\n"
            f"Query: {query}\nRewrite:"
        )
        text = self.generator.generate(prompt, max_new_tokens=48).strip()
        return text or query

    def multi_query(self, query: str, *, count: int = 3) -> list[str]:
        if count <= 0:
            return []
        prompt = (
            f"Write {count} different search queries for the same information need. "
            "Use different wording and useful keywords. Return one query per line.\n"
            f"Original query: {query}\nQueries:"
        )
        generated = _clean_lines(
            self.generator.generate(prompt, max_new_tokens=96),
            limit=count,
        )
        variants = [query]
        for item in generated:
            if item not in variants:
                variants.append(item)
        return variants[: count + 1]

    def query2doc(self, query: str) -> str:
        prompt = (
            "Write a short pseudo-document that would be relevant to this search query. "
            "Use concrete domain terms likely to occur in a relevant document.\n"
            f"Query: {query}\nPseudo-document:"
        )
        pseudo = self.generator.generate(prompt, max_new_tokens=96).strip()
        return f"{query} {pseudo}".strip()

    def hyde_document(self, query: str) -> str:
        prompt = (
            "Write a concise hypothetical passage that directly answers the question. "
            "Do not mention that it is hypothetical.\n"
            f"Question: {query}\nPassage:"
        )
        text = self.generator.generate(prompt, max_new_tokens=96).strip()
        return text or query

    def decompose(self, query: str, *, max_parts: int = 3) -> list[str]:
        if max_parts <= 0:
            return []
        prompt = (
            "Decompose the information need into independent search subquestions. "
            f"Return at most {max_parts} subquestions, one per line. If decomposition is unnecessary, return the original query.\n"
            f"Question: {query}\nSubquestions:"
        )
        parts = _clean_lines(
            self.generator.generate(prompt, max_new_tokens=96),
            limit=max_parts,
        )
        return parts or [query]
