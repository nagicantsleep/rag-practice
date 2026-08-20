from __future__ import annotations

from dataclasses import dataclass

from rag_practice.indexing.chunking import split_sentences
from rag_practice.ir.text import tokenize
from rag_practice.reranking.selection import RankedCandidate


@dataclass(frozen=True)
class ExtractiveResult:
    text: str
    cited_candidate_ids: tuple[str, ...]


def _sentence_score(question: str, sentence: str) -> float:
    query_terms = set(tokenize(question))
    sentence_terms = set(tokenize(sentence))
    if not query_terms or not sentence_terms:
        return 0.0
    return len(query_terms & sentence_terms) / len(query_terms)


class QueryAwareExtractiveAnswerer:
    """Qrel-blind deterministic answerer used to isolate context quality."""

    def __init__(self, *, max_sentences: int = 2) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        self.max_sentences = max_sentences

    def answer(self, question: str, context: list[RankedCandidate]) -> ExtractiveResult:
        scored: list[tuple[float, int, int, str, str]] = []
        for context_index, candidate in enumerate(context):
            for sentence_index, sentence in enumerate(split_sentences(candidate.text)):
                score = _sentence_score(question, sentence)
                if score > 0.0:
                    scored.append((score, context_index, sentence_index, sentence, candidate.id))

        scored.sort(key=lambda row: (-row[0], row[1], row[2], row[4]))
        chosen = scored[: self.max_sentences]
        if not chosen:
            return ExtractiveResult("", ())

        chosen.sort(key=lambda row: (row[1], row[2]))
        text = " ".join(row[3] for row in chosen)
        cited: list[str] = []
        for row in chosen:
            if row[4] not in cited:
                cited.append(row[4])
        return ExtractiveResult(text, tuple(cited))
