"""BM25 retrieval implemented directly from its scoring formula."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping

from .inverted_index import InvertedIndex, Tokenizer
from .text import tokenize


class BM25Index:
    """Okapi BM25 over an in-memory inverted index."""

    def __init__(
        self,
        documents: Mapping[str, str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Tokenizer = tokenize,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")

        self.index = InvertedIndex(documents, tokenizer=tokenizer)
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer

    def idf(self, term: str) -> float:
        """Robertson/Sparck Jones IDF with a +1 guard for positivity."""

        n = self.index.document_count
        df = self.index.document_frequency(term)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str, document_id: str) -> float:
        if document_id not in self.index.documents:
            raise KeyError(document_id)

        query_terms = Counter(self.tokenizer(query))
        frequencies = self.index.document_term_frequencies[document_id]
        document_length = self.index.document_lengths[document_id]
        average_length = self.index.average_document_length

        score = 0.0
        for term, query_frequency in query_terms.items():
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue

            denominator = term_frequency + self.k1 * (
                1.0 - self.b + self.b * document_length / average_length
            )
            normalized_tf = term_frequency * (self.k1 + 1.0) / denominator
            score += query_frequency * self.idf(term) * normalized_tf

        return score

    def search(self, query: str, *, k: int = 5) -> list[tuple[str, float]]:
        if k <= 0:
            return []

        query_terms = Counter(self.tokenizer(query))
        candidates: set[str] = set()
        for term in query_terms:
            candidates.update(posting.document_id for posting in self.index.postings_for(term))

        scores: defaultdict[str, float] = defaultdict(float)
        average_length = self.index.average_document_length

        for term, query_frequency in query_terms.items():
            term_idf = self.idf(term)
            for posting in self.index.postings_for(term):
                document_length = self.index.document_lengths[posting.document_id]
                denominator = posting.term_frequency + self.k1 * (
                    1.0 - self.b + self.b * document_length / average_length
                )
                normalized_tf = posting.term_frequency * (self.k1 + 1.0) / denominator
                scores[posting.document_id] += query_frequency * term_idf * normalized_tf

        ranked = [
            (document_id, scores[document_id])
            for document_id in candidates
            if scores[document_id] > 0.0
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:k]
