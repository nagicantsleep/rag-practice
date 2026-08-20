"""TF-IDF retrieval with cosine similarity."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping

from .inverted_index import InvertedIndex, Tokenizer
from .text import tokenize
from .vector import sparse_cosine_similarity


class TfidfIndex:
    """Classic TF-IDF retriever using log-scaled term frequency and cosine."""

    def __init__(
        self,
        documents: Mapping[str, str],
        *,
        tokenizer: Tokenizer = tokenize,
    ) -> None:
        self.index = InvertedIndex(documents, tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self._document_vectors = {
            document_id: self._vectorize_frequencies(frequencies)
            for document_id, frequencies in self.index.document_term_frequencies.items()
        }

    def idf(self, term: str) -> float:
        """Smoothed inverse document frequency.

        ``log((N + 1) / (df + 1)) + 1`` keeps unseen and ubiquitous terms
        numerically well-behaved for this educational implementation.
        """

        n = self.index.document_count
        df = self.index.document_frequency(term)
        return math.log((n + 1) / (df + 1)) + 1.0

    @staticmethod
    def tf(term_frequency: int) -> float:
        """Log-scaled term frequency."""

        if term_frequency <= 0:
            return 0.0
        return 1.0 + math.log(term_frequency)

    def _vectorize_frequencies(self, frequencies: Mapping[str, int]) -> dict[str, float]:
        return {
            term: self.tf(frequency) * self.idf(term)
            for term, frequency in frequencies.items()
        }

    def vectorize_query(self, query: str) -> dict[str, float]:
        return self._vectorize_frequencies(Counter(self.tokenizer(query)))

    def score(self, query: str, document_id: str) -> float:
        if document_id not in self.index.documents:
            raise KeyError(document_id)
        query_vector = self.vectorize_query(query)
        return sparse_cosine_similarity(query_vector, self._document_vectors[document_id])

    def search(self, query: str, *, k: int = 5) -> list[tuple[str, float]]:
        if k <= 0:
            return []

        query_vector = self.vectorize_query(query)
        scored = [
            (document_id, sparse_cosine_similarity(query_vector, document_vector))
            for document_id, document_vector in self._document_vectors.items()
        ]
        scored = [item for item in scored if item[1] > 0.0]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:k]
