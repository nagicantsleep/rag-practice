"""A tiny inverted index exposing the statistics used by TF-IDF and BM25."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .text import tokenize

Tokenizer = Callable[[str], list[str]]


@dataclass(frozen=True)
class Posting:
    document_id: str
    term_frequency: int


class InvertedIndex:
    """In-memory inverted index with document and collection statistics."""

    def __init__(
        self,
        documents: Mapping[str, str],
        *,
        tokenizer: Tokenizer = tokenize,
    ) -> None:
        if not documents:
            raise ValueError("documents must not be empty")

        self.documents = dict(documents)
        self.tokenizer = tokenizer
        self.document_term_frequencies: dict[str, Counter[str]] = {}
        self.document_lengths: dict[str, int] = {}
        postings: defaultdict[str, list[Posting]] = defaultdict(list)

        for document_id, text in self.documents.items():
            terms = tokenizer(text)
            frequencies = Counter(terms)
            self.document_term_frequencies[document_id] = frequencies
            self.document_lengths[document_id] = len(terms)
            for term, frequency in frequencies.items():
                postings[term].append(Posting(document_id, frequency))

        self.postings = dict(postings)
        self.document_count = len(self.documents)
        self.total_terms = sum(self.document_lengths.values())
        self.average_document_length = self.total_terms / self.document_count

    def document_frequency(self, term: str) -> int:
        """Return the number of documents containing ``term``."""

        return len(self.postings.get(term, ()))

    def postings_for(self, term: str) -> tuple[Posting, ...]:
        """Return postings for ``term`` as an immutable tuple."""

        return tuple(self.postings.get(term, ()))
