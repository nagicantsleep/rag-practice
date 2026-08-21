from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from rag_practice.ir.text import tokenize


@dataclass(frozen=True)
class ServingDocument:
    id: str
    text: str
    roles: tuple[str, ...]
    updated_at: datetime
    trusted: bool
    source_version: int


@dataclass(frozen=True)
class QueryTrace:
    cache_hit: bool
    index_generation: int
    candidate_count: int
    acl_filtered: int
    stale_filtered: int
    untrusted_filtered: int
    returned_ids: tuple[str, ...]
    latency_ms: float


@dataclass(frozen=True)
class QueryResponse:
    documents: tuple[ServingDocument, ...]
    trace: QueryTrace


class MutableLexicalIndex:
    """Small inspectable inverted index with true per-document upsert/delete updates."""

    def __init__(self, documents: Iterable[ServingDocument] = ()) -> None:
        self.documents: dict[str, ServingDocument] = {}
        self.document_terms: dict[str, set[str]] = {}
        self.postings: dict[str, set[str]] = {}
        self.generation = 0
        for document in documents:
            self._upsert(document, increment=False)

    def _upsert(self, document: ServingDocument, *, increment: bool) -> None:
        if document.id in self.documents:
            self._remove_terms(document.id)
        terms = set(tokenize(document.text))
        self.documents[document.id] = document
        self.document_terms[document.id] = terms
        for term in terms:
            self.postings.setdefault(term, set()).add(document.id)
        if increment:
            self.generation += 1

    def upsert(self, document: ServingDocument) -> None:
        self._upsert(document, increment=True)

    def _remove_terms(self, document_id: str) -> None:
        for term in self.document_terms.get(document_id, set()):
            ids = self.postings.get(term)
            if ids is None:
                continue
            ids.discard(document_id)
            if not ids:
                del self.postings[term]
        self.document_terms.pop(document_id, None)

    def delete(self, document_id: str) -> bool:
        if document_id not in self.documents:
            return False
        self._remove_terms(document_id)
        del self.documents[document_id]
        self.generation += 1
        return True

    def score(self, query: str, document_id: str) -> float:
        if document_id not in self.documents:
            return 0.0
        query_terms = tokenize(query)
        if not query_terms:
            return 0.0
        n_documents = max(len(self.documents), 1)
        terms = self.document_terms[document_id]
        score = 0.0
        for term in query_terms:
            if term not in terms:
                continue
            df = len(self.postings.get(term, ()))
            score += math.log(1.0 + (n_documents + 1.0) / (df + 1.0))
        return score

    def rank(self, query: str, candidate_ids: Sequence[str], *, k: int = 1) -> list[str]:
        scored = [
            (document_id, self.score(query, document_id))
            for document_id in candidate_ids
            if document_id in self.documents
        ]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [document_id for document_id, _ in scored[:k]]

    @property
    def posting_entries(self) -> int:
        return sum(len(ids) for ids in self.postings.values())


def parse_document(payload: dict[str, object]) -> ServingDocument:
    return ServingDocument(
        id=str(payload["id"]),
        text=str(payload["text"]),
        roles=tuple(str(role) for role in payload["roles"]),
        updated_at=datetime.fromisoformat(str(payload["updated_at"])),
        trusted=bool(payload["trusted"]),
        source_version=int(payload["source_version"]),
    )


class GuardedServingIndex:
    def __init__(
        self,
        documents: Iterable[ServingDocument],
        *,
        clock: datetime,
        max_age_days: int,
        require_trusted: bool = True,
    ) -> None:
        self.index = MutableLexicalIndex(documents)
        self.clock = clock
        self.max_age_days = max_age_days
        self.require_trusted = require_trusted
        self.cache: dict[tuple[object, ...], tuple[ServingDocument, ...]] = {}
        self.cached_policy_counts: dict[tuple[object, ...], tuple[int, int, int, int]] = {}

    def _cache_key(self, query: str, roles: Sequence[str]) -> tuple[object, ...]:
        return (
            query,
            tuple(sorted(roles)),
            self.index.generation,
            self.max_age_days,
            self.require_trusted,
        )

    def upsert(self, document: ServingDocument) -> float:
        started = time.perf_counter()
        self.index.upsert(document)
        return (time.perf_counter() - started) * 1000.0

    def delete(self, document_id: str) -> tuple[bool, float]:
        started = time.perf_counter()
        deleted = self.index.delete(document_id)
        return deleted, (time.perf_counter() - started) * 1000.0

    def query(self, query: str, *, roles: Sequence[str], k: int = 1) -> QueryResponse:
        started = time.perf_counter()
        key = self._cache_key(query, roles)
        cached = self.cache.get(key)
        if cached is not None:
            candidate_count, acl_filtered, stale_filtered, untrusted_filtered = self.cached_policy_counts[key]
            return QueryResponse(
                documents=cached,
                trace=QueryTrace(
                    cache_hit=True,
                    index_generation=self.index.generation,
                    candidate_count=candidate_count,
                    acl_filtered=acl_filtered,
                    stale_filtered=stale_filtered,
                    untrusted_filtered=untrusted_filtered,
                    returned_ids=tuple(document.id for document in cached),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                ),
            )

        allowed_roles = set(roles)
        candidates: list[str] = []
        acl_filtered = 0
        stale_filtered = 0
        untrusted_filtered = 0
        cutoff = self.clock - timedelta(days=self.max_age_days)
        for document_id, document in sorted(self.index.documents.items()):
            if not (allowed_roles & set(document.roles)):
                acl_filtered += 1
                continue
            if document.updated_at < cutoff:
                stale_filtered += 1
                continue
            if self.require_trusted and not document.trusted:
                untrusted_filtered += 1
                continue
            candidates.append(document_id)

        ranked_ids = self.index.rank(query, candidates, k=k)
        returned = tuple(self.index.documents[document_id] for document_id in ranked_ids)
        self.cache[key] = returned
        self.cached_policy_counts[key] = (
            len(candidates),
            acl_filtered,
            stale_filtered,
            untrusted_filtered,
        )
        return QueryResponse(
            documents=returned,
            trace=QueryTrace(
                cache_hit=False,
                index_generation=self.index.generation,
                candidate_count=len(candidates),
                acl_filtered=acl_filtered,
                stale_filtered=stale_filtered,
                untrusted_filtered=untrusted_filtered,
                returned_ids=ranked_ids and tuple(ranked_ids) or (),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            ),
        )


class UnsafeServingIndex:
    """Deliberately unsafe baseline: query-only cache, no policy filters, no invalidation."""

    def __init__(self, documents: Iterable[ServingDocument]) -> None:
        self.index = MutableLexicalIndex(documents)
        self.cache: dict[str, tuple[ServingDocument, ...]] = {}

    def upsert(self, document: ServingDocument) -> float:
        started = time.perf_counter()
        self.index.upsert(document)
        return (time.perf_counter() - started) * 1000.0

    def delete(self, document_id: str) -> tuple[bool, float]:
        started = time.perf_counter()
        deleted = self.index.delete(document_id)
        return deleted, (time.perf_counter() - started) * 1000.0

    def query(self, query: str, *, roles: Sequence[str], k: int = 1) -> QueryResponse:
        del roles
        started = time.perf_counter()
        if query in self.cache:
            cached = self.cache[query]
            return QueryResponse(
                documents=cached,
                trace=QueryTrace(
                    cache_hit=True,
                    index_generation=self.index.generation,
                    candidate_count=len(self.index.documents),
                    acl_filtered=0,
                    stale_filtered=0,
                    untrusted_filtered=0,
                    returned_ids=tuple(document.id for document in cached),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                ),
            )
        ranked_ids = self.index.rank(query, sorted(self.index.documents), k=k)
        returned = tuple(self.index.documents[document_id] for document_id in ranked_ids)
        self.cache[query] = returned
        return QueryResponse(
            documents=returned,
            trace=QueryTrace(
                cache_hit=False,
                index_generation=self.index.generation,
                candidate_count=len(self.index.documents),
                acl_filtered=0,
                stale_filtered=0,
                untrusted_filtered=0,
                returned_ids=tuple(ranked_ids),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            ),
        )
