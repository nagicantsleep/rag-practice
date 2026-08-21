"""Transparent pre-filter, post-filter, and unfiltered BM25 controls."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from rag_practice.ir.bm25 import BM25Index
from rag_practice.sources.base import SourceHit, SourceRecord


@dataclass(frozen=True)
class FilterRequest:
    """User/security context plus explicit query constraints.

    ``tenant`` and ``role`` are hard authorization constraints. ``product``,
    ``region`` and date bounds are explicit query filters. All are hard
    predicates once present; the distinction matters for security analysis.
    """

    tenant: str
    role: str
    product: str | None = None
    region: str | None = None
    updated_after: str | None = None
    updated_before: str | None = None


class FilterPredicate:
    """Evaluate authorization and explicit metadata constraints."""

    @staticmethod
    def security_allowed(record: SourceRecord, request: FilterRequest) -> bool:
        metadata = record.metadata
        record_tenant = str(metadata.get("tenant", ""))
        if record_tenant not in {request.tenant, "shared"}:
            return False
        roles = {str(role) for role in metadata.get("allowed_roles", ())}
        return not roles or request.role in roles

    @classmethod
    def matches(cls, record: SourceRecord, request: FilterRequest) -> bool:
        if not cls.security_allowed(record, request):
            return False
        metadata = record.metadata
        if request.product is not None and metadata.get("product") != request.product:
            return False
        if request.region is not None and metadata.get("region") != request.region:
            return False
        updated_at = str(metadata.get("updated_at", ""))
        if request.updated_after is not None and updated_at < request.updated_after:
            return False
        if request.updated_before is not None and updated_at > request.updated_before:
            return False
        return True


@dataclass(frozen=True)
class FilterSearchTrace:
    strategy: str
    hits: tuple[SourceHit, ...]
    total_records: int
    records_indexed_for_lexical_search: int
    ranked_candidates_examined: int
    rejected_after_ranking: int
    eligible_records: int
    latency_ms: float


class FilterAwareBM25:
    """Compare filter placement while holding lexical scoring fixed."""

    def __init__(self, records: dict[str, SourceRecord]) -> None:
        if not records:
            raise ValueError("records must not be empty")
        self.records = dict(records)
        self.predicate = FilterPredicate()

    def _rank(
        self,
        record_ids: list[str],
        query: str,
        *,
        limit: int,
    ) -> list[tuple[str, float]]:
        if limit <= 0 or not record_ids:
            return []
        corpus = {
            record_id: f"{self.records[record_id].title} {self.records[record_id].content}"
            for record_id in record_ids
        }
        return BM25Index(corpus).search(query, k=limit)

    def search(
        self,
        query: str,
        request: FilterRequest,
        *,
        strategy: Literal["unfiltered", "postfilter", "prefilter"],
        limit: int = 3,
        candidate_limit: int | None = None,
    ) -> FilterSearchTrace:
        started = perf_counter()
        all_ids = list(self.records)
        eligible_ids = [
            record_id
            for record_id, record in self.records.items()
            if self.predicate.matches(record, request)
        ]

        if strategy == "unfiltered":
            ranking = self._rank(all_ids, query, limit=limit)
            selected = ranking
            indexed = len(all_ids)
            examined = len(ranking)
            rejected = 0
        elif strategy == "postfilter":
            budget = candidate_limit if candidate_limit is not None else limit
            ranking = self._rank(all_ids, query, limit=budget)
            selected = [
                (record_id, score)
                for record_id, score in ranking
                if self.predicate.matches(self.records[record_id], request)
            ][:limit]
            indexed = len(all_ids)
            examined = len(ranking)
            rejected = sum(
                not self.predicate.matches(self.records[record_id], request)
                for record_id, _ in ranking
            )
        elif strategy == "prefilter":
            selected = self._rank(eligible_ids, query, limit=limit)
            indexed = len(eligible_ids)
            examined = len(selected)
            rejected = 0
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        hits = tuple(
            SourceHit(
                record=self.records[record_id],
                score=score,
                rank=rank,
                details={"strategy": strategy},
            )
            for rank, (record_id, score) in enumerate(selected, start=1)
        )
        return FilterSearchTrace(
            strategy=strategy,
            hits=hits,
            total_records=len(all_ids),
            records_indexed_for_lexical_search=indexed,
            ranked_candidates_examined=examined,
            rejected_after_ranking=rejected,
            eligible_records=len(eligible_ids),
            latency_ms=(perf_counter() - started) * 1000.0,
        )
