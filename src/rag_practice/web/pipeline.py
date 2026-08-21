"""Small Web RAG pipeline that keeps source acquisition and ranking observable."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from time import perf_counter

from rag_practice.sources.base import Source, SourceHit
from .ranking import WebRankingPolicy


@dataclass(frozen=True)
class WebRAGResult:
    query: str
    answer: str
    citations: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    hits: tuple[SourceHit, ...]
    trace: dict[str, float | int | bool]


class ExtractiveWebRAG:
    """Search, optionally rerank, then return the top web page verbatim.

    The intentionally simple answerer makes stale/source-selection failures
    visible instead of allowing a generator to hide them.
    """

    def __init__(
        self,
        source: Source,
        *,
        policy: WebRankingPolicy | None = None,
        candidate_limit: int = 8,
        top_k: int = 3,
    ) -> None:
        if candidate_limit <= 0 or top_k <= 0:
            raise ValueError("candidate_limit and top_k must be positive")
        if candidate_limit < top_k:
            raise ValueError("candidate_limit must be >= top_k")
        self.source = source
        self.policy = policy
        self.candidate_limit = candidate_limit
        self.top_k = top_k

    def ask(self, query: str, *, as_of: date) -> WebRAGResult:
        started = perf_counter()
        search_started = perf_counter()
        candidates = self.source.search(query, limit=self.candidate_limit)
        search_ms = (perf_counter() - search_started) * 1000.0

        rerank_started = perf_counter()
        if self.policy is None:
            hits = [
                SourceHit(
                    record=hit.record,
                    score=hit.score,
                    rank=rank,
                    details=hit.details,
                )
                for rank, hit in enumerate(candidates[: self.top_k], start=1)
            ]
        else:
            hits = self.policy.rerank(
                query,
                candidates,
                as_of=as_of,
                limit=self.top_k,
            )
        rerank_ms = (perf_counter() - rerank_started) * 1000.0

        if hits:
            answer = hits[0].record.content
            citations = (hits[0].record.locator,)
        else:
            answer = ""
            citations = ()

        return WebRAGResult(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_ids=tuple(hit.record.id for hit in hits),
            hits=tuple(hits),
            trace={
                "source_calls": 1,
                "candidate_count": len(candidates),
                "search_ms": search_ms,
                "rerank_ms": rerank_ms,
                "end_to_end_ms": (perf_counter() - started) * 1000.0,
                "policy_applied": self.policy is not None,
            },
        )
