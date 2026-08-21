"""Transparent Web RAG mechanics for M08."""

from .pipeline import ExtractiveWebRAG, WebRAGResult
from .ranking import WebRankingPolicy, query_requires_freshness
from .source import SnapshotWebSource, WebPage

__all__ = [
    "ExtractiveWebRAG",
    "SnapshotWebSource",
    "WebPage",
    "WebRAGResult",
    "WebRankingPolicy",
    "query_requires_freshness",
]
