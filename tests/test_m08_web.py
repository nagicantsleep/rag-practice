"""Regression tests for the M08 Web RAG source/ranking mechanics."""

from datetime import date

from rag_practice.sources.base import SourceRecord
from rag_practice.web.pipeline import ExtractiveWebRAG
from rag_practice.web.ranking import WebRankingPolicy, query_requires_freshness
from rag_practice.web.source import SnapshotWebSource, WebPage
from rag_practice.evaluation.web import duplicate_rate


def _pages() -> list[WebPage]:
    return [
        WebPage(
            id="old",
            url="https://official.test/old",
            domain="official.test",
            title="Product release",
            text="Product 2 is the current stable release.",
            updated_at=date(2026, 1, 1),
            authority=1.0,
            canonical_url="https://official.test/old",
        ),
        WebPage(
            id="new",
            url="https://official.test/new",
            domain="official.test",
            title="Product release",
            text="Product 3 is the current stable release.",
            updated_at=date(2026, 8, 18),
            authority=1.0,
            canonical_url="https://official.test/new",
        ),
        WebPage(
            id="forum",
            url="https://forum.test/current",
            domain="forum.test",
            title="What is the current stable Product version?",
            text="Product 2 is the current stable release.",
            updated_at=date(2026, 8, 19),
            authority=0.2,
            canonical_url="https://forum.test/current",
        ),
        WebPage(
            id="mirror",
            url="https://mirror.test/new",
            domain="mirror.test",
            title="Product release mirror",
            text="Product 3 is the current stable release.",
            updated_at=date(2026, 8, 19),
            authority=0.4,
            canonical_url="https://official.test/new",
        ),
    ]


def test_freshness_intent_detection_is_explicit() -> None:
    assert query_requires_freshness("What is the latest release?")
    assert query_requires_freshness("Who currently leads the team?")
    assert not query_requires_freshness("What was the previous release?")


def test_policy_prefers_authoritative_current_page_over_fresh_forum_conflict() -> None:
    source = SnapshotWebSource(_pages(), index_metadata=True)
    pipeline = ExtractiveWebRAG(source, policy=WebRankingPolicy(), top_k=3)
    result = pipeline.ask(
        "What is the current stable Product version?",
        as_of=date(2026, 8, 20),
    )
    assert result.retrieved_ids[0] == "new"
    assert "Product 3" in result.answer


def test_policy_collapses_canonical_duplicates() -> None:
    source = SnapshotWebSource(_pages(), index_metadata=True)
    candidates = source.search("Product current stable release", limit=4)
    ranked = WebRankingPolicy().rerank(
        "Product current stable release",
        candidates,
        as_of=date(2026, 8, 20),
        limit=4,
    )
    canonical = [hit.record.metadata["canonical_url"] for hit in ranked]
    assert len(canonical) == len(set(canonical))


def test_duplicate_metric_counts_wasted_slots() -> None:
    records = {
        "a": SourceRecord("a", "web", "a", "a", "a", {"canonical_url": "x"}),
        "b": SourceRecord("b", "web", "b", "b", "b", {"canonical_url": "x"}),
        "c": SourceRecord("c", "web", "c", "c", "c", {"canonical_url": "y"}),
    }
    assert duplicate_rate(["a", "b", "c"], records, k=3) == 1 / 3
