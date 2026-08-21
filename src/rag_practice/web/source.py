"""Deterministic web-snapshot source used to isolate Web RAG mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from rag_practice.ir.bm25 import BM25Index
from rag_practice.sources.base import SourceHit, SourceRecord


@dataclass(frozen=True)
class WebPage:
    id: str
    url: str
    domain: str
    title: str
    text: str
    updated_at: date
    authority: float
    canonical_url: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.authority <= 1.0:
            raise ValueError("authority must be in [0, 1]")

    def to_record(self) -> SourceRecord:
        return SourceRecord(
            id=self.id,
            source_type="web",
            locator=self.url,
            title=self.title,
            content=self.text,
            metadata={
                "domain": self.domain,
                "updated_at": self.updated_at.isoformat(),
                "authority": self.authority,
                "canonical_url": self.canonical_url,
            },
        )


class SnapshotWebSource:
    """BM25 search over a frozen web snapshot.

    ``index_metadata=False`` is the body-only lexical baseline.
    ``index_metadata=True`` additionally indexes URL-domain/title metadata.
    """

    name = "snapshot_web"

    def __init__(self, pages: list[WebPage], *, index_metadata: bool) -> None:
        if not pages:
            raise ValueError("pages must not be empty")
        self.pages = {page.id: page for page in pages}
        if len(self.pages) != len(pages):
            raise ValueError("page ids must be unique")
        self.index_metadata = index_metadata
        documents = {}
        for page in pages:
            documents[page.id] = (
                f"{page.domain} {page.title} {page.text}"
                if index_metadata
                else page.text
            )
        self.index = BM25Index(documents)

    def search(self, query: str, *, limit: int = 5) -> list[SourceHit]:
        ranked = self.index.search(query, k=limit)
        return [
            SourceHit(
                record=self.pages[page_id].to_record(),
                score=score,
                rank=rank,
                details={"lexical_score": score, "index_metadata": self.index_metadata},
            )
            for rank, (page_id, score) in enumerate(ranked, start=1)
        ]
