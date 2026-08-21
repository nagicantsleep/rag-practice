"""Minimal source/tool contract shared by specialized-source RAG labs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from collections.abc import Mapping


@dataclass(frozen=True)
class SourceRecord:
    """One retrievable record exposed by an external or structured source."""

    id: str
    source_type: str
    locator: str
    title: str
    content: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceHit:
    """A ranked source result plus transparent scoring details."""

    record: SourceRecord
    score: float
    rank: int
    details: Mapping[str, object] = field(default_factory=dict)


class Source(Protocol):
    """Smallest useful contract for a queryable RAG source."""

    name: str

    def search(self, query: str, *, limit: int = 5) -> list[SourceHit]:
        """Return ranked records for ``query``."""
        ...
