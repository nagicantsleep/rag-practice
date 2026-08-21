"""Query-aware Web RAG reranking: relevance + authority + freshness + dedupe."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from rag_practice.ir.text import tokenize
from rag_practice.sources.base import SourceHit


_FRESHNESS_TERMS = {
    "current",
    "currently",
    "latest",
    "newest",
    "now",
    "recent",
    "today",
}


def query_requires_freshness(query: str) -> bool:
    return bool(set(tokenize(query)) & _FRESHNESS_TERMS)


@dataclass(frozen=True)
class WebRankingPolicy:
    """Transparent score fusion for a small Web RAG candidate set."""

    current_lexical_weight: float = 0.35
    current_authority_weight: float = 0.45
    current_freshness_weight: float = 0.20
    static_lexical_weight: float = 0.65
    static_authority_weight: float = 0.35
    freshness_horizon_days: int = 365
    deduplicate: bool = True

    def __post_init__(self) -> None:
        current_total = (
            self.current_lexical_weight
            + self.current_authority_weight
            + self.current_freshness_weight
        )
        static_total = self.static_lexical_weight + self.static_authority_weight
        if abs(current_total - 1.0) > 1e-9:
            raise ValueError("current-intent weights must sum to 1")
        if abs(static_total - 1.0) > 1e-9:
            raise ValueError("static-intent weights must sum to 1")
        if self.freshness_horizon_days <= 0:
            raise ValueError("freshness_horizon_days must be positive")

    def rerank(
        self,
        query: str,
        hits: list[SourceHit],
        *,
        as_of: date,
        limit: int,
    ) -> list[SourceHit]:
        if limit <= 0 or not hits:
            return []

        lexical_scores = [float(hit.details.get("lexical_score", hit.score)) for hit in hits]
        minimum = min(lexical_scores)
        maximum = max(lexical_scores)
        current_intent = query_requires_freshness(query)

        scored: list[SourceHit] = []
        for hit, lexical_score in zip(hits, lexical_scores, strict=True):
            if maximum > minimum:
                lexical_normalized = (lexical_score - minimum) / (maximum - minimum)
            else:
                lexical_normalized = 1.0

            authority = float(hit.record.metadata.get("authority", 0.0))
            updated_at = date.fromisoformat(str(hit.record.metadata["updated_at"]))
            age_days = max(0, (as_of - updated_at).days)
            freshness = max(0.0, 1.0 - age_days / self.freshness_horizon_days)

            if current_intent:
                score = (
                    self.current_lexical_weight * lexical_normalized
                    + self.current_authority_weight * authority
                    + self.current_freshness_weight * freshness
                )
            else:
                score = (
                    self.static_lexical_weight * lexical_normalized
                    + self.static_authority_weight * authority
                )

            scored.append(
                SourceHit(
                    record=hit.record,
                    score=score,
                    rank=0,
                    details={
                        **dict(hit.details),
                        "lexical_normalized": lexical_normalized,
                        "authority": authority,
                        "freshness": freshness,
                        "current_intent": current_intent,
                    },
                )
            )

        scored.sort(key=lambda hit: (-hit.score, hit.record.id))

        selected: list[SourceHit] = []
        seen_canonical: set[str] = set()
        for hit in scored:
            canonical = str(hit.record.metadata.get("canonical_url", hit.record.locator))
            if self.deduplicate and canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
            selected.append(
                SourceHit(
                    record=hit.record,
                    score=hit.score,
                    rank=len(selected) + 1,
                    details=hit.details,
                )
            )
            if len(selected) >= limit:
                break
        return selected
