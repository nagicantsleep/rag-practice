"""Source-aware evaluation for the M08 Web RAG sub-lab."""

from __future__ import annotations

from statistics import fmean
from collections.abc import Mapping, Sequence

from rag_practice.evaluation.retrieval import evaluate_rankings
from rag_practice.sources.base import SourceRecord


def _contains(answer: str, reference: str) -> float:
    return float(reference.casefold() in answer.casefold())


def duplicate_rate(
    ranked_ids: Sequence[str],
    records: Mapping[str, SourceRecord],
    *,
    k: int,
) -> float:
    selected = list(ranked_ids[:k])
    if not selected:
        return 0.0
    canonical = [
        str(records[record_id].metadata.get("canonical_url", records[record_id].locator))
        for record_id in selected
    ]
    return (len(canonical) - len(set(canonical))) / len(canonical)


def evaluate_web_system(
    *,
    rankings: Mapping[str, Sequence[str]],
    answers: Mapping[str, str],
    traces: Mapping[str, Mapping[str, float | int | bool]],
    questions: Sequence[Mapping[str, object]],
    records: Mapping[str, SourceRecord],
) -> dict[str, float]:
    qrels = {
        str(question["id"]): {
            str(record_id): 1.0
            for record_id in question["relevant"]
        }
        for question in questions
    }
    metrics = evaluate_rankings(rankings, qrels, ks=(1, 3))

    stale_top1 = []
    low_authority_top1 = []
    answer_correct = []
    grounded = []
    duplicate_at_3 = []

    for question in questions:
        query_id = str(question["id"])
        ranked = list(rankings.get(query_id, ()))
        top_id = ranked[0] if ranked else None
        stale_ids = {str(item) for item in question.get("stale_ids", [])}
        if bool(question.get("requires_freshness", False)):
            stale_top1.append(float(top_id in stale_ids if top_id else False))

        min_authority = float(question.get("min_authority", 0.0))
        if top_id is None:
            low_authority_top1.append(1.0)
            grounded.append(0.0)
        else:
            authority = float(records[top_id].metadata.get("authority", 0.0))
            low_authority_top1.append(float(authority < min_authority))
            grounded.append(float(answers.get(query_id, "") == records[top_id].content))

        answer_correct.append(
            _contains(answers.get(query_id, ""), str(question["answer"]))
        )
        duplicate_at_3.append(duplicate_rate(ranked, records, k=3))

    metrics.update(
        {
            "stale_top1_rate": fmean(stale_top1) if stale_top1 else 0.0,
            "low_authority_top1_rate": fmean(low_authority_top1),
            "duplicate_rate@3": fmean(duplicate_at_3),
            "answer_contains_reference": fmean(answer_correct),
            "grounded_answer_rate": fmean(grounded),
            "mean_source_calls": fmean(float(t["source_calls"]) for t in traces.values()),
            "mean_search_ms": fmean(float(t["search_ms"]) for t in traces.values()),
            "mean_rerank_ms": fmean(float(t["rerank_ms"]) for t in traces.values()),
            "mean_end_to_end_ms": fmean(float(t["end_to_end_ms"]) for t in traces.values()),
        }
    )
    return metrics
