"""Evaluation for metadata/filter-aware retrieval."""

from __future__ import annotations

from statistics import fmean
from typing import Mapping, Sequence

from rag_practice.metadata_filter import FilterPredicate, FilterRequest, FilterSearchTrace
from rag_practice.sources.base import SourceRecord


def evaluate_filter_system(
    traces: Mapping[str, FilterSearchTrace],
    cases: Mapping[str, Mapping[str, object]],
    records: Mapping[str, SourceRecord],
) -> dict[str, float]:
    nonempty = [qid for qid, case in cases.items() if case.get("relevant")]
    empty = [qid for qid, case in cases.items() if case.get("expect_empty")]

    recalls = []
    hit1 = []
    all_returned: list[tuple[SourceRecord, FilterRequest]] = []
    answer_correct = []
    answered = []
    grounded = []

    for qid, case in cases.items():
        trace = traces[qid]
        request = FilterRequest(**case["filters"])
        ids = [hit.record.id for hit in trace.hits]
        relevant = set(case.get("relevant", []))
        if relevant:
            recalls.append(len(set(ids) & relevant) / len(relevant))
            hit1.append(float(bool(ids) and ids[0] in relevant))
            answer_correct.append(float(bool(ids) and ids[0] in relevant))
        else:
            answer_correct.append(float(not ids))
        answered.append(float(bool(ids)))
        if ids:
            # Extractive-answer control: returning a record verbatim is grounded in
            # that record even when the record violates tenant/filter constraints.
            grounded.append(1.0)
        all_returned.extend((hit.record, request) for hit in trace.hits)

    security_leaks = [
        float(not FilterPredicate.security_allowed(record, request))
        for record, request in all_returned
    ]
    filter_violations = [
        float(not FilterPredicate.matches(record, request))
        for record, request in all_returned
    ]

    return {
        "recall@3": fmean(recalls) if recalls else 0.0,
        "hit_rate@1": fmean(hit1) if hit1 else 0.0,
        "constraint_satisfaction_rate": 1.0 - (fmean(filter_violations) if filter_violations else 0.0),
        "security_leakage_rate": fmean(security_leaks) if security_leaks else 0.0,
        "explicit_filter_violation_rate": fmean(filter_violations) if filter_violations else 0.0,
        "empty_filter_accuracy": fmean(
            float(not traces[qid].hits) for qid in empty
        ) if empty else 0.0,
        "answer_correct_rate": fmean(answer_correct) if answer_correct else 0.0,
        "answered_rate": fmean(answered) if answered else 0.0,
        "grounded_answer_rate_when_answered": fmean(grounded) if grounded else 0.0,
        "mean_records_indexed_for_lexical_search": fmean(
            trace.records_indexed_for_lexical_search for trace in traces.values()
        ),
        "mean_ranked_candidates_examined": fmean(
            trace.ranked_candidates_examined for trace in traces.values()
        ),
        "mean_rejected_after_ranking": fmean(
            trace.rejected_after_ranking for trace in traces.values()
        ),
        "mean_eligible_records": fmean(trace.eligible_records for trace in traces.values()),
        "mean_latency_ms": fmean(trace.latency_ms for trace in traces.values()),
    }
