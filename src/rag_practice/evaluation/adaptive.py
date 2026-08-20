from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean

from rag_practice.adaptive.control import ControlTrace
from rag_practice.adaptive.router import Route


def _safe_mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def evaluate_control_traces(
    rows: Sequence[Mapping[str, object]],
    traces: Mapping[str, ControlTrace],
) -> dict[str, float]:
    if not rows:
        return {}

    route_hits: list[float] = []
    evidence_recalls: list[float] = []
    complete_recalls: list[float] = []
    calls: list[float] = []
    unnecessary_calls: list[float] = []
    iterative_under_routes: list[float] = []
    correction_tp = 0
    correction_fp = 0
    correction_fn = 0

    for row in rows:
        query_id = str(row["id"])
        trace = traces[query_id]
        expected_route = Route(str(row["route"]))
        route_hits.append(float(trace.route == expected_route))
        calls.append(float(trace.retrieval_calls))

        if expected_route == Route.NO_RETRIEVAL:
            unnecessary_calls.append(float(trace.retrieval_calls > 0))
        if expected_route == Route.ITERATIVE:
            iterative_under_routes.append(float(trace.route != Route.ITERATIVE))

        relevant = {str(item) for item in row.get("relevant_document_ids", [])}
        selected = set(trace.selected_document_ids)
        if relevant:
            recall = len(selected & relevant) / len(relevant)
            evidence_recalls.append(recall)
            complete_recalls.append(float(relevant.issubset(selected)))

        expected_correction = bool(row.get("needs_correction", False))
        if trace.correction_triggered and expected_correction:
            correction_tp += 1
        elif trace.correction_triggered and not expected_correction:
            correction_fp += 1
        elif expected_correction and not trace.correction_triggered:
            correction_fn += 1

    correction_precision = correction_tp / (correction_tp + correction_fp) if correction_tp + correction_fp else 0.0
    correction_recall = correction_tp / (correction_tp + correction_fn) if correction_tp + correction_fn else 0.0

    return {
        "route_accuracy": _safe_mean(route_hits),
        "evidence_recall": _safe_mean(evidence_recalls),
        "evidence_complete": _safe_mean(complete_recalls),
        "mean_retrieval_calls": _safe_mean(calls),
        "unnecessary_retrieval_rate": _safe_mean(unnecessary_calls),
        "iterative_under_route_rate": _safe_mean(iterative_under_routes),
        "correction_precision": correction_precision,
        "correction_recall": correction_recall,
    }
