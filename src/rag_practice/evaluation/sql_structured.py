"""Evaluation helpers for structured SQL RAG."""

from __future__ import annotations

from statistics import fmean
from typing import Mapping, Sequence

from rag_practice.evaluation.retrieval import recall_at_k


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def evidence_complete(
    retrieved: Sequence[str],
    relevant: Sequence[str],
) -> float:
    wanted = set(relevant)
    return float(bool(wanted) and wanted.issubset(set(retrieved)))


def evaluate_flat_row_baseline(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Sequence[str]],
    *,
    k: int = 5,
) -> dict[str, float]:
    if not qrels:
        return {}
    return {
        f"evidence_recall@{k}": fmean(
            recall_at_k(rankings.get(query_id, ()), relevant, k)
            for query_id, relevant in qrels.items()
        ),
        f"evidence_complete@{k}": fmean(
            evidence_complete(rankings.get(query_id, ())[:k], relevant)
            for query_id, relevant in qrels.items()
        ),
    }


def evaluate_structured_traces(
    traces: Mapping[str, object],
    cases: Mapping[str, Mapping[str, object]],
) -> dict[str, float]:
    safe_ids = [qid for qid, case in cases.items() if not case.get("unsafe") and not case.get("unsupported")]
    unsafe_ids = [qid for qid, case in cases.items() if case.get("unsafe")]
    unsupported_ids = [qid for qid, case in cases.items() if case.get("unsupported")]
    nonempty_ids = [qid for qid in safe_ids if not cases[qid].get("expect_empty")]
    empty_ids = [qid for qid in safe_ids if cases[qid].get("expect_empty")]

    execution_success = [
        float(getattr(traces[qid], "status") == "ok") for qid in safe_ids
    ]
    answer_exact = [
        float(
            normalize_answer(getattr(traces[qid], "answer"))
            == normalize_answer(str(cases[qid]["answer"]))
        )
        for qid in safe_ids
    ]
    evidence_recalls = []
    complete = []
    schema_recall = []
    schema_precision = []
    for qid in safe_ids:
        trace = traces[qid]
        relevant = list(cases[qid].get("evidence", []))
        retrieved = list(getattr(trace, "evidence_ids"))
        if relevant:
            evidence_recalls.append(len(set(retrieved) & set(relevant)) / len(set(relevant)))
            complete.append(evidence_complete(retrieved, relevant))
        expected_tables = set(cases[qid].get("tables", []))
        planned_tables = set(getattr(trace, "plan").tables if getattr(trace, "plan") else ())
        schema_recall.append(
            len(planned_tables & expected_tables) / len(expected_tables)
            if expected_tables else 1.0
        )
        schema_precision.append(
            len(planned_tables & expected_tables) / len(planned_tables)
            if planned_tables else float(not expected_tables)
        )

    result = {
        "execution_success_rate": fmean(execution_success) if execution_success else 0.0,
        "answer_exact_match": fmean(answer_exact) if answer_exact else 0.0,
        "evidence_recall": fmean(evidence_recalls) if evidence_recalls else 0.0,
        "evidence_complete_rate": fmean(complete) if complete else 0.0,
        "schema_table_recall": fmean(schema_recall) if schema_recall else 0.0,
        "schema_table_precision": fmean(schema_precision) if schema_precision else 0.0,
        "unsafe_rejection_rate": fmean(
            float(getattr(traces[qid], "status") == "rejected") for qid in unsafe_ids
        ) if unsafe_ids else 0.0,
        "unsupported_handling_rate": fmean(
            float(getattr(traces[qid], "status") == "planning_error") for qid in unsupported_ids
        ) if unsupported_ids else 0.0,
        "empty_result_accuracy": fmean(
            float(getattr(traces[qid], "answer") == str(cases[qid]["answer"]))
            for qid in empty_ids
        ) if empty_ids else 0.0,
        "nonempty_answer_exact_match": fmean(
            float(
                normalize_answer(getattr(traces[qid], "answer"))
                == normalize_answer(str(cases[qid]["answer"]))
            )
            for qid in nonempty_ids
        ) if nonempty_ids else 0.0,
        "mean_execution_ms": fmean(
            getattr(traces[qid], "execution_ms") for qid in safe_ids
        ) if safe_ids else 0.0,
        "mean_end_to_end_ms": fmean(
            getattr(traces[qid], "end_to_end_ms") for qid in cases
        ) if cases else 0.0,
    }
    return result
