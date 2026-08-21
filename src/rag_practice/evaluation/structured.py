from __future__ import annotations
from collections import defaultdict
from statistics import fmean


def recall_at_budget(ranked: list[str], relevant: list[str], budget: int | None = None) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    use_budget = budget if budget is not None else len(relevant_set)
    return len(set(ranked[:use_budget]) & relevant_set) / len(relevant_set)


def evidence_complete_at_budget(ranked: list[str], relevant: list[str], budget: int | None = None) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    use_budget = budget if budget is not None else len(relevant_set)
    return float(relevant_set <= set(ranked[:use_budget]))


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def summarize_traces(traces: list[dict]) -> dict:
    if not traces:
        return {}
    summary = {
        "mean_recall@3": fmean(t["recall@3"] for t in traces),
        "mean_recall@5": fmean(t["recall@5"] for t in traces),
        "mean_recall@10": fmean(t["recall@10"] for t in traces),
        "mean_recall_at_evidence_budget": fmean(t["recall_at_evidence_budget"] for t in traces),
        "evidence_complete_at_budget": fmean(t["evidence_complete_at_budget"] for t in traces),
        "mrr": fmean(t["reciprocal_rank"] for t in traces),
        "mean_query_ms": fmean(t["query_ms"] for t in traces),
    }
    by_task: dict[str, list[dict]] = defaultdict(list)
    for trace in traces:
        by_task[trace["task"]].append(trace)
    summary["by_task"] = {
        task: {
            "queries": len(items),
            "mean_recall_at_evidence_budget": fmean(i["recall_at_evidence_budget"] for i in items),
            "evidence_complete_at_budget": fmean(i["evidence_complete_at_budget"] for i in items),
            "mrr": fmean(i["reciprocal_rank"] for i in items),
        }
        for task, items in sorted(by_task.items())
    }
    return summary
