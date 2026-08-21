"""Evaluation helpers for M08.4 Code RAG."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rag_practice.code_rag import PythonRepositoryIndex


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_code_rankings(
    index: PythonRepositoryIndex,
    queries: Sequence[Mapping[str, object]],
    *,
    system: str,
    rankings: Mapping[str, Sequence[str]],
    latencies_ms: Mapping[str, float],
    k: int = 4,
) -> dict[str, object]:
    """Evaluate file- or symbol-level retrieval without conflating the units."""

    recalls: list[float] = []
    completes: list[float] = []
    primary_hits: list[float] = []
    single_answer_exact: list[float] = []
    dependency_complete: list[float] = []
    callsite_confusions: list[float] = []
    duplicate_confusions: list[float] = []
    context_chars: list[float] = []
    per_query: list[dict[str, object]] = []

    file_level = system == "file_bm25"

    for query in queries:
        query_id = str(query["id"])
        relevant_symbols = [str(item) for item in query["relevant_symbols"]]
        primary_symbol = str(query["primary_symbol"])
        ranking = list(rankings[query_id])[:k]

        if file_level:
            relevant = sorted({symbol_id.split("::", 1)[0] for symbol_id in relevant_symbols})
            primary = primary_symbol.split("::", 1)[0]
            retrieved = ranking
            chars = sum(len(index.files[path]) for path in retrieved)
        else:
            relevant = relevant_symbols
            primary = primary_symbol
            retrieved = ranking
            chars = sum(len(index.symbols[symbol_id].source) for symbol_id in retrieved)

        relevant_set = set(relevant)
        retrieved_set = set(retrieved)
        recall = len(relevant_set & retrieved_set) / len(relevant_set) if relevant_set else 1.0
        complete = float(relevant_set.issubset(retrieved_set))
        primary_hit = float(bool(retrieved) and retrieved[0] == primary)

        recalls.append(recall)
        completes.append(complete)
        primary_hits.append(primary_hit)
        context_chars.append(float(chars))

        if len(relevant_symbols) == 1:
            single_answer_exact.append(primary_hit)
        if str(query["task"]) in {"dependency", "change_locality"}:
            dependency_complete.append(complete)
        if str(query["task"]) == "implementation_vs_callsite":
            callsite_confusions.append(1.0 - primary_hit)
        if str(query["task"]) == "duplicate_symbol":
            duplicate_confusions.append(1.0 - primary_hit)

        if file_level:
            locators: list[str] = []
        else:
            locators = [index.symbols[symbol_id].locator for symbol_id in retrieved]

        per_query.append(
            {
                "id": query_id,
                "task": query["task"],
                "query": query["query"],
                "relevant": relevant,
                "ranking": retrieved,
                "primary": primary,
                "recall": recall,
                "complete": bool(complete),
                "primary_hit@1": bool(primary_hit),
                "locators": locators,
                "context_chars": chars,
                "latency_ms": latencies_ms[query_id],
            }
        )

    return {
        "metrics": {
            "recall@4": _mean(recalls),
            "evidence_complete@4": _mean(completes),
            "primary_hit@1": _mean(primary_hits),
            "single_evidence_answer_location_exact": _mean(single_answer_exact),
            "dependency_complete@4": _mean(dependency_complete),
            "callsite_confusion_rate": _mean(callsite_confusions),
            "duplicate_symbol_confusion_rate": _mean(duplicate_confusions),
            "mean_context_chars@4": _mean(context_chars),
            "mean_query_ms": _mean(list(latencies_ms.values())),
            "indexed_units": float(len(index.files) if file_level else len(index.symbols)),
            "exact_line_locators_available": 0.0 if file_level else 1.0,
        },
        "per_query": per_query,
    }
