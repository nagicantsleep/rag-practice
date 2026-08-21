"""Evaluation for M08.7 long-context vs retrieval routing."""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import mean
from time import perf_counter
from typing import Callable

from rag_practice.long_context.routing import (
    ContextSelection,
    DeterministicEvidenceReader,
    ExplicitLongContextRouter,
    LongContextBenchmark,
    Route,
    RoutingQuery,
    select_context,
)

RoutePolicy = Callable[[RoutingQuery, int], Route]


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", value.lower()))


def answer_is_correct(query: RoutingQuery, answer: str, *, abstain_token: str = "ABSTAIN") -> bool:
    if query.answer_kind == "list":
        if answer == abstain_token:
            return False
        expected = {_normalize(item) for item in query.expected_answer.split(";") if item.strip()}
        actual = {_normalize(item) for item in answer.split(";") if item.strip()}
        return actual == expected
    return _normalize(answer) == _normalize(query.expected_answer)


def answer_is_grounded(query: RoutingQuery, answer: str, selection: ContextSelection) -> bool | None:
    if answer == "ABSTAIN":
        return None
    context = "\n".join(selection.texts)
    lowered = query.question.lower()

    if "mandatory checks" in lowered or "mandatory checkpoints" in lowered:
        count = len(re.findall(r"Mandatory check(?:point)?\s*[—-]\s*", context, re.IGNORECASE))
        try:
            return int(answer.strip()) == count
        except ValueError:
            return False

    if "higher reserve" in lowered:
        pairs = re.findall(
            r"Orion (North|South) reserve:\s*(\d+)\s*units",
            context,
            re.IGNORECASE,
        )
        reserves = {site.lower(): int(value) for site, value in pairs}
        if {"north", "south"} <= reserves.keys():
            expected = "north" if reserves["north"] > reserves["south"] else "south"
            return _normalize(answer) == expected
        return False

    if query.answer_kind == "list":
        return all(_normalize(item) in _normalize(context) for item in answer.split(";") if item.strip())

    return _normalize(answer) in _normalize(context)


def _evidence_metrics(query: RoutingQuery, selection: ContextSelection) -> tuple[float | None, bool | None]:
    if not query.relevant:
        return None, None
    selected = set(selection.section_ids)
    relevant = set(query.relevant)
    recall = len(selected & relevant) / len(relevant)
    return recall, relevant <= selected


def evaluate_policy(
    benchmark: LongContextBenchmark,
    *,
    name: str,
    route_policy: RoutePolicy,
    reader: DeterministicEvidenceReader | object | None = None,
) -> dict[str, object]:
    active_reader = reader or DeterministicEvidenceReader(
        abstain_token=benchmark.contract.abstain_token
    )
    rows: list[dict[str, object]] = []

    for query in benchmark.queries:
        bundle = benchmark.bundles[query.bundle_id]
        route = route_policy(query, bundle.word_count)
        selection = select_context(benchmark, query, route=route)
        read_start = perf_counter()
        answer = active_reader.answer(query.question, selection.texts)
        read_ms = (perf_counter() - read_start) * 1000.0
        evidence_recall, evidence_complete = _evidence_metrics(query, selection)
        grounded = answer_is_grounded(query, answer, selection)
        no_evidence = not query.relevant
        rows.append(
            {
                "id": query.id,
                "bundle_id": query.bundle_id,
                "task_class": query.task_class,
                "question": query.question,
                "preferred_route": query.preferred_route,
                "route": route,
                "route_correct": route == query.preferred_route,
                "section_ids": list(selection.section_ids),
                "retrieval_scores": list(selection.retrieval_scores),
                "retrieval_calls": selection.retrieval_calls,
                "context_words": selection.context_words,
                "full_context_words": selection.full_context_words,
                "context_fraction": (
                    selection.context_words / selection.full_context_words
                    if selection.full_context_words
                    else 0.0
                ),
                "selection_latency_ms": selection.latency_ms,
                "reader_latency_ms": read_ms,
                "latency_ms": selection.latency_ms + read_ms,
                "answer": answer,
                "expected_answer": query.expected_answer,
                "answer_correct": answer_is_correct(
                    query,
                    answer,
                    abstain_token=benchmark.contract.abstain_token,
                ),
                "answer_grounded": grounded,
                "no_evidence": no_evidence,
                "abstention_correct": (
                    answer == benchmark.contract.abstain_token if no_evidence else None
                ),
                "evidence_recall": evidence_recall,
                "evidence_complete": evidence_complete,
            }
        )

    evidence_rows = [row for row in rows if row["evidence_recall"] is not None]
    grounded_rows = [row for row in rows if row["answer_grounded"] is not None]
    no_evidence_rows = [row for row in rows if row["no_evidence"]]
    preferred_direct = [row for row in rows if row["preferred_route"] == "direct"]
    preferred_retrieve = [row for row in rows if row["preferred_route"] == "retrieve"]

    metrics = {
        "route_accuracy": mean(float(row["route_correct"]) for row in rows),
        "evidence_recall": mean(float(row["evidence_recall"]) for row in evidence_rows),
        "evidence_complete": mean(float(row["evidence_complete"]) for row in evidence_rows),
        "answer_accuracy": mean(float(row["answer_correct"]) for row in rows),
        "grounded_answer_rate": (
            mean(float(row["answer_grounded"]) for row in grounded_rows) if grounded_rows else 0.0
        ),
        "abstention_accuracy": (
            mean(float(row["abstention_correct"]) for row in no_evidence_rows)
            if no_evidence_rows
            else 0.0
        ),
        "mean_context_words": mean(float(row["context_words"]) for row in rows),
        "mean_context_fraction": mean(float(row["context_fraction"]) for row in rows),
        "mean_retrieval_calls": mean(float(row["retrieval_calls"]) for row in rows),
        "unnecessary_retrieval_rate": (
            mean(float(row["route"] == "retrieve") for row in preferred_direct)
            if preferred_direct
            else 0.0
        ),
        "unnecessary_full_context_rate": (
            mean(float(row["route"] == "direct") for row in preferred_retrieve)
            if preferred_retrieve
            else 0.0
        ),
        "mean_latency_ms": mean(float(row["latency_ms"]) for row in rows),
    }

    by_task: dict[str, dict[str, float]] = {}
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_class"])].append(row)
    for task_class, task_rows in sorted(grouped.items()):
        task_evidence = [row for row in task_rows if row["evidence_recall"] is not None]
        by_task[task_class] = {
            "count": float(len(task_rows)),
            "answer_accuracy": mean(float(row["answer_correct"]) for row in task_rows),
            "route_accuracy": mean(float(row["route_correct"]) for row in task_rows),
            "evidence_complete": (
                mean(float(row["evidence_complete"]) for row in task_evidence)
                if task_evidence
                else 0.0
            ),
            "mean_context_words": mean(float(row["context_words"]) for row in task_rows),
        }

    return {
        "system": name,
        "metrics": metrics,
        "by_task": by_task,
        "per_query": rows,
    }


def mechanism_suite(benchmark: LongContextBenchmark) -> dict[str, object]:
    router = ExplicitLongContextRouter(benchmark.contract)

    def always_direct(_: RoutingQuery, __: int) -> Route:
        return "direct"

    def always_retrieve(_: RoutingQuery, __: int) -> Route:
        return "retrieve"

    def explicit(query: RoutingQuery, bundle_words: int) -> Route:
        return router.route(query.question, bundle_words)

    return {
        "benchmark": {
            "bundle_count": len(benchmark.bundles),
            "query_count": len(benchmark.queries),
            "retrieval_top_k": benchmark.contract.retrieval_top_k,
            "direct_word_threshold": benchmark.contract.direct_word_threshold,
            "global_route_markers": list(benchmark.contract.global_route_markers),
        },
        "systems": [
            evaluate_policy(benchmark, name="always_direct", route_policy=always_direct),
            evaluate_policy(benchmark, name="always_retrieve", route_policy=always_retrieve),
            evaluate_policy(benchmark, name="explicit_router", route_policy=explicit),
        ],
    }


def format_mechanism_markdown(result: dict[str, object]) -> str:
    systems = result["systems"]
    lines = [
        "# M08.7 Long-context vs retrieval routing results",
        "",
        "Frozen benchmark: 4 context bundles / 12 queries. Direct reading and retrieval share the same evidence bundles; only context selection changes.",
        "",
        "| System | Route acc | Evidence recall | Evidence complete | Answer acc | Grounded | Abstention | Context words | Context fraction | Retrieval calls | Unnecessary retrieval | Unnecessary full context |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in systems:
        metrics = system["metrics"]
        lines.append(
            "| {name} | {route:.3f} | {recall:.3f} | {complete:.3f} | {answer:.3f} | "
            "{grounded:.3f} | {abstain:.3f} | {words:.1f} | {fraction:.3f} | "
            "{calls:.2f} | {ur:.3f} | {uf:.3f} |".format(
                name=system["system"],
                route=metrics["route_accuracy"],
                recall=metrics["evidence_recall"],
                complete=metrics["evidence_complete"],
                answer=metrics["answer_accuracy"],
                grounded=metrics["grounded_answer_rate"],
                abstain=metrics["abstention_accuracy"],
                words=metrics["mean_context_words"],
                fraction=metrics["mean_context_fraction"],
                calls=metrics["mean_retrieval_calls"],
                ur=metrics["unnecessary_retrieval_rate"],
                uf=metrics["unnecessary_full_context_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- `always_direct` is a full-context mechanism ceiling, not evidence that every transformer should receive all available context.",
            "- `always_retrieve` uses the frozen BM25 top-2 budget. Global questions can require more evidence sections than the retrieval window can hold.",
            "- `explicit_router` sees only question text and bundle word count; qrels, expected answers, answerability, and pretrained outputs are unavailable at runtime.",
            "- Context footprint and retrieval calls are part of the policy objective; answer correctness alone does not decide the preferred route.",
            "- The benchmark is tiny and synthetic. Perfect controlled routing demonstrates the declared mechanism boundary only.",
            "",
        ]
    )
    return "\n".join(lines)
