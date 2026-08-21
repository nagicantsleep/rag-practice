"""Pinned pretrained-reader evaluation for M08.7."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Callable

from rag_practice.evaluation.long_context import answer_is_correct, answer_is_grounded
from rag_practice.long_context.routing import (
    ExplicitLongContextRouter,
    LongContextBenchmark,
    Route,
    RoutingQuery,
    select_context,
)
from rag_practice.long_context.smollm import SmolLM2ContextReader

RoutePolicy = Callable[[RoutingQuery, int], Route]


def evaluate_pretrained_policy(
    benchmark: LongContextBenchmark,
    *,
    name: str,
    route_policy: RoutePolicy,
    reader: SmolLM2ContextReader,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for query in benchmark.queries:
        bundle = benchmark.bundles[query.bundle_id]
        route = route_policy(query, bundle.word_count)
        selection = select_context(benchmark, query, route=route)
        answer = reader.answer(query.question, selection.texts)
        trace = reader.last_trace
        if trace is None:
            raise RuntimeError("reader did not record a generation trace")

        if query.relevant:
            relevant = set(query.relevant)
            selected = set(selection.section_ids)
            evidence_recall: float | None = len(relevant & selected) / len(relevant)
            evidence_complete: bool | None = relevant <= selected
        else:
            evidence_recall = None
            evidence_complete = None

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
                "prompt_tokens": trace.prompt_tokens,
                "output_tokens": trace.output_tokens,
                "generation_ms": trace.generation_ms,
                "latency_ms": selection.latency_ms + trace.generation_ms,
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
                    answer.strip() == benchmark.contract.abstain_token if no_evidence else None
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
        "mean_prompt_tokens": mean(float(row["prompt_tokens"]) for row in rows),
        "mean_output_tokens": mean(float(row["output_tokens"]) for row in rows),
        "mean_generation_ms": mean(float(row["generation_ms"]) for row in rows),
        "mean_latency_ms": mean(float(row["latency_ms"]) for row in rows),
    }

    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_class"])].append(row)
    by_task: dict[str, dict[str, float]] = {}
    for task_class, task_rows in sorted(grouped.items()):
        task_evidence = [row for row in task_rows if row["evidence_complete"] is not None]
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
            "mean_prompt_tokens": mean(float(row["prompt_tokens"]) for row in task_rows),
        }

    return {"system": name, "metrics": metrics, "by_task": by_task, "per_query": rows}


def pretrained_suite(
    benchmark: LongContextBenchmark,
    reader: SmolLM2ContextReader,
) -> dict[str, object]:
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
        "reader": reader.metadata(),
        "systems": [
            evaluate_pretrained_policy(
                benchmark, name="smollm_always_direct", route_policy=always_direct, reader=reader
            ),
            evaluate_pretrained_policy(
                benchmark,
                name="smollm_always_retrieve",
                route_policy=always_retrieve,
                reader=reader,
            ),
            evaluate_pretrained_policy(
                benchmark, name="smollm_explicit_router", route_policy=explicit, reader=reader
            ),
        ],
    }


def format_pretrained_markdown(result: dict[str, object]) -> str:
    reader = result["reader"]
    lines = [
        "# M08.7 pinned SmolLM2 long-context routing results",
        "",
        f"Reader: `{reader['model_id']}` pinned to `{reader['revision']}`; CPU/float32 greedy generation.",
        "",
        "The reader receives only the question plus context chosen by each route policy. Qrels, expected answers, preferred routes, and answerability labels are excluded from the prompt.",
        "",
        "| System | Route acc | Evidence complete | Answer acc | Grounded | Abstention | Context words | Prompt tokens | Retrieval calls | Generation ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in result["systems"]:
        metrics = system["metrics"]
        lines.append(
            "| {name} | {route:.3f} | {complete:.3f} | {answer:.3f} | {grounded:.3f} | "
            "{abstain:.3f} | {words:.1f} | {tokens:.1f} | {calls:.2f} | {ms:.2f} |".format(
                name=system["system"],
                route=metrics["route_accuracy"],
                complete=metrics["evidence_complete"],
                answer=metrics["answer_accuracy"],
                grounded=metrics["grounded_answer_rate"],
                abstain=metrics["abstention_accuracy"],
                words=metrics["mean_context_words"],
                tokens=metrics["mean_prompt_tokens"],
                calls=metrics["mean_retrieval_calls"],
                ms=metrics["mean_generation_ms"],
            )
        )
    lines.extend(
        [
            "",
            "## Runtime / representation",
            "",
            f"- model load: {reader['model_load_ms']:.2f} ms",
            f"- parameters: {reader['parameter_count']} / {reader['parameter_bytes']} logical bytes",
            f"- tokenizer model max length: {reader['model_max_length']}",
            f"- torch: `{reader['torch_version']}`; transformers: `{reader['transformers_version']}`",
            "",
            "## Guardrails",
            "",
            "- This is one tiny pinned reader on a frozen synthetic benchmark, not a general long-context leaderboard claim.",
            "- Retrieval evidence completeness and reader answer quality are recorded separately; a model failure is not credited to routing, and a retrieval miss is not hidden by generation.",
            "- Raw generated answers are scored as emitted. No expected-answer-aware cleanup or post-hoc extraction is added.",
            "- The explicit router remains deterministic and qrel-blind; pretrained outputs do not influence its decisions.",
            "",
        ]
    )
    return "\n".join(lines)
