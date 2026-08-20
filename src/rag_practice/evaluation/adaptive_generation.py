from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean

from rag_practice.adaptive.generation import AdaptiveAnswerTrace
from rag_practice.adaptive.router import Route
from rag_practice.evaluation.rag import answer_contains_reference, grounded_token_recall, token_f1


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def is_refusal(answer: str) -> bool:
    normalized = " ".join(answer.lower().replace("'", " ").split())
    markers = (
        "i do not know",
        "i don t know",
        "cannot determine",
        "can not determine",
        "insufficient context",
        "not enough information",
    )
    return any(marker in normalized for marker in markers)


def evaluate_adaptive_answers(
    rows: Sequence[Mapping[str, object]],
    traces: Mapping[str, AdaptiveAnswerTrace],
    *,
    primary_documents: Mapping[str, str],
    fallback_documents: Mapping[str, str],
) -> dict[str, float]:
    if not rows:
        return {}

    token_f1_values: list[float] = []
    contains_values: list[float] = []
    grounded_values: list[float] = []
    evidence_complete_values: list[float] = []
    route_hits: list[float] = []
    total_calls: list[float] = []
    active_calls: list[float] = []
    attempts: list[float] = []
    generation_ms: list[float] = []
    prompt_words: list[float] = []
    output_words: list[float] = []
    unnecessary_retrieval: list[float] = []
    answerable_refusals: list[float] = []
    unanswerable_refusals: list[float] = []

    for row in rows:
        query_id = str(row["id"])
        trace = traces[query_id]
        expected_route = Route(str(row["route"]))
        answerable = bool(row.get("answerable", True))
        predicted_refusal = trace.refused or is_refusal(trace.final_answer)

        route_hits.append(float(trace.control.route == expected_route))
        total_calls.append(float(trace.total_retrieval_calls))
        active_calls.append(float(trace.active_retrieval_calls))
        attempts.append(float(len(trace.attempts)))
        generation_ms.append(trace.total_generation_ms)
        prompt_words.append(float(trace.total_prompt_words))
        output_words.append(float(trace.total_output_words))

        if expected_route == Route.NO_RETRIEVAL:
            unnecessary_retrieval.append(float(trace.total_retrieval_calls > 0))

        relevant = {str(item) for item in row.get("relevant_document_ids", [])}
        if relevant:
            evidence_complete_values.append(float(relevant.issubset(set(trace.final_context_ids))))

        if answerable:
            reference = str(row["reference"])
            token_f1_values.append(token_f1(trace.final_answer, reference))
            contains_values.append(answer_contains_reference(trace.final_answer, reference))
            answerable_refusals.append(float(predicted_refusal))
            if expected_route != Route.NO_RETRIEVAL:
                contexts = [
                    primary_documents[document_id]
                    if document_id in primary_documents
                    else fallback_documents[document_id]
                    for document_id in trace.final_context_ids
                ]
                grounded_values.append(grounded_token_recall(trace.final_answer, contexts))
        else:
            unanswerable_refusals.append(float(predicted_refusal))

    return {
        "answer_token_f1": _mean(token_f1_values),
        "answer_contains_reference": _mean(contains_values),
        "grounded_token_recall_retrieval_queries": _mean(grounded_values),
        "final_evidence_complete": _mean(evidence_complete_values),
        "route_accuracy": _mean(route_hits),
        "mean_total_retrieval_calls": _mean(total_calls),
        "mean_active_retrieval_calls": _mean(active_calls),
        "mean_generation_attempts": _mean(attempts),
        "mean_generation_ms": _mean(generation_ms),
        "mean_prompt_words": _mean(prompt_words),
        "mean_output_words": _mean(output_words),
        "unnecessary_retrieval_rate": _mean(unnecessary_retrieval),
        "answerable_refusal_rate": _mean(answerable_refusals),
        "unanswerable_refusal_recall": _mean(unanswerable_refusals),
        "unsupported_answer_rate": 1.0 - _mean(unanswerable_refusals),
    }
