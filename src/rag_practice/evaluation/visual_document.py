"""Evaluation for visual-document/page-image retrieval and grounding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_VISUAL_TASKS = {"visual_layout", "table_layout", "chart_layout", "cross_modal", "region_layout"}


def evaluate_visual_document_system(
    queries: Sequence[Mapping[str, object]],
    *,
    rankings: Mapping[str, Sequence[str]],
    answers: Mapping[str, str],
    evidence_modalities: Mapping[str, Sequence[str]],
    visual_candidates_scored: Mapping[str, int],
    regions: Mapping[str, str | None],
) -> dict[str, float]:
    retrieval_recalls: list[float] = []
    hit_at_1: list[float] = []
    visual_hit_at_1: list[float] = []
    cross_modal_hit_at_1: list[float] = []
    text_hit_at_1: list[float] = []
    no_evidence: list[float] = []
    answer_correct: list[float] = []
    visual_grounded: list[float] = []
    region_accuracy: list[float] = []

    for query in queries:
        query_id = str(query["id"])
        task = str(query["task"])
        relevant = {str(item) for item in query["relevant"]}
        ranking = list(rankings.get(query_id, ()))
        top = ranking[0] if ranking else None

        if relevant:
            retrieval_recalls.append(len(relevant.intersection(ranking[:3])) / len(relevant))
            hit = float(top in relevant)
            hit_at_1.append(hit)
            if task in _VISUAL_TASKS:
                visual_hit_at_1.append(hit)
                modalities = set(evidence_modalities.get(query_id, ()))
                visual_grounded.append(float(hit == 1.0 and "page_image" in modalities))
            if task == "cross_modal":
                cross_modal_hit_at_1.append(hit)
            if task == "text_sufficient":
                text_hit_at_1.append(hit)
        else:
            no_evidence.append(float(not ranking))

        answer_correct.append(float(answers.get(query_id) == str(query["answer"])))
        expected_region = query.get("region")
        if expected_region is not None:
            region_accuracy.append(
                float(top in relevant and regions.get(query_id) == str(expected_region))
            )

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "recall@3": mean(retrieval_recalls),
        "hit_rate@1": mean(hit_at_1),
        "visual_required_hit@1": mean(visual_hit_at_1),
        "cross_modal_hit@1": mean(cross_modal_hit_at_1),
        "text_sufficient_hit@1": mean(text_hit_at_1),
        "no_evidence_accuracy": mean(no_evidence),
        "answer_correct_rate": mean(answer_correct),
        "visual_evidence_grounded_rate": mean(visual_grounded),
        "region_locator_accuracy": mean(region_accuracy),
        "mean_visual_candidates_scored": mean(
            [float(visual_candidates_scored.get(str(query["id"]), 0)) for query in queries]
        ),
    }
