from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter_ns

from rag_practice.evaluation.visual_document import evaluate_visual_document_system
from rag_practice.visual_document import (
    BASE_MODEL_NAME,
    BASE_MODEL_REVISION,
    ColSmolPageRetriever,
    MODEL_NAME,
    MODEL_REVISION,
    VisualDocumentIndex,
)

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "m08_visual_document"
RESULTS = Path(__file__).resolve().parent / "results"


def load_queries() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (BENCHMARK / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    index = VisualDocumentIndex(BENCHMARK)
    queries = load_queries()
    retriever = ColSmolPageRetriever(index)

    rankings: dict[str, list[str]] = {}
    answers: dict[str, str] = {}
    modalities: dict[str, list[str]] = {}
    visual_candidates: dict[str, int] = {}
    regions: dict[str, str | None] = {}
    per_query: list[dict[str, object]] = []

    for query in queries:
        qid = str(query["id"])
        text = str(query["query"])
        started = perf_counter_ns()
        result = retriever.retrieve(text, k=3)
        latency_ms = (perf_counter_ns() - started) / 1_000_000
        ranking = [page_id for page_id, _ in result.ranking]
        scores = [float(score) for _, score in result.ranking]
        top_id = ranking[0] if ranking else None

        # The pretrained control is retrieval-only. Page identity answers can follow
        # directly from the retrieved page for visual page-selection tasks, but the
        # evaluator never supplies frozen OCR text or deterministic region features.
        answer = index.answer(
            text,
            result,
            allow_text=False,
            allow_pixels=True,
        )
        rankings[qid] = ranking
        answers[qid] = answer
        modalities[qid] = list(result.evidence_modalities)
        visual_candidates[qid] = result.visual_candidates_scored
        regions[qid] = None
        top_asset = index.by_id[top_id] if top_id else None
        per_query.append(
            {
                "id": qid,
                "task": query["task"],
                "query": text,
                "relevant": query["relevant"],
                "ranking": ranking,
                "scores": scores,
                "top_page_locator": top_asset.locator if top_asset else None,
                "top_region": None,
                "top_region_locator": None,
                "answer": answer,
                "expected_answer": query["answer"],
                "answer_correct": answer == query["answer"],
                "evidence_modalities": list(result.evidence_modalities),
                "visual_candidates_scored": result.visual_candidates_scored,
                "latency_ms": latency_ms,
            }
        )

    metrics = evaluate_visual_document_system(
        queries,
        rankings=rankings,
        answers=answers,
        evidence_modalities=modalities,
        visual_candidates_scored=visual_candidates,
        regions=regions,
    )
    metrics["mean_query_ms"] = sum(float(item["latency_ms"]) for item in per_query) / len(per_query)

    result: dict[str, object] = {
        "control": "pinned pretrained ColSmol text-to-page-image retrieval",
        "guardrail": (
            "ranking receives query text and rendered page pixels only; frozen OCR text, titles, "
            "document ids, qrels, expected answers, deterministic visual features, and region "
            "heuristics are excluded from pretrained ranking"
        ),
        "model": {
            "adapter_name": MODEL_NAME,
            "adapter_revision": MODEL_REVISION,
            "base_name": BASE_MODEL_NAME,
            "base_revision": BASE_MODEL_REVISION,
            "device": "cpu",
            "dtype": "float32",
            "model_load_ms": retriever.model_load_ms,
            "index_build_ms": retriever.index_build_ms,
            "embedding_shape": retriever.embedding_shape,
            "embedding_bytes": retriever.embedding_bytes,
        },
        "benchmark": {"pages": len(index.assets), "queries": len(queries)},
        "metrics": metrics,
        "per_query": per_query,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "colsmol_results.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# M08.6 pinned ColSmol page-image retrieval results",
        "",
        f"Adapter: `{MODEL_NAME}` pinned to `{MODEL_REVISION}`.",
        f"Base: `{BASE_MODEL_NAME}` pinned to `{BASE_MODEL_REVISION}`.",
        "",
        "Ranking uses query text + rendered page pixels only. OCR text, page titles/document ids, qrels, expected answers, deterministic visual markers, and region heuristics are excluded from ranking.",
        "",
        "| Recall@3 | Hit@1 | Visual Hit@1 | Cross-modal Hit@1 | Text Hit@1 | No-evidence | Answer correct | Visual grounded | Region locator | Visual candidates |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {metrics['recall@3']:.3f} | {metrics['hit_rate@1']:.3f} | "
            f"{metrics['visual_required_hit@1']:.3f} | {metrics['cross_modal_hit@1']:.3f} | "
            f"{metrics['text_sufficient_hit@1']:.3f} | {metrics['no_evidence_accuracy']:.3f} | "
            f"{metrics['answer_correct_rate']:.3f} | {metrics['visual_evidence_grounded_rate']:.3f} | "
            f"{metrics['region_locator_accuracy']:.3f} | {metrics['mean_visual_candidates_scored']:.1f} |"
        ),
        "",
        "## Runtime / representation",
        "",
        f"- model load: {retriever.model_load_ms:.2f} ms",
        f"- page index build: {retriever.index_build_ms:.2f} ms",
        f"- embedding shape: `{retriever.embedding_shape}`",
        f"- embedding bytes: {retriever.embedding_bytes}",
        f"- mean query latency: {metrics['mean_query_ms']:.2f} ms",
        "",
        "## Guardrails",
        "",
        "- This is a frozen tiny synthetic benchmark, not a general visual-document leaderboard claim.",
        "- Both the adapter and its full-weight base are pinned because the adapter's upstream base default revision is mutable.",
        "- The retriever is exhaustive and has no abstention policy; no-evidence errors are retained.",
        "- Region locator accuracy is intentionally zero unless the pretrained retrieval control itself exposes region provenance; no deterministic region heuristic is added after ranking.",
        "- Text/table value questions are not answered from frozen OCR after pretrained retrieval; retrieval quality and answer capability stay separate.",
    ]
    markdown = "\n".join(lines) + "\n"
    (RESULTS / "colsmol_results.md").write_text(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
