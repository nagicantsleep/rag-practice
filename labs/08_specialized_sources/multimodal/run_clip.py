from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter_ns

from rag_practice.evaluation.multimodal import evaluate_multimodal_system
from rag_practice.multimodal import CLIPTextToImageRetriever, MultimodalImageIndex, RetrievalResult

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "m08_multimodal"
RESULTS = Path(__file__).resolve().parent / "results"
MODEL_NAME = "openai/clip-vit-base-patch32"
MODEL_REVISION = "b97b0100e55e367c057773c2a614676470b0d575"


def load_queries():
    return [json.loads(line) for line in (BENCHMARK / "queries.jsonl").read_text().splitlines() if line.strip()]


def render_markdown(result):
    metrics = result["metrics"]
    lines = [
        "# M08.5 pretrained CLIP retrieval control",
        "",
        f"Model: `{result['model']['name']}` @ `{result['model']['revision']}`",
        "",
        "The CLIP control receives only query text and image pixels for ranking. Titles, captions, site metadata, qrels, and answer labels are not exposed to the retriever.",
        "",
        "| Recall@3 | Hit@1 | Visual Hit@1 | Cross-modal Hit@1 | Text Hit@1 | No-evidence | Answer correct | Visual grounded | Visual candidates |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {metrics['recall@3']:.3f} | {metrics['hit_rate@1']:.3f} | {metrics['visual_required_hit@1']:.3f} | {metrics['cross_modal_hit@1']:.3f} | {metrics['text_sufficient_hit@1']:.3f} | {metrics['no_evidence_accuracy']:.3f} | {metrics['answer_correct_rate']:.3f} | {metrics['visual_evidence_grounded_rate']:.3f} | {metrics['mean_visual_candidates_scored']:.1f} |",
        "",
        "## Interpretation guardrails",
        "",
        "- This is an exhaustive text-to-image CLIP retrieval control, not a multimodal fusion system.",
        "- Image embeddings are built from the frozen P3 raster pixels only; captions and metadata are excluded.",
        "- The existing deterministic pixel reader is used only after retrieval to score answer correctness separately from retrieval quality.",
        "- CLIP always returns ranked images here, so no-evidence accuracy tests whether a retrieval-only control can abstain without an explicit rejection mechanism.",
        "- The benchmark is tiny and synthetic; the result is retained whether it helps or hurts relative to the handcrafted controls.",
    ]
    return "\n".join(lines) + "\n"


def main():
    index = MultimodalImageIndex(BENCHMARK)
    image_paths = {asset.id: BENCHMARK / "images" / asset.file_name for asset in index.assets}
    clip = CLIPTextToImageRetriever(
        image_paths,
        model_name=MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
    )
    queries = load_queries()
    rankings = {}
    answers = {}
    modalities = {}
    visual_candidates = {}
    per_query = []

    for query in queries:
        qid = str(query["id"])
        started = perf_counter_ns()
        ranking = clip.search(str(query["query"]), k=3)
        latency_ms = (perf_counter_ns() - started) / 1_000_000
        result = RetrievalResult(tuple(ranking), ("image",), len(index.assets))
        answer = index.answer(str(query["query"]), result, allow_text=False, allow_pixels=True)
        ranked_ids = [item[0] for item in ranking]
        rankings[qid] = ranked_ids
        answers[qid] = answer
        modalities[qid] = list(result.evidence_modalities)
        visual_candidates[qid] = result.visual_candidates_scored
        per_query.append(
            {
                "id": qid,
                "task": query["task"],
                "query": query["query"],
                "relevant": query["relevant"],
                "ranking": ranked_ids,
                "scores": [item[1] for item in ranking],
                "top_locator": index.by_id[ranked_ids[0]].locator if ranked_ids else None,
                "answer": answer,
                "expected_answer": query["answer"],
                "answer_correct": answer == query["answer"],
                "evidence_modalities": list(result.evidence_modalities),
                "visual_candidates_scored": result.visual_candidates_scored,
                "latency_ms": latency_ms,
            }
        )

    metrics = evaluate_multimodal_system(
        queries,
        rankings=rankings,
        answers=answers,
        evidence_modalities=modalities,
        visual_candidates_scored=visual_candidates,
    )
    metrics["mean_query_ms"] = sum(float(item["latency_ms"]) for item in per_query) / len(per_query)
    result = {
        "hypothesis": "a real pretrained aligned text-image representation may recover visual semantics but cannot rely on hidden captions or metadata and may still fail identity/abstention on the frozen toy benchmark",
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "device": "cpu",
            "embedding_dimensions": clip.dimensions,
            "model_load_ms": clip.model_load_ms,
            "index_build_ms": clip.index_build_ms,
            "logical_index_bytes": clip.logical_index_bytes(),
        },
        "benchmark": {"images": len(index.assets), "queries": len(queries)},
        "metrics": metrics,
        "per_query": per_query,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "clip_results.json").write_text(json.dumps(result, indent=2) + "\n")
    markdown = render_markdown(result)
    (RESULTS / "clip_results.md").write_text(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
