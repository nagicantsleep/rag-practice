from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter_ns

from rag_practice.evaluation.visual_document import evaluate_visual_document_system
from rag_practice.visual_document import VisualDocumentIndex

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "m08_visual_document"
RESULTS = Path(__file__).resolve().parent / "results"


def load_queries() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (BENCHMARK / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]


def evaluate_system(
    index: VisualDocumentIndex,
    queries: list[dict[str, object]],
    *,
    name: str,
) -> dict[str, object]:
    if name == "ocr_surrogate":
        retrieve = index.retrieve_ocr
        allow_text, allow_pixels = True, False
    elif name == "page_native":
        retrieve = index.retrieve_pages
        allow_text, allow_pixels = False, True
    elif name == "ocr_page_fusion":
        retrieve = index.retrieve_hybrid
        allow_text, allow_pixels = True, True
    else:
        raise ValueError(name)

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
        result = retrieve(text, k=3)
        latency_ms = (perf_counter_ns() - started) / 1_000_000
        ranking = [item[0] for item in result.ranking]
        top_id = ranking[0] if ranking else None
        top_region = result.region_for(top_id) if top_id else None
        answer = index.answer(
            text,
            result,
            allow_text=allow_text,
            allow_pixels=allow_pixels,
        )
        rankings[qid] = ranking
        answers[qid] = answer
        modalities[qid] = list(result.evidence_modalities)
        visual_candidates[qid] = result.visual_candidates_scored
        regions[qid] = top_region
        top_asset = index.by_id[top_id] if top_id else None
        per_query.append(
            {
                "id": qid,
                "task": query["task"],
                "query": text,
                "relevant": query["relevant"],
                "ranking": ranking,
                "top_page_locator": top_asset.locator if top_asset else None,
                "top_region": top_region,
                "top_region_locator": (
                    top_asset.region_locator(top_region) if top_asset and top_region else None
                ),
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
    return {"metrics": metrics, "per_query": per_query}


def render_markdown(result: dict[str, object]) -> str:
    systems = result["systems"]
    assert isinstance(systems, dict)
    lines = [
        "# M08.6 Visual-document / page-image RAG results",
        "",
        "Benchmark: 6 frozen synthetic document pages, 10 queries (text, layout, table, chart, cross-modal, region, no-evidence).",
        "",
        "| System | Recall@3 | Hit@1 | Visual Hit@1 | Cross-modal Hit@1 | Text Hit@1 | No-evidence | Answer correct | Visual grounded | Region locator | Visual candidates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("ocr_surrogate", "page_native", "ocr_page_fusion"):
        metrics = systems[name]["metrics"]
        lines.append(
            f"| {name} | {metrics['recall@3']:.3f} | {metrics['hit_rate@1']:.3f} | "
            f"{metrics['visual_required_hit@1']:.3f} | {metrics['cross_modal_hit@1']:.3f} | "
            f"{metrics['text_sufficient_hit@1']:.3f} | {metrics['no_evidence_accuracy']:.3f} | "
            f"{metrics['answer_correct_rate']:.3f} | {metrics['visual_evidence_grounded_rate']:.3f} | "
            f"{metrics['region_locator_accuracy']:.3f} | {metrics['mean_visual_candidates_scored']:.1f} |"
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- OCR surrogate retrieval only sees frozen text extraction; a page hit or correct short answer is not visual/layout evidence.",
        "- Page-native retrieval sees exact raster pixels and layout markers but deliberately cannot read frozen OCR facts; it is a deterministic mechanism control, not a learned document model.",
        "- OCR+page fusion keeps both modalities explicit and records page plus region locators rather than collapsing raster evidence into OCR text.",
        "- No-evidence behavior is evaluated separately; visual similarity or lexical overlap is not automatically an abstention policy.",
        "- The benchmark is tiny and synthetic. Perfect fusion scores demonstrate the controlled evidence mechanism only.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    index = VisualDocumentIndex(BENCHMARK)
    queries = load_queries()
    result: dict[str, object] = {
        "hypothesis": (
            "OCR and page-raster evidence are complementary: OCR can recover text facts without "
            "grounding layout claims, while image-only retrieval can recover layout but cannot "
            "substitute for text extraction or abstention"
        ),
        "benchmark": {
            "pages": len(index.assets),
            "queries": len(queries),
            "ocr_chars_indexed": index.ocr_chars_indexed,
            "raster_source_bytes": index.raster_source_bytes,
            "raster_rgb_bytes": index.raster_rgb_bytes,
        },
        "systems": {},
    }
    systems = result["systems"]
    assert isinstance(systems, dict)
    for name in ("ocr_surrogate", "page_native", "ocr_page_fusion"):
        systems[name] = evaluate_system(index, queries, name=name)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    markdown = render_markdown(result)
    (RESULTS / "results.md").write_text(markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
