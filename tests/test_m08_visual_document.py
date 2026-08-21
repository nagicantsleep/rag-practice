from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_practice.evaluation.visual_document import evaluate_visual_document_system
from rag_practice.visual_document import VisualDocumentIndex

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "m08_visual_document"


def _queries() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (BENCHMARK / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _metrics(index: VisualDocumentIndex, system: str) -> dict[str, float]:
    if system == "ocr_surrogate":
        retrieve = index.retrieve_ocr
        allow_text, allow_pixels = True, False
    elif system == "page_native":
        retrieve = index.retrieve_pages
        allow_text, allow_pixels = False, True
    elif system == "ocr_page_fusion":
        retrieve = index.retrieve_hybrid
        allow_text, allow_pixels = True, True
    else:
        raise ValueError(system)

    rankings: dict[str, list[str]] = {}
    answers: dict[str, str] = {}
    modalities: dict[str, list[str]] = {}
    candidates: dict[str, int] = {}
    regions: dict[str, str | None] = {}
    queries = _queries()
    for query in queries:
        qid = str(query["id"])
        text = str(query["query"])
        result = retrieve(text, k=3)
        ranking = [item[0] for item in result.ranking]
        top = ranking[0] if ranking else None
        rankings[qid] = ranking
        answers[qid] = index.answer(
            text,
            result,
            allow_text=allow_text,
            allow_pixels=allow_pixels,
        )
        modalities[qid] = list(result.evidence_modalities)
        candidates[qid] = result.visual_candidates_scored
        regions[qid] = result.region_for(top) if top else None
    return evaluate_visual_document_system(
        queries,
        rankings=rankings,
        answers=answers,
        evidence_modalities=modalities,
        visual_candidates_scored=candidates,
        regions=regions,
    )


def test_frozen_visual_document_benchmark_shape_and_rasters() -> None:
    index = VisualDocumentIndex(BENCHMARK)
    assert len(index.assets) == 6
    assert len(_queries()) == 10
    assert set(index.page_payloads) == {asset.file_name for asset in index.assets}
    assert {image.size for image in index.images.values()} == {(128, 176)}
    assert all(image.mode == "RGB" for image in index.images.values())
    assert index.raster_source_bytes > 0
    assert index.raster_rgb_bytes == 6 * 128 * 176 * 3


def test_ocr_can_recover_table_text_without_visual_grounding() -> None:
    index = VisualDocumentIndex(BENCHMARK)
    query = next(item for item in _queries() if item["id"] == "v5")
    result = index.retrieve_ocr(str(query["query"]), k=3)
    assert result.ranking[0][0] == "alpha_p1"
    assert result.evidence_modalities == ("ocr_text",)
    assert result.region_for("alpha_p1") is None
    assert index.answer(
        str(query["query"]), result, allow_text=True, allow_pixels=False
    ) == "150"


def test_page_native_returns_region_provenance() -> None:
    index = VisualDocumentIndex(BENCHMARK)
    query = next(item for item in _queries() if item["id"] == "v8")
    result = index.retrieve_pages(str(query["query"]), k=3)
    assert result.ranking[0][0] == "gamma_p2"
    assert result.evidence_modalities == ("page_image",)
    assert result.region_for("gamma_p2") == "lower-right"
    assert index.by_id["gamma_p2"].region_locator("lower-right").endswith("#lower-right")


def test_controlled_fusion_satisfies_frozen_benchmark() -> None:
    index = VisualDocumentIndex(BENCHMARK)
    metrics = _metrics(index, "ocr_page_fusion")
    for key in (
        "recall@3",
        "hit_rate@1",
        "visual_required_hit@1",
        "cross_modal_hit@1",
        "text_sufficient_hit@1",
        "no_evidence_accuracy",
        "answer_correct_rate",
        "visual_evidence_grounded_rate",
        "region_locator_accuracy",
    ):
        assert metrics[key] == pytest.approx(1.0)
    assert metrics["mean_visual_candidates_scored"] == pytest.approx(3.2)


def test_controls_preserve_expected_evidence_failures() -> None:
    index = VisualDocumentIndex(BENCHMARK)
    ocr = _metrics(index, "ocr_surrogate")
    page = _metrics(index, "page_native")

    assert ocr["recall@3"] == pytest.approx(0.875)
    assert ocr["visual_evidence_grounded_rate"] == pytest.approx(0.0)
    assert ocr["region_locator_accuracy"] == pytest.approx(0.0)
    assert ocr["no_evidence_accuracy"] == pytest.approx(0.5)
    assert ocr["answer_correct_rate"] == pytest.approx(0.4)

    assert page["recall@3"] == pytest.approx(0.75)
    assert page["text_sufficient_hit@1"] == pytest.approx(0.0)
    assert page["visual_evidence_grounded_rate"] == pytest.approx(1.0)
    assert page["region_locator_accuracy"] == pytest.approx(1.0)
    assert page["no_evidence_accuracy"] == pytest.approx(1.0)
    assert page["answer_correct_rate"] == pytest.approx(0.7)
