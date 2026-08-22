from pathlib import Path

from rag_practice.calibration.core import (
    FEATURE_NAMES,
    LogisticCalibrator,
    baseline_confidences,
    build_runtime_trace,
    load_benchmark,
)
from rag_practice.evaluation.calibration import evaluate_calibration, raw_correctness


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "m12_calibration"


def _case(benchmark, case_id):
    return next(case for case in benchmark.cases if case.id == case_id)


def test_m12_benchmark_split_sizes_and_entity_isolation() -> None:
    benchmark = load_benchmark(DATA / "benchmark.json")
    assert len(benchmark.cases_for("train")) == 12
    assert len(benchmark.cases_for("calibration")) == 10
    assert len(benchmark.cases_for("test_id")) == 10
    assert len(benchmark.cases_for("test_ood")) == 8
    entities = {
        split: {case.entity_id for case in benchmark.cases_for(split)}
        for split in ("train", "calibration", "test_id", "test_ood")
    }
    for left, left_entities in entities.items():
        for right, right_entities in entities.items():
            if left < right:
                assert left_entities.isdisjoint(right_entities)


def test_m12_runtime_feature_vector_is_frozen_width() -> None:
    benchmark = load_benchmark(DATA / "benchmark.json")
    case = _case(benchmark, "q01")
    trace = build_runtime_trace(case.id, case.entity_id, case.question, benchmark.documents)
    assert len(trace.features) == len(FEATURE_NAMES) == 10
    assert tuple(trace.feature_dict()) == FEATURE_NAMES


def test_m12_direct_train_case_is_answered_from_valid_top1() -> None:
    benchmark = load_benchmark(DATA / "benchmark.json")
    case = _case(benchmark, "q01")
    trace = build_runtime_trace(case.id, case.entity_id, case.question, benchmark.documents)
    assert trace.answer == "BLUE"
    assert trace.evidence_ids[0] == "trn01-a"
    assert trace.feature_dict()["top1_valid"] == 1.0
    assert raw_correctness(case, trace) == 1


def test_m12_train_near_miss_is_confident_but_wrong() -> None:
    benchmark = load_benchmark(DATA / "benchmark.json")
    case = _case(benchmark, "q07")
    trace = build_runtime_trace(case.id, case.entity_id, case.question, benchmark.documents)
    assert trace.evidence_ids[0] == "trn07-a"
    assert trace.answer == "WRONG07"
    assert raw_correctness(case, trace) == 0
    assert baseline_confidences(trace)["top1"] > 0.0


def test_m12_train_stale_case_exposes_invalid_top1_signal() -> None:
    benchmark = load_benchmark(DATA / "benchmark.json")
    case = _case(benchmark, "q10")
    trace = build_runtime_trace(case.id, case.entity_id, case.question, benchmark.documents)
    assert trace.evidence_ids[0] == "trn10-a"
    assert trace.feature_dict()["top1_valid"] == 0.0
    assert raw_correctness(case, trace) == 0


def test_m12_train_conflict_signal_detects_distinct_valid_answers() -> None:
    benchmark = load_benchmark(DATA / "benchmark.json")
    case = _case(benchmark, "q11")
    trace = build_runtime_trace(case.id, case.entity_id, case.question, benchmark.documents)
    assert trace.feature_dict()["conflict_signal"] == 1.0
    assert raw_correctness(case, trace) == 0


def test_m12_logistic_fit_is_deterministic() -> None:
    benchmark = load_benchmark(DATA / "benchmark.json")
    rows = []
    for case in benchmark.cases_for("train"):
        trace = build_runtime_trace(case.id, case.entity_id, case.question, benchmark.documents)
        rows.append((trace.features, raw_correctness(case, trace)))
    first = LogisticCalibrator.fit(rows)
    second = LogisticCalibrator.fit(rows)
    assert first == second
    assert len(first.weights) == 10


def test_m12_evaluator_emits_frozen_methods_and_test_splits() -> None:
    results = evaluate_calibration(DATA)
    assert set(results["methods"]) == {"constant", "top1", "margin", "hand_composed", "logistic"}
    assert set(results["metrics"]) == {"test_id", "test_ood"}
    expected_sizes = {"test_id": 10, "test_ood": 8}
    for split_name, split in results["metrics"].items():
        for metrics in split.values():
            assert 0.0 <= metrics["brier"] <= 1.0
            assert 0.0 <= metrics["ece"] <= 1.0
            assert 0.0 <= metrics["coverage"] <= 1.0
            assert 0.0 <= metrics["selective_risk"] <= 1.0
            curve = metrics["risk_coverage_curve"]
            assert len(curve) == expected_sizes[split_name]
            assert curve[0]["coverage"] == 1 / expected_sizes[split_name]
            assert curve[-1]["coverage"] == 1.0
    assert results["timing"]["mean_trace_feature_ms"] >= 0.0
    assert results["timing"]["logistic_fit_ms"] >= 0.0
    assert results["timing"]["mean_logistic_predict_ms"] >= 0.0
    assert results["timing"]["model_calls"] == 0.0
