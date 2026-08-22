"""Evaluation for M12 calibrated and drift-aware RAG."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from rag_practice.calibration.core import (
    FEATURE_NAMES,
    BenchmarkCase,
    LogisticCalibrator,
    RuntimeTrace,
    baseline_confidences,
    build_runtime_trace,
    load_benchmark,
)


METHODS = ("constant", "top1", "margin", "hand_composed", "logistic")
THRESHOLDS = tuple(round(index * 0.05, 2) for index in range(21))
ECE_BINS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0000001))


def raw_correctness(case: BenchmarkCase, trace: RuntimeTrace) -> int:
    if not case.answerable:
        return 0
    if trace.answer != case.expected_answer:
        return 0
    evidence = set(trace.evidence_ids)
    if not set(case.required_evidence_ids) <= evidence:
        return 0
    if trace.retrieved and trace.retrieved[0].document.id in set(case.forbidden_evidence_ids):
        return 0
    conflict = bool(trace.feature_dict()["conflict_signal"])
    if conflict and not (case.expected_answer == "CONFLICT" and trace.answer == "CONFLICT"):
        return 0
    return 1


def _build_rows(data_root: Path) -> tuple[list[dict[str, Any]], LogisticCalibrator]:
    benchmark = load_benchmark(data_root / "benchmark.json")
    traces: dict[str, RuntimeTrace] = {}
    cases = {case.id: case for case in benchmark.cases}
    for case in benchmark.cases:
        traces[case.id] = build_runtime_trace(case.id, case.entity_id, case.question, benchmark.documents)

    train_rows = [
        (traces[case.id].features, raw_correctness(case, traces[case.id]))
        for case in benchmark.cases_for("train")
    ]
    calibrator = LogisticCalibrator.fit(train_rows)

    rows: list[dict[str, Any]] = []
    for case in benchmark.cases:
        trace = traces[case.id]
        correctness = raw_correctness(case, trace)
        confidences = baseline_confidences(trace)
        confidences["logistic"] = calibrator.predict(trace.features)
        rows.append(
            {
                "id": case.id,
                "entity_id": case.entity_id,
                "question": case.question,
                "split": case.split,
                "class": case.scenario,
                "shift_class": case.shift_class,
                "answerable": case.answerable,
                "expected_answer": case.expected_answer,
                "required_evidence_ids": list(case.required_evidence_ids),
                "forbidden_evidence_ids": list(case.forbidden_evidence_ids),
                "answer": trace.answer,
                "correct": correctness,
                "evidence_ids": list(trace.evidence_ids),
                "retrieved": [
                    {
                        "id": item.document.id,
                        "entity_id": item.document.entity_id,
                        "score": item.score,
                        "active": item.document.active,
                        "trusted": item.document.trusted,
                    }
                    for item in trace.retrieved
                ],
                "features": trace.feature_dict(),
                "confidences": confidences,
            }
        )
    return rows, calibrator


def _choose_threshold(rows: list[dict[str, Any]], method: str) -> float:
    candidates: list[tuple[float, float, float]] = []
    for threshold in THRESHOLDS:
        answered = [row for row in rows if row["confidences"][method] >= threshold]
        coverage = len(answered) / len(rows)
        if coverage < 0.60:
            continue
        risk = (sum(1 - row["correct"] for row in answered) / len(answered)) if answered else 0.0
        candidates.append((risk, -coverage, -threshold))
    if not candidates:
        raise AssertionError("threshold=0 must satisfy the frozen coverage constraint")
    best = min(candidates)
    return -best[2]


def _reliability(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    output = []
    for index, (lower, upper) in enumerate(ECE_BINS):
        if index < len(ECE_BINS) - 1:
            selected = [row for row in rows if lower <= row["confidences"][method] < upper]
        else:
            selected = [row for row in rows if lower <= row["confidences"][method] <= 1.0]
        output.append(
            {
                "lower": lower,
                "upper": 1.0 if index == len(ECE_BINS) - 1 else upper,
                "count": len(selected),
                "mean_confidence": mean(row["confidences"][method] for row in selected) if selected else 0.0,
                "empirical_correctness": mean(row["correct"] for row in selected) if selected else 0.0,
            }
        )
    return output


def _aurc_and_targets(rows: list[dict[str, Any]], method: str) -> tuple[float, dict[str, float]]:
    ordered = sorted(rows, key=lambda row: (-row["confidences"][method], row["id"]))
    prefix_risks: list[float] = []
    errors = 0
    for index, row in enumerate(ordered, start=1):
        errors += 1 - row["correct"]
        prefix_risks.append(errors / index)
    targets: dict[str, float] = {}
    n = len(ordered)
    for target in (0.50, 0.70, 0.90):
        k = max(1, math.ceil(target * n))
        targets[f"risk_at_{target:.2f}_coverage"] = prefix_risks[k - 1]
    return mean(prefix_risks), targets


def _method_metrics(rows: list[dict[str, Any]], method: str, threshold: float) -> dict[str, Any]:
    probabilities = [row["confidences"][method] for row in rows]
    correctness = [row["correct"] for row in rows]
    brier = mean((probability - target) ** 2 for probability, target in zip(probabilities, correctness, strict=True))
    log_loss = mean(
        -(target * math.log(min(1 - 1e-6, max(1e-6, probability)))
          + (1 - target) * math.log(min(1 - 1e-6, max(1e-6, 1 - probability))))
        for probability, target in zip(probabilities, correctness, strict=True)
    )
    reliability = _reliability(rows, method)
    ece = sum(
        (bucket["count"] / len(rows))
        * abs(bucket["mean_confidence"] - bucket["empirical_correctness"])
        for bucket in reliability
    )
    answered = [row for row in rows if row["confidences"][method] >= threshold]
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    wrong_answered = sum(1 - row["correct"] for row in answered)
    false_abstentions = sum(
        row["answerable"] and row["confidences"][method] < threshold for row in rows
    )
    correct_conf = [row["confidences"][method] for row in rows if row["correct"]]
    incorrect_conf = [row["confidences"][method] for row in rows if not row["correct"]]
    aurc, targets = _aurc_and_targets(rows, method)
    metrics = {
        "threshold": threshold,
        "full_coverage_accuracy": mean(correctness),
        "brier": brier,
        "log_loss": log_loss,
        "ece": ece,
        "mean_confidence": mean(probabilities),
        "mean_confidence_correct": mean(correct_conf) if correct_conf else 0.0,
        "mean_confidence_incorrect": mean(incorrect_conf) if incorrect_conf else 0.0,
        "coverage": len(answered) / len(rows),
        "selective_risk": wrong_answered / len(answered) if answered else 0.0,
        "false_answer_rate": wrong_answered / len(rows),
        "false_abstention_rate": false_abstentions / len(answerable) if answerable else 0.0,
        "abstention_accuracy": (
            sum(row["confidences"][method] < threshold for row in unanswerable) / len(unanswerable)
            if unanswerable
            else 1.0
        ),
        "aurc": aurc,
        "reliability": reliability,
    }
    metrics.update(targets)
    return metrics


def evaluate_calibration(data_root: str | Path) -> dict[str, Any]:
    root = Path(data_root)
    raw = json.loads((root / "benchmark.json").read_text())
    rows, calibrator = _build_rows(root)
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "calibration", "test_id", "test_ood")
    }
    thresholds = {
        method: _choose_threshold(by_split["calibration"], method)
        for method in METHODS
    }
    metrics: dict[str, dict[str, Any]] = {}
    for split in ("test_id", "test_ood"):
        metrics[split] = {
            method: _method_metrics(by_split[split], method, thresholds[method])
            for method in METHODS
        }
    drift = {}
    for method in METHODS:
        id_metrics = metrics["test_id"][method]
        ood_metrics = metrics["test_ood"][method]
        drift[method] = {
            name: ood_metrics[name] - id_metrics[name]
            for name in (
                "full_coverage_accuracy",
                "brier",
                "ece",
                "aurc",
                "coverage",
                "selective_risk",
                "mean_confidence",
            )
        }
    return {
        "benchmark_version": raw["version"],
        "seed": raw["seed"],
        "feature_names": list(FEATURE_NAMES),
        "methods": list(METHODS),
        "thresholds": thresholds,
        "logistic": {
            "weights": list(calibrator.weights),
            "intercept": calibrator.intercept,
        },
        "metrics": metrics,
        "drift": drift,
        "rows": rows,
    }


def render_calibration_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# M12 calibrated RAG results",
        "",
        "Thresholds are selected on the frozen calibration split only and then applied unchanged to test-ID and test-OOD.",
        "",
    ]
    for split in ("test_id", "test_ood"):
        lines.extend(
            [
                f"## {split}",
                "",
                "| Method | Accuracy | Brier | ECE | AURC | Threshold | Coverage | Selective risk | False-answer rate | Abstention acc |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in METHODS:
            metric = results["metrics"][split][method]
            lines.append(
                f"| {method} | {metric['full_coverage_accuracy']:.3f} | {metric['brier']:.3f} | "
                f"{metric['ece']:.3f} | {metric['aurc']:.3f} | {metric['threshold']:.2f} | "
                f"{metric['coverage']:.3f} | {metric['selective_risk']:.3f} | "
                f"{metric['false_answer_rate']:.3f} | {metric['abstention_accuracy']:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Drift: OOD minus ID",
            "",
            "| Method | Accuracy Δ | Brier Δ | ECE Δ | AURC Δ | Coverage Δ | Selective risk Δ | Mean confidence Δ |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        delta = results["drift"][method]
        lines.append(
            f"| {method} | {delta['full_coverage_accuracy']:+.3f} | {delta['brier']:+.3f} | "
            f"{delta['ece']:+.3f} | {delta['aurc']:+.3f} | {delta['coverage']:+.3f} | "
            f"{delta['selective_risk']:+.3f} | {delta['mean_confidence']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "Calibration quality and selective risk are reported separately. Timing/scale claims are not part of M12.1.",
            "",
        ]
    )
    return "\n".join(lines)
