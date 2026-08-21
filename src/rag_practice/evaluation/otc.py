"""Evaluation for M11.1 baselines."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from rag_practice.otc.baselines import BaselineResult, BaselineSuite


def _load_benchmark(root: Path) -> dict[str, Any]:
    return json.loads((root / "benchmark.json").read_text())


def _value_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        enum_like = (
            expected.isupper()
            or "_" in expected
            or expected in {"DENIED", "CONFLICT", "UNKNOWN", "NOT_READ"}
        )
        if enum_like:
            return expected.strip().casefold() == actual.strip().casefold()
        return expected.strip() == actual.strip()
    return expected == actual


def _field_accuracy(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    if not expected:
        return 1.0
    return sum(
        1 for key, value in expected.items()
        if key in actual and _value_equal(value, actual[key])
    ) / len(expected)


def _result_row(task: dict[str, Any], result: BaselineResult) -> dict[str, Any]:
    expected = task["expected"]
    required = set(task.get("evidence_ids", []))
    forbidden = set(task.get("forbidden_evidence_ids", []))
    seen = set(result.evidence_ids)
    required_sources = set(task.get("required_source_families", []))
    used_sources = set(result.source_families)

    field_accuracy = _field_accuracy(expected, result.answer)
    evidence_recall = len(required & seen) / len(required) if required else 1.0
    evidence_precision = len(required & seen) / len(seen) if seen else (1.0 if not required else 0.0)
    forbidden_exposure = bool(forbidden & seen)
    source_recall = (
        len(required_sources & used_sources) / len(required_sources)
        if required_sources else 1.0
    )
    source_precision = (
        len(required_sources & used_sources) / len(used_sources)
        if used_sources else (1.0 if not required_sources else 0.0)
    )
    answer_match = field_accuracy == 1.0
    task_success = answer_match and evidence_recall == 1.0 and not forbidden_exposure

    return {
        "task_id": task["id"],
        "class": task["class"],
        "split": task["split"],
        "user_id": task["user_id"],
        "snapshot": task["snapshot"],
        "question": task["question"],
        "expected": expected,
        "answer": result.answer,
        "answer_field_accuracy": field_accuracy,
        "answer_match": answer_match,
        "task_success": task_success,
        "required_evidence_ids": sorted(required),
        "evidence_ids": result.evidence_ids,
        "evidence_recall": evidence_recall,
        "evidence_precision": evidence_precision,
        "forbidden_evidence_ids": sorted(forbidden),
        "forbidden_exposure": forbidden_exposure,
        "required_source_families": sorted(required_sources),
        "source_families": result.source_families,
        "source_recall": source_recall,
        "source_precision": source_precision,
        "retrieved_documents": result.retrieved_documents,
        "latency_ms": result.latency_ms,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    sensitive = [r for r in rows if r["class"] == "unauthorized_sensitive_query"]
    stale = [r for r in rows if r["class"] == "stale_evidence_rejection"]
    untrusted = [r for r in rows if r["class"] == "adversarial_untrusted_evidence"]
    conflict = [r for r in rows if r["class"] == "conflicting_source_evidence"]
    no_evidence = [r for r in rows if r["class"] == "no_evidence_unknown_root_cause"]
    mutation = [r for r in rows if r["class"] in {"mutation_before_update", "mutation_after_update"}]

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)

    return {
        "task_success": mean(float(r["task_success"]) for r in rows),
        "answer_field_accuracy": mean(r["answer_field_accuracy"] for r in rows),
        "evidence_recall": mean(r["evidence_recall"] for r in rows),
        "evidence_precision": mean(r["evidence_precision"] for r in rows),
        "source_recall": mean(r["source_recall"] for r in rows),
        "source_precision": mean(r["source_precision"] for r in rows),
        "forbidden_exposure_rate": mean(float(r["forbidden_exposure"]) for r in rows),
        "unauthorized_exposure_rate": (
            mean(float(r["forbidden_exposure"]) for r in sensitive) if sensitive else 0.0
        ),
        "stale_exposure_rate": (
            mean(float(r["forbidden_exposure"]) for r in stale) if stale else 0.0
        ),
        "untrusted_exposure_rate": (
            mean(float(r["forbidden_exposure"]) for r in untrusted) if untrusted else 0.0
        ),
        "conflict_task_success": (
            mean(float(r["task_success"]) for r in conflict) if conflict else 0.0
        ),
        "no_evidence_task_success": (
            mean(float(r["task_success"]) for r in no_evidence) if no_evidence else 0.0
        ),
        "mutation_task_success": (
            mean(float(r["task_success"]) for r in mutation) if mutation else 0.0
        ),
        "mean_latency_ms": mean(r["latency_ms"] for r in rows),
        "by_class": {
            key: {
                "count": len(group),
                "task_success": mean(float(r["task_success"]) for r in group),
                "answer_field_accuracy": mean(r["answer_field_accuracy"] for r in group),
                "evidence_recall": mean(r["evidence_recall"] for r in group),
            }
            for key, group in sorted(by_class.items())
        },
    }


def evaluate_baselines(
    data_root: str | Path,
    *,
    split: str = "test",
) -> dict[str, Any]:
    root = Path(data_root)
    benchmark = _load_benchmark(root)
    suite = BaselineSuite(root)
    systems: dict[str, Callable[[str, str, str], BaselineResult]] = {
        "no_retrieval": suite.no_retrieval,
        "document_only": suite.document_only,
        "structured_only": suite.structured_only,
        "fixed_mixed": suite.fixed_mixed,
    }

    output: dict[str, Any] = {
        "benchmark_version": benchmark["version"],
        "benchmark_clock": benchmark["benchmark_clock"],
        "split": split,
        "systems": {},
    }
    tasks = [task for task in benchmark["tasks"] if task["split"] == split]
    for name, system in systems.items():
        rows = [
            _result_row(
                task,
                system(task["question"], task["user_id"], task["snapshot"]),
            )
            for task in tasks
        ]
        output["systems"][name] = {
            "metrics": _aggregate(rows),
            "rows": rows,
        }
    return output


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# M11.1 baseline results",
        "",
        f"Frozen benchmark split: `{results['split']}`. Benchmark clock: `{results['benchmark_clock']}`.",
        "",
        "| System | Task success | Field acc | Evidence recall | Evidence precision | Source recall | Unauthorized exposure | Stale exposure | Untrusted exposure | Mean ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, payload in results["systems"].items():
        m = payload["metrics"]
        lines.append(
            f"| {name} | {m['task_success']:.3f} | {m['answer_field_accuracy']:.3f} | "
            f"{m['evidence_recall']:.3f} | {m['evidence_precision']:.3f} | "
            f"{m['source_recall']:.3f} | {m['unauthorized_exposure_rate']:.3f} | "
            f"{m['stale_exposure_rate']:.3f} | {m['untrusted_exposure_rate']:.3f} | "
            f"{m['mean_latency_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Task success is strict: all expected answer fields must match, all required evidence must be present, and forbidden evidence must not be exposed.",
            "Timings are implementation sanity measurements, not production throughput claims.",
            "",
        ]
    )
    return "\n".join(lines)
