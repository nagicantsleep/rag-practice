"""Evaluation for the frozen M11.2 integrated copilot control."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from rag_practice.otc.integrated import IntegratedCopilot, IntegratedResult

from .otc import _aggregate, _result_row


def _load_benchmark(root: Path) -> dict[str, Any]:
    return json.loads((root / "benchmark.json").read_text())


def _integrated_row(task: dict[str, Any], result: IntegratedResult) -> dict[str, Any]:
    row = _result_row(task, result)
    row.update(
        actions=result.actions,
        action_count=len(result.actions),
        stop_reason=result.stop_reason,
        rejected_unauthorized_ids=result.rejected_unauthorized_ids,
        rejected_stale_ids=result.rejected_stale_ids,
        rejected_untrusted_ids=result.rejected_untrusted_ids,
    )
    return row


def evaluate_integrated(data_root: str | Path, *, split: str = "test") -> dict[str, Any]:
    root = Path(data_root)
    benchmark = _load_benchmark(root)
    copilot = IntegratedCopilot(root)
    tasks = [task for task in benchmark["tasks"] if task["split"] == split]
    rows = [
        _integrated_row(
            task,
            copilot.run(task["question"], task["user_id"], task["snapshot"]),
        )
        for task in tasks
    ]
    metrics = _aggregate(rows)
    metrics.update(
        mean_action_count=mean(row["action_count"] for row in rows),
        max_action_count=max(row["action_count"] for row in rows),
        rejected_unauthorized_count=sum(len(row["rejected_unauthorized_ids"]) for row in rows),
        rejected_stale_count=sum(len(row["rejected_stale_ids"]) for row in rows),
        rejected_untrusted_count=sum(len(row["rejected_untrusted_ids"]) for row in rows),
    )
    return {
        "benchmark_version": benchmark["version"],
        "benchmark_clock": benchmark["benchmark_clock"],
        "split": split,
        "system": "integrated_copilot",
        "metrics": metrics,
        "rows": rows,
    }


def render_integrated_markdown(results: dict[str, Any]) -> str:
    m = results["metrics"]
    lines = [
        "# M11.2 integrated copilot results",
        "",
        f"Frozen benchmark split: `{results['split']}`. Benchmark clock: `{results['benchmark_clock']}`.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| strict task success | {m['task_success']:.3f} |",
        f"| answer field accuracy | {m['answer_field_accuracy']:.3f} |",
        f"| evidence recall | {m['evidence_recall']:.3f} |",
        f"| evidence precision | {m['evidence_precision']:.3f} |",
        f"| source recall | {m['source_recall']:.3f} |",
        f"| source precision | {m['source_precision']:.3f} |",
        f"| unauthorized exposure | {m['unauthorized_exposure_rate']:.3f} |",
        f"| stale exposure | {m['stale_exposure_rate']:.3f} |",
        f"| untrusted exposure | {m['untrusted_exposure_rate']:.3f} |",
        f"| conflict task success | {m['conflict_task_success']:.3f} |",
        f"| no-evidence task success | {m['no_evidence_task_success']:.3f} |",
        f"| mutation task success | {m['mutation_task_success']:.3f} |",
        f"| mean action count | {m['mean_action_count']:.3f} |",
        f"| max action count | {m['max_action_count']} |",
        f"| mean latency ms | {m['mean_latency_ms']:.3f} |",
        "",
        "Every row persists action sequence, stop reason, evidence IDs, and rejected unauthorized/stale/untrusted IDs. Timings are implementation sanity measurements, not production throughput claims.",
        "",
    ]
    return "\n".join(lines)
