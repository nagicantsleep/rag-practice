"""Evaluator for the frozen M11.3 production-serving contract."""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from rag_practice.otc.serving import GuardedOtcServing

from .otc import _result_row


_REQUESTS = [
    (
        "p1",
        "t10",
        "U-OPS",
        "g0",
        "What is the current Helios shipment state for SO-1008 before any later carrier update?",
        False,
    ),
    (
        "p2",
        "t10",
        "U-OPS",
        "g0",
        "What is the current Helios shipment state for SO-1008 before any later carrier update?",
        True,
    ),
    (
        "p3",
        "t12",
        "U-OPS",
        "g0",
        "Show the payment status, credit-hold state, and hold reason for Cedar order SO-1003.",
        False,
    ),
    (
        "p4",
        "t13",
        "U-FIN",
        "g0",
        "Show the payment status, credit-hold state, and hold reason for Cedar order SO-1003.",
        False,
    ),
    (
        "p5",
        "t11",
        "U-OPS",
        "g0",
        "Using the contract effective at the benchmark time, what is Epsilon Retail's delivery commitment and is SO-1005 already in breach?",
        False,
    ),
    (
        "p6",
        "t16",
        "U-OPS",
        "g0",
        "What action should operations take for Gamma order SO-1007's address exception?",
        False,
    ),
    (
        "p8",
        "t18",
        "U-OPS",
        "g1",
        "At snapshot g1 after the carrier update, what confirmed exception explains Helios order SO-1008 and what escalation applies?",
        False,
    ),
    (
        "p9",
        "t18",
        "U-OPS",
        "g1",
        "At snapshot g1 after the carrier update, what confirmed exception explains Helios order SO-1008 and what escalation applies?",
        True,
    ),
]

_TRACE_FIELDS = {
    "request_sequence",
    "user_id",
    "roles",
    "snapshot_id",
    "generation",
    "cache_hit",
    "cache_key",
    "actions",
    "evidence_ids",
    "source_families",
    "rejected_unauthorized_ids",
    "rejected_stale_ids",
    "rejected_untrusted_ids",
    "stop_reason",
    "action_count",
    "integrated_latency_ms",
    "serving_latency_ms",
    "synthetic_tool_cost",
}


def _load_benchmark(root: Path) -> dict[str, Any]:
    return json.loads((root / "benchmark.json").read_text())


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _scale_records(size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + size)
    return [
        {
            "id": f"SCALE-{size}-{index:04d}",
            "customer_id": f"IRRELEVANT-{rng.randrange(10_000_000):07d}",
            "status": "ARCHIVED",
        }
        for index in range(size)
    ]


def _scale_sanity(root: Path) -> list[dict[str, Any]]:
    target_question = "What is the latest operational status and ETA for Juno order SO-1010?"
    target_expected = {
        "shipment_status": "IN_TRANSIT",
        "eta": "2026-08-15T22:00:00Z",
        "latest_event_code": "OUT_FOR_DELIVERY",
    }
    output: list[dict[str, Any]] = []
    for size in (100, 1000):
        serving = GuardedOtcServing(root)
        build_ms = serving.load_scale_records(_scale_records(size, 53))
        cold = serving.query(target_question, "U-OPS", snapshot_id="g0")
        warm = serving.query(target_question, "U-OPS", snapshot_id="g0")
        upsert_ms = serving.upsert_scale_record(
            {"id": f"SCALE-{size}-UPSERT", "customer_id": "IRRELEVANT-UP", "status": "ARCHIVED"}
        )
        delete_ms = serving.delete_scale_record(f"SCALE-{size}-UPSERT")
        stable = all(cold.result.answer.get(k) == v for k, v in target_expected.items())
        stable = stable and all(warm.result.answer.get(k) == v for k, v in target_expected.items())
        output.append(
            {
                "size": size,
                "build_ms": build_ms,
                "cold_query_ms": cold.trace.serving_latency_ms,
                "warm_query_ms": warm.trace.serving_latency_ms,
                "upsert_ms": upsert_ms,
                "delete_ms": delete_ms,
                "cache_entries": len(serving.cache),
                "logical_record_count": serving.logical_record_count,
                "target_answer_stable": stable,
            }
        )
    return output


def evaluate_production(data_root: str | Path) -> dict[str, Any]:
    root = Path(data_root)
    benchmark = _load_benchmark(root)
    task_by_id = {row["id"]: row for row in benchmark["tasks"]}
    serving = GuardedOtcServing(root)
    rows: list[dict[str, Any]] = []
    mutation: dict[str, Any] | None = None

    for request_id, task_id, user_id, snapshot, question, expected_cache in _REQUESTS:
        if request_id == "p8" and mutation is None:
            mutation = serving.apply_frozen_g1_mutation()
        response = serving.query(question, user_id, snapshot_id=snapshot)
        eval_row = _result_row(task_by_id[task_id], response.result)
        trace = serving.trace_dict(response)
        rows.append(
            {
                "request_id": request_id,
                "task_id": task_id,
                "expected_cache_hit": expected_cache,
                "cache_expectation_correct": trace["cache_hit"] == expected_cache,
                "trace_complete": _TRACE_FIELDS <= set(trace)
                and all(trace[field] is not None for field in _TRACE_FIELDS),
                "trace": trace,
                **eval_row,
            }
        )

    assert mutation is not None
    latencies = [row["trace"]["serving_latency_ms"] for row in rows]
    denied = next(row for row in rows if row["request_id"] == "p3")
    finance = next(row for row in rows if row["request_id"] == "p4")
    post_mutation = next(row for row in rows if row["request_id"] == "p8")

    role_isolation = (
        denied["answer"].get("decision") == "DENIED"
        and "FIN-1003" not in denied["evidence_ids"]
        and not denied["trace"]["cache_hit"]
        and "FIN-1003" in finance["evidence_ids"]
        and not finance["trace"]["cache_hit"]
    )
    generation_invalidation = (
        mutation["before_generation"] == 0
        and mutation["after_generation"] == 1
        and post_mutation["trace"]["generation"] == 1
        and not post_mutation["trace"]["cache_hit"]
    )
    mutation_correct = (
        mutation["active_snapshot"] == "g1"
        and mutation["appended_event_ids"] == ["EV-H003"]
        and mutation["replaced_shipment_ids"] == ["SH-1008@g1"]
        and post_mutation["answer"].get("root_cause") == "VEHICLE_BREAKDOWN"
    )

    stale_rows = [row for row in rows if row["task_id"] == "t11"]
    untrusted_rows = [row for row in rows if row["task_id"] == "t16"]
    unauthorized_rows = [row for row in rows if row["task_id"] == "t12"]
    scale = _scale_sanity(root)

    metrics = {
        "answer_field_accuracy": mean(row["answer_field_accuracy"] for row in rows),
        "evidence_recall": mean(row["evidence_recall"] for row in rows),
        "forbidden_exposure_rate": mean(float(row["forbidden_exposure"]) for row in rows),
        "cache_expectation_accuracy": mean(float(row["cache_expectation_correct"]) for row in rows),
        "cache_hit_rate": mean(float(row["trace"]["cache_hit"]) for row in rows),
        "role_isolation_correctness": float(role_isolation),
        "generation_invalidation_correctness": float(generation_invalidation),
        "mutation_correctness": float(mutation_correct),
        "unauthorized_exposure_rate": mean(float(row["forbidden_exposure"]) for row in unauthorized_rows),
        "stale_exposure_rate": mean(float(row["forbidden_exposure"]) for row in stale_rows),
        "untrusted_exposure_rate": mean(float(row["forbidden_exposure"]) for row in untrusted_rows),
        "observability_completeness": mean(float(row["trace_complete"]) for row in rows),
        "mean_serving_latency_ms": mean(latencies),
        "p50_serving_latency_ms": _percentile(latencies, 0.50),
        "p95_serving_latency_ms": _percentile(latencies, 0.95),
        "mean_action_count": mean(row["trace"]["action_count"] for row in rows),
        "mean_synthetic_tool_cost": mean(row["trace"]["synthetic_tool_cost"] for row in rows),
        "scale_target_answer_stability": mean(float(row["target_answer_stable"]) for row in scale),
    }
    return {
        "benchmark_version": benchmark["version"],
        "benchmark_clock": benchmark["benchmark_clock"],
        "system": "guarded_otc_serving",
        "metrics": metrics,
        "mutation": mutation,
        "rows": rows,
        "scale": scale,
    }


def render_production_markdown(results: dict[str, Any]) -> str:
    m = results["metrics"]
    lines = [
        "# M11.3 production-serving results",
        "",
        f"Frozen benchmark clock: `{results['benchmark_clock']}`.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| answer field accuracy | {m['answer_field_accuracy']:.3f} |",
        f"| evidence recall | {m['evidence_recall']:.3f} |",
        f"| cache expectation accuracy | {m['cache_expectation_accuracy']:.3f} |",
        f"| cache hit rate | {m['cache_hit_rate']:.3f} |",
        f"| role isolation | {m['role_isolation_correctness']:.3f} |",
        f"| generation invalidation | {m['generation_invalidation_correctness']:.3f} |",
        f"| mutation correctness | {m['mutation_correctness']:.3f} |",
        f"| unauthorized exposure | {m['unauthorized_exposure_rate']:.3f} |",
        f"| stale exposure | {m['stale_exposure_rate']:.3f} |",
        f"| untrusted exposure | {m['untrusted_exposure_rate']:.3f} |",
        f"| observability completeness | {m['observability_completeness']:.3f} |",
        f"| p50 serving ms | {m['p50_serving_latency_ms']:.3f} |",
        f"| p95 serving ms | {m['p95_serving_latency_ms']:.3f} |",
        f"| mean actions | {m['mean_action_count']:.3f} |",
        f"| mean synthetic tool cost | {m['mean_synthetic_tool_cost']:.3f} |",
        "",
        "## Scale sanity",
        "",
        "| Extra records | Stable | Build ms | Cold ms | Warm ms | Upsert ms | Delete ms | Cache entries | Logical records |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results["scale"]:
        lines.append(
            f"| {row['size']} | {int(row['target_answer_stable'])} | {row['build_ms']:.3f} | "
            f"{row['cold_query_ms']:.3f} | {row['warm_query_ms']:.3f} | {row['upsert_ms']:.3f} | "
            f"{row['delete_ms']:.3f} | {row['cache_entries']} | {row['logical_record_count']} |"
        )
    lines.extend(
        [
            "",
            "Timings and synthetic tool cost are educational implementation sanity measurements, not provider billing, database throughput, ANN performance, or concurrency claims.",
            "",
        ]
    )
    return "\n".join(lines)
