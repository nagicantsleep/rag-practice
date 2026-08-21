from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path

from rag_practice.training.retriever_finetune import (
    describe_model,
    evaluate_model_on_split,
    load_pinned_model,
    load_training_benchmark,
    mine_train_hard_negatives,
    train_pair_only,
    train_with_hard_negatives,
)


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "benchmarks" / "m10_training" / "dataset.json"
RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_JSON = RESULT_DIR / "training_results.json"
RESULT_MD = RESULT_DIR / "training_results.md"


def _metric_row(result: dict[str, object]) -> dict[str, float]:
    metrics = result["all"]
    assert isinstance(metrics, dict)
    return {
        "recall@1": float(metrics["recall@1"]),
        "recall@3": float(metrics["recall@3"]),
        "mrr": float(metrics["mrr"]),
        "mean_score_margin": float(metrics["mean_score_margin"]),
    }


def _delta(candidate: dict[str, object], baseline: dict[str, object]) -> dict[str, float]:
    candidate_row = _metric_row(candidate)
    baseline_row = _metric_row(baseline)
    return {key: candidate_row[key] - baseline_row[key] for key in candidate_row}


def _render_markdown(payload: dict[str, object]) -> str:
    systems = payload["systems"]
    assert isinstance(systems, dict)
    lines = [
        "# M10.1 retrieval training results",
        "",
        f"Pinned baseline: `{payload['model']['name']}@{payload['model']['revision']}`",
        "",
        "The benchmark, split, model revision, mining policy, and optimization hyperparameters were frozen before fine-tuned results were inspected.",
        "",
        "## Held-out test summary",
        "",
        "| System | Recall@1 | Recall@3 | MRR | Mean relevant-minus-best-negative margin | Training ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("baseline", "Pinned pretrained baseline"),
        ("pair_only", "Pair-only fine-tune"),
        ("hard_negative", "Hard-negative fine-tune"),
    ):
        system = systems[key]
        assert isinstance(system, dict)
        test = system["test"]
        assert isinstance(test, dict)
        row = _metric_row(test)
        training_ms = system.get("training", {}).get("training_ms", 0.0)
        lines.append(
            f"| {label} | {row['recall@1']:.3f} | {row['recall@3']:.3f} | "
            f"{row['mrr']:.3f} | {row['mean_score_margin']:.4f} | {float(training_ms):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Hard negatives",
            "",
            "Hard negatives are mined only from TRAIN documents with the untouched pinned baseline. Dev/test documents are never candidates for mining.",
            "",
            "| Query | Positive | Positive rank | Mined negative | Negative rank |",
            "| --- | --- | ---: | --- | ---: |",
        ]
    )
    for item in payload["mined_hard_negatives"]:
        lines.append(
            f"| {item['query_id']} | {item['positive_document_id']} | {item['positive_rank']} | "
            f"{item['negative_document_id']} | {item['negative_rank']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "This is a tiny synthetic domain-adaptation control. A positive delta is not a general fine-tuning claim, and a zero/negative delta is retained rather than tuned away. Dev metrics are diagnostic only; no post-test hyperparameter selection is allowed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    benchmark = load_training_benchmark(BENCHMARK_PATH)
    baseline_model, baseline_load_ms = load_pinned_model(
        max_sequence_length=benchmark.config.max_sequence_length
    )
    mined, mined_records = mine_train_hard_negatives(baseline_model, benchmark.train)
    baseline_dev = evaluate_model_on_split(baseline_model, benchmark.dev)
    baseline_test = evaluate_model_on_split(baseline_model, benchmark.test)
    model_metadata = describe_model(baseline_model, model_load_ms=baseline_load_ms)
    del baseline_model
    gc.collect()

    pair_model, pair_training = train_pair_only(benchmark.train, benchmark.config)
    pair_dev = evaluate_model_on_split(pair_model, benchmark.dev)
    pair_test = evaluate_model_on_split(pair_model, benchmark.test)
    del pair_model
    gc.collect()

    hard_model, hard_training = train_with_hard_negatives(
        benchmark.train, benchmark.config, mined
    )
    hard_dev = evaluate_model_on_split(hard_model, benchmark.dev)
    hard_test = evaluate_model_on_split(hard_model, benchmark.test)
    del hard_model
    gc.collect()

    payload: dict[str, object] = {
        "experiment_id": "m10_retriever_finetune_v1",
        "benchmark": "benchmarks/m10_training@v1",
        "freeze_rule": "dataset/model/mining/optimizer contract frozen before fine-tuned inspection",
        "model": model_metadata,
        "training_config": asdict(benchmark.config),
        "mined_hard_negatives": mined_records,
        "systems": {
            "baseline": {
                "dev": baseline_dev,
                "test": baseline_test,
            },
            "pair_only": {
                "training": pair_training,
                "dev": pair_dev,
                "test": pair_test,
                "test_delta_vs_baseline": _delta(pair_test, baseline_test),
            },
            "hard_negative": {
                "training": hard_training,
                "dev": hard_dev,
                "test": hard_test,
                "test_delta_vs_baseline": _delta(hard_test, baseline_test),
            },
        },
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    RESULT_MD.write_text(_render_markdown(payload))
    print(json.dumps({
        "baseline_test": _metric_row(baseline_test),
        "pair_only_test": _metric_row(pair_test),
        "hard_negative_test": _metric_row(hard_test),
    }, indent=2))


if __name__ == "__main__":
    main()
