from __future__ import annotations

import json
from pathlib import Path

from rag_practice.training.linear_reranker import (
    LinearPairwiseReranker,
    evaluate_reranker,
    load_reranker_contract,
)
from rag_practice.training.retriever_finetune import (
    load_pinned_model,
    load_training_benchmark,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "benchmarks" / "m10_training" / "dataset.json"
CONTRACT = ROOT / "benchmarks" / "m10_training" / "reranker_contract.json"
RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_JSON = RESULT_DIR / "reranker_results.json"
RESULT_MD = RESULT_DIR / "reranker_results.md"


def main() -> None:
    benchmark = load_training_benchmark(DATASET)
    config = load_reranker_contract(CONTRACT)
    baseline_model, model_load_ms = load_pinned_model(
        max_sequence_length=benchmark.config.max_sequence_length
    )
    reranker = LinearPairwiseReranker(config)
    reranker.fit(baseline_model=baseline_model, split=benchmark.train)
    dev = evaluate_reranker(
        reranker=reranker, baseline_model=baseline_model, split=benchmark.dev
    )
    test = evaluate_reranker(
        reranker=reranker, baseline_model=baseline_model, split=benchmark.test
    )
    payload = {
        "experiment_id": "m10_linear_reranker_v1",
        "benchmark": "benchmarks/m10_training@v1 (unchanged post retriever evidence)",
        "contract": "benchmarks/m10_training/reranker_contract.json@v1",
        "model_load_ms": model_load_ms,
        "config": config.__dict__,
        "training": {
            "training_pair_count": reranker.training_pair_count,
            "training_ms": reranker.training_ms,
            "loss_history": reranker.loss_history,
        },
        "parameters": reranker.parameters_payload(),
        "dev": dev,
        "test": test,
        "caveat": "Reranker architecture was frozen after retriever evidence on the same benchmark; this is a post-hoc training control, not a fresh held-out benchmark claim.",
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    RESULT_MD.write_text(
        "# M10.1b learned reranker results\n\n"
        "| Split | Candidate Recall@3 | Rerank Recall@1 | MRR | Mean rerank margin | Mean rerank ms |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n"
        f"| dev | {dev['candidate_recall@3']:.3f} | {dev['recall@1']:.3f} | {dev['mrr']:.3f} | {dev['mean_rerank_margin']:.4f} | {dev['mean_rerank_ms']:.4f} |\n"
        f"| test | {test['candidate_recall@3']:.3f} | {test['recall@1']:.3f} | {test['mrr']:.3f} | {test['mean_rerank_margin']:.4f} | {test['mean_rerank_ms']:.4f} |\n\n"
        f"Training pairs: `{reranker.training_pair_count}`; training ms: `{reranker.training_ms:.1f}`.\n\n"
        "This is a transparent post-hoc training control on the unchanged M10.1 split. Candidate recall is measured before reranking; missing positives cannot be repaired by the reranker.\n"
    )
    print(json.dumps({"dev": {k: v for k, v in dev.items() if not isinstance(v, (dict, list))}, "test": {k: v for k, v in test.items() if not isinstance(v, (dict, list))}}, indent=2))


if __name__ == "__main__":
    main()
