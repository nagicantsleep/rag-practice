from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

from rag_practice.evaluation.retrieval import evaluate_rankings
from rag_practice.retrieval.late_interaction import LateInteractionRetriever
from rag_practice.retrieval.learned_sparse import LearnedSparseRetriever
from rag_practice.retrieval.neural_dual_encoder import TrainingPair

ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def qrels_for(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        row["id"]: {document_id: 1.0 for document_id in row["relevant_document_ids"]}
        for row in rows
    }


def evaluate_subset(rankings, qrels, rows, query_class):
    ids = [row["id"] for row in rows if row["class"] == query_class]
    return evaluate_rankings(
        {query_id: rankings[query_id] for query_id in ids},
        {query_id: qrels[query_id] for query_id in ids},
        ks=(1, 3),
    )


def evaluate_model(name, model, fit_kwargs, corpus, pairs, queries, qrels):
    start = perf_counter()
    losses = model.fit(corpus, pairs, **fit_kwargs)
    training_ms = (perf_counter() - start) * 1000

    rankings = {}
    latencies = []
    for row in queries:
        start = perf_counter()
        results = model.search(row["query"], k=10)
        latencies.append((perf_counter() - start) * 1000)
        rankings[row["id"]] = [document_id for document_id, _ in results]

    result = {
        "config": fit_kwargs,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "training_ms": training_ms,
        "latency_ms_mean": fmean(latencies),
        "all": evaluate_rankings(rankings, qrels, ks=(1, 3)),
        "exact": evaluate_subset(rankings, qrels, queries, "exact"),
        "semantic": evaluate_subset(rankings, qrels, queries, "semantic"),
        "top1": {row["id"]: rankings[row["id"]][0] for row in queries},
    }

    if name == "learned_sparse":
        result.update(
            {
                "vocab_size": len(model.vocabulary),
                "mean_nonzero_document_dimensions": model.mean_nonzero_dimensions(),
                "interpretable_expansion_example": {
                    "query": "orientation of numerical representations",
                    "top_dimensions": sorted(
                        model.encode_query("orientation of numerical representations").items(),
                        key=lambda item: -item[1],
                    )[:10],
                },
            }
        )
    else:
        result.update(
            {
                "dimensions_per_token": model.dimensions,
                "document_token_vectors": sum(
                    len(text.split()) for text in corpus.values()
                ),
            }
        )
    return result


def main() -> None:
    corpus_rows = load_jsonl(ROOT / "benchmarks/m00_ir/corpus.jsonl")
    train_rows = load_jsonl(ROOT / "benchmarks/m02_retrieval/train.jsonl")
    query_rows = load_jsonl(ROOT / "benchmarks/m02_retrieval/queries.jsonl")
    corpus = {row["id"]: row["text"] for row in corpus_rows}
    pairs = [TrainingPair(row["query"], row["document_id"]) for row in train_rows]
    qrels = qrels_for(query_rows)

    result = {
        "experiment_id": "m02_advanced_mechanics_v1",
        "benchmark": "benchmarks/m02_retrieval@v1",
        "note": (
            "These are mechanism-focused SPLADE-style and ColBERT-style models, "
            "not pretrained checkpoint reproductions."
        ),
        "learned_sparse": evaluate_model(
            "learned_sparse",
            LearnedSparseRetriever(seed=7),
            {"epochs": 300, "learning_rate": 0.05, "sparsity_lambda": 0.01},
            corpus,
            pairs,
            query_rows,
            qrels,
        ),
        "late_interaction": evaluate_model(
            "late_interaction",
            LateInteractionRetriever(dimensions=16, seed=7),
            {"epochs": 100, "learning_rate": 0.05, "temperature": 0.15},
            corpus,
            pairs,
            query_rows,
            qrels,
        ),
    }

    output = ROOT / "labs/02_retrieval_families/results/advanced_mechanics.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
