from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

from rag_practice.core.models import Chunk
from rag_practice.embeddings.hashing import HashingEmbedder
from rag_practice.evaluation.retrieval import evaluate_rankings
from rag_practice.ir.bm25 import BM25Index
from rag_practice.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from rag_practice.retrieval.neural_dual_encoder import TinyNeuralDualEncoder, TrainingPair
from rag_practice.retrieval.vector_index import InMemoryVectorIndex

ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def qrels_for(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        row["id"]: {document_id: 1.0 for document_id in row["relevant_document_ids"]}
        for row in rows
    }


def evaluate_subset(rankings, qrels, query_rows, query_class):
    ids = [row["id"] for row in query_rows if row["class"] == query_class]
    return evaluate_rankings(
        {query_id: rankings[query_id] for query_id in ids},
        {query_id: qrels[query_id] for query_id in ids},
        ks=(1, 3),
    )


def build_hash_index(corpus: dict[str, str]):
    hashing = HashingEmbedder(256)
    index = InMemoryVectorIndex(hashing.dimensions)
    chunks = [
        Chunk(f"{document_id}::chunk-0", document_id, text, 0, len(text.split()))
        for document_id, text in corpus.items()
    ]
    index.add(chunks, hashing.embed_many([chunk.text for chunk in chunks]))
    return hashing, index


def score_query(query: str, bm25, hashing, hash_index, neural):
    bm25_results = bm25.search(query, k=10)
    bm25_scores = dict(bm25_results)
    bm25_ids = [document_id for document_id, _ in bm25_results]

    hash_results = hash_index.search(hashing.embed(query), k=10)
    hash_ids = [item.chunk.document_id for item in hash_results]

    neural_results = neural.search(query, k=10)
    neural_scores = dict(neural_results)
    neural_ids = [document_id for document_id, _ in neural_results]

    return bm25_scores, bm25_ids, hash_ids, neural_scores, neural_ids


def tune_bm25_weight(dev_rows, bm25, hashing, hash_index, neural):
    qrels = qrels_for(dev_rows)
    candidates = [index / 10 for index in range(11)]
    trials = []
    for alpha in candidates:
        rankings = {}
        for row in dev_rows:
            bm25_scores, _, _, neural_scores, _ = score_query(
                row["query"], bm25, hashing, hash_index, neural
            )
            fused = weighted_score_fusion(
                [bm25_scores, neural_scores], [alpha, 1.0 - alpha], limit=10
            )
            rankings[row["id"]] = [document_id for document_id, _ in fused]
        metrics = evaluate_rankings(rankings, qrels, ks=(1, 3))
        trials.append({"bm25_weight": alpha, "metrics": metrics})

    best = max(
        trials,
        key=lambda trial: (
            trial["metrics"]["recall@1"],
            trial["metrics"]["mrr"],
            trial["metrics"]["ndcg@3"],
            -trial["bm25_weight"],
        ),
    )
    return best["bm25_weight"], trials


def main() -> None:
    corpus_rows = load_jsonl(ROOT / "benchmarks/m00_ir/corpus.jsonl")
    train_rows = load_jsonl(ROOT / "benchmarks/m02_retrieval/train.jsonl")
    dev_rows = load_jsonl(ROOT / "benchmarks/m02_retrieval/dev.jsonl")
    query_rows = load_jsonl(ROOT / "benchmarks/m02_retrieval/queries.jsonl")
    corpus = {row["id"]: row["text"] for row in corpus_rows}
    qrels = qrels_for(query_rows)

    bm25 = BM25Index(corpus)
    hashing, hash_index = build_hash_index(corpus)

    neural = TinyNeuralDualEncoder(dimensions=32, seed=7)
    pairs = [TrainingPair(row["query"], row["document_id"]) for row in train_rows]
    train_start = perf_counter()
    losses = neural.fit(
        corpus,
        pairs,
        epochs=400,
        learning_rate=0.05,
        temperature=0.08,
    )
    training_ms = (perf_counter() - train_start) * 1000

    bm25_weight, tuning_trials = tune_bm25_weight(
        dev_rows, bm25, hashing, hash_index, neural
    )

    methods = ("bm25", "hashing", "neural", "hybrid_rrf", "hybrid_weighted")
    rankings = {method: {} for method in methods}
    latencies = {method: [] for method in methods}
    per_query = []

    for row in query_rows:
        query = row["query"]

        start = perf_counter()
        bm25_results = bm25.search(query, k=10)
        latencies["bm25"].append((perf_counter() - start) * 1000)
        bm25_scores = dict(bm25_results)
        bm25_ids = [document_id for document_id, _ in bm25_results]
        rankings["bm25"][row["id"]] = bm25_ids

        start = perf_counter()
        hash_results = hash_index.search(hashing.embed(query), k=10)
        latencies["hashing"].append((perf_counter() - start) * 1000)
        hash_ids = [item.chunk.document_id for item in hash_results]
        rankings["hashing"][row["id"]] = hash_ids

        start = perf_counter()
        neural_results = neural.search(query, k=10)
        latencies["neural"].append((perf_counter() - start) * 1000)
        neural_scores = dict(neural_results)
        neural_ids = [document_id for document_id, _ in neural_results]
        rankings["neural"][row["id"]] = neural_ids

        start = perf_counter()
        rrf = reciprocal_rank_fusion([bm25_ids, neural_ids], k=60, limit=10)
        latencies["hybrid_rrf"].append((perf_counter() - start) * 1000)
        rankings["hybrid_rrf"][row["id"]] = [document_id for document_id, _ in rrf]

        start = perf_counter()
        weighted = weighted_score_fusion(
            [bm25_scores, neural_scores],
            [bm25_weight, 1.0 - bm25_weight],
            limit=10,
        )
        latencies["hybrid_weighted"].append((perf_counter() - start) * 1000)
        rankings["hybrid_weighted"][row["id"]] = [
            document_id for document_id, _ in weighted
        ]

        per_query.append(
            {
                "id": row["id"],
                "class": row["class"],
                "query": query,
                "relevant": row["relevant_document_ids"],
                "top1": {
                    method: rankings[method][row["id"]][0]
                    if rankings[method][row["id"]]
                    else None
                    for method in methods
                },
            }
        )

    metrics = {}
    for method, method_rankings in rankings.items():
        metrics[method] = {
            "all": evaluate_rankings(method_rankings, qrels, ks=(1, 3)),
            "exact": evaluate_subset(method_rankings, qrels, query_rows, "exact"),
            "semantic": evaluate_subset(method_rankings, qrels, query_rows, "semantic"),
        }

    result = {
        "experiment_id": "m02_retrieval_core_v1",
        "benchmark": "benchmarks/m02_retrieval@v1",
        "corpus": "benchmarks/m00_ir@v1",
        "training": {
            "examples": len(pairs),
            "epochs": 400,
            "dimensions": 32,
            "seed": 7,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "training_ms": training_ms,
        },
        "hybrid_tuning": {
            "dev_queries": len(dev_rows),
            "selected_bm25_weight": bm25_weight,
            "selected_neural_weight": 1.0 - bm25_weight,
            "trials": tuning_trials,
        },
        "metrics": metrics,
        "latency_ms_mean": {
            method: fmean(values) for method, values in latencies.items()
        },
        "per_query": per_query,
    }

    output = ROOT / "labs/02_retrieval_families/results/core.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
