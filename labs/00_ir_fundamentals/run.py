"""Run the deterministic M00 TF-IDF vs BM25 retrieval benchmark."""

from __future__ import annotations

import json
import time
from pathlib import Path

from rag_practice.evaluation import evaluate_rankings
from rag_practice.ir import BM25Index, TfidfIndex

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "m00_ir"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_jsonl(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_method(name: str, retriever: object, queries: list[dict[str, str]], qrels: dict) -> dict:
    rankings: dict[str, list[str]] = {}
    latencies_ms: list[float] = []

    for query in queries:
        start = time.perf_counter()
        results = retriever.search(query["text"], k=5)  # type: ignore[attr-defined]
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        rankings[query["id"]] = [document_id for document_id, _ in results]

    metrics = evaluate_rankings(rankings, qrels, ks=(1, 3, 5))
    failures = []
    for query in queries:
        query_id = query["id"]
        relevant = set(qrels[query_id])
        top_5 = rankings[query_id][:5]
        if not relevant.intersection(top_5):
            failures.append(
                {
                    "query_id": query_id,
                    "query": query["text"],
                    "relevant": sorted(relevant),
                    "retrieved": top_5,
                    "failure_type": "retrieval_miss",
                }
            )

    return {
        "method": name,
        "metrics": metrics,
        "system": {
            "queries": len(queries),
            "mean_retrieval_latency_ms": sum(latencies_ms) / len(latencies_ms),
        },
        "rankings": rankings,
        "failures": failures,
    }


def main() -> None:
    corpus_rows = load_jsonl(BENCHMARK / "corpus.jsonl")
    queries = load_jsonl(BENCHMARK / "queries.jsonl")
    qrels = json.loads((BENCHMARK / "qrels.json").read_text())
    documents = {row["id"]: row["text"] for row in corpus_rows}

    experiments = [
        run_method("tfidf_cosine", TfidfIndex(documents), queries, qrels),
        run_method("bm25", BM25Index(documents), queries, qrels),
    ]

    result = {
        "experiment_id": "m00_lexical_baselines_v1",
        "milestone": "M00",
        "dataset": "benchmarks/m00_ir@v1",
        "top_k": 5,
        "experiments": experiments,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "baseline.json"
    output.write_text(json.dumps(result, indent=2) + "\n")

    for experiment in experiments:
        metrics = experiment["metrics"]
        print(
            f"{experiment['method']}: "
            f"MRR={metrics['mrr']:.3f} "
            f"Recall@5={metrics['recall@5']:.3f} "
            f"nDCG@5={metrics['ndcg@5']:.3f} "
            f"failures={len(experiment['failures'])}"
        )
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
