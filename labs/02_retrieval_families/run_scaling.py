from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

from rag_practice.core.models import Chunk
from rag_practice.embeddings.hashing import HashingEmbedder
from rag_practice.evaluation.retrieval import evaluate_rankings
from rag_practice.evaluation.scaling import expand_corpus
from rag_practice.ir.bm25 import BM25Index
from rag_practice.retrieval.fusion import weighted_score_fusion
from rag_practice.retrieval.pretrained import SentenceTransformerRetriever
from rag_practice.retrieval.vector_index import InMemoryVectorIndex

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1c82ace116a2629de82404c4be48c0e5d4cf08be"
SCALES = (10, 100, 1000)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def qrels_for(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        row["id"]: {document_id: 1.0 for document_id in row["relevant_document_ids"]}
        for row in rows
    }


def build_hash_index(corpus: dict[str, str]):
    hashing = HashingEmbedder(256)
    chunks = [
        Chunk(f"{document_id}::chunk-0", document_id, text, 0, len(text.split()))
        for document_id, text in corpus.items()
    ]
    start = perf_counter()
    vectors = hashing.embed_many([chunk.text for chunk in chunks])
    index = InMemoryVectorIndex(hashing.dimensions)
    index.add(chunks, vectors)
    build_ms = (perf_counter() - start) * 1000
    return hashing, index, build_ms


def tune_bm25_weight(dev_rows, corpus, pretrained):
    bm25 = BM25Index(corpus)
    pretrained.fit(corpus)
    qrels = qrels_for(dev_rows)
    trials = []
    for step in range(11):
        alpha = step / 10
        rankings = {}
        for row in dev_rows:
            sparse = dict(bm25.search(row["query"], k=10))
            dense = dict(pretrained.search(row["query"], k=10))
            fused = weighted_score_fusion(
                [sparse, dense], [alpha, 1.0 - alpha], limit=10
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
    query_rows = load_jsonl(ROOT / "benchmarks/m02_retrieval/queries.jsonl")
    dev_rows = load_jsonl(ROOT / "benchmarks/m02_retrieval/dev.jsonl")
    base_corpus = {row["id"]: row["text"] for row in corpus_rows}
    qrels = qrels_for(query_rows)

    pretrained = SentenceTransformerRetriever(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
    )
    bm25_weight, tuning_trials = tune_bm25_weight(
        dev_rows, base_corpus, pretrained
    )

    scale_results = []
    for total_documents in SCALES:
        corpus = expand_corpus(base_corpus, total_documents)

        start = perf_counter()
        bm25 = BM25Index(corpus)
        bm25_build_ms = (perf_counter() - start) * 1000

        hashing, hash_index, hashing_build_ms = build_hash_index(corpus)
        pretrained.fit(corpus)

        methods = ("bm25", "hashing", "pretrained", "hybrid")
        rankings = {method: {} for method in methods}
        latencies = {method: [] for method in methods}

        for row in query_rows:
            query = row["query"]

            start = perf_counter()
            bm25_results = bm25.search(query, k=50)
            sparse_ms = (perf_counter() - start) * 1000
            latencies["bm25"].append(sparse_ms)
            sparse_scores = dict(bm25_results)
            rankings["bm25"][row["id"]] = [doc for doc, _ in bm25_results[:10]]

            start = perf_counter()
            hash_results = hash_index.search(hashing.embed(query), k=10)
            latencies["hashing"].append((perf_counter() - start) * 1000)
            rankings["hashing"][row["id"]] = [
                item.chunk.document_id for item in hash_results
            ]

            start = perf_counter()
            dense_results = pretrained.search(query, k=50)
            dense_ms = (perf_counter() - start) * 1000
            latencies["pretrained"].append(dense_ms)
            dense_scores = dict(dense_results)
            rankings["pretrained"][row["id"]] = [
                doc for doc, _ in dense_results[:10]
            ]

            fusion_start = perf_counter()
            fused = weighted_score_fusion(
                [sparse_scores, dense_scores],
                [bm25_weight, 1.0 - bm25_weight],
                limit=10,
            )
            fusion_ms = (perf_counter() - fusion_start) * 1000
            latencies["hybrid"].append(sparse_ms + dense_ms + fusion_ms)
            rankings["hybrid"][row["id"]] = [doc for doc, _ in fused]

        metrics = {
            method: evaluate_rankings(method_rankings, qrels, ks=(1, 3))
            for method, method_rankings in rankings.items()
        }
        postings = sum(len(items) for items in bm25.index.postings.values())
        scale_results.append(
            {
                "documents": total_documents,
                "metrics": metrics,
                "mean_query_ms": {
                    method: fmean(values) for method, values in latencies.items()
                },
                "build_ms": {
                    "bm25": bm25_build_ms,
                    "hashing": hashing_build_ms,
                    "pretrained": pretrained.index_build_ms,
                },
                "footprint": {
                    "bm25_vocabulary_terms": len(bm25.index.postings),
                    "bm25_posting_entries": postings,
                    "hashing_logical_float32_bytes": total_documents * 256 * 4,
                    "pretrained_logical_float32_bytes": pretrained.logical_index_bytes(),
                },
                "top1": {
                    method: {
                        row["id"]: rankings[method][row["id"]][0]
                        if rankings[method][row["id"]]
                        else None
                        for row in query_rows
                    }
                    for method in methods
                },
            }
        )

    result = {
        "experiment_id": "m02_scaling_stress_v1",
        "benchmark": "m02 held-out queries + deterministic off-topic distractors",
        "warning": "This is a candidate-set scaling/robustness stress test, not a broader semantic benchmark.",
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION},
        "hybrid_tuning": {
            "selected_bm25_weight": bm25_weight,
            "selected_pretrained_weight": 1.0 - bm25_weight,
            "dev_trials": tuning_trials,
        },
        "scales": scale_results,
    }

    output_dir = ROOT / "labs/02_retrieval_families/results"
    json_path = output_dir / "scaling.json"
    md_path = output_dir / "scaling.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# M02 Scaling Stress Test",
        "",
        "Experiment: `m02_scaling_stress_v1`",
        "",
        "This experiment keeps the same held-out M02 target questions and grows the candidate corpus with deterministic, deliberately off-topic distractors. It measures candidate-set robustness, build/search latency, and representation growth; it does **not** claim broader language-domain coverage.",
        "",
        f"Hybrid BM25 weight selected on the separate dev set: **{bm25_weight:.1f}** (pretrained dense: **{1.0 - bm25_weight:.1f}**).",
        "",
        "## Quality and latency by corpus size",
        "",
        "| Docs | Method | Recall@1 | Recall@3 | MRR | Mean query ms |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for scale in scale_results:
        for method in ("bm25", "hashing", "pretrained", "hybrid"):
            metric = scale["metrics"][method]
            lines.append(
                f"| {scale['documents']} | {method} | {metric['recall@1']:.3f} | {metric['recall@3']:.3f} | {metric['mrr']:.3f} | {scale['mean_query_ms'][method]:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Representation growth",
            "",
            "| Docs | BM25 postings | Hashing float32 payload | MiniLM float32 payload |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for scale in scale_results:
        fp = scale["footprint"]
        lines.append(
            f"| {scale['documents']} | {fp['bm25_posting_entries']} | {fp['hashing_logical_float32_bytes']} B | {fp['pretrained_logical_float32_bytes']} B |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The benchmark deliberately separates quality stability from systems cost. Exact quality can remain flat while exhaustive vector scans become linearly more expensive. Later production milestones will replace these educational exhaustive scans with ANN/index-serving systems; M02 keeps them visible so the cost model is obvious.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
