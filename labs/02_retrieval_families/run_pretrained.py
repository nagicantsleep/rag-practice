from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

import sentence_transformers

from rag_practice.evaluation.retrieval import evaluate_rankings
from rag_practice.retrieval.pretrained import SentenceTransformerRetriever

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1c82ace116a2629de82404c4be48c0e5d4cf08be"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def qrels_for(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        row["id"]: {document_id: 1.0 for document_id in row["relevant_document_ids"]}
        for row in rows
    }


def subset_metrics(rankings, qrels, rows, query_class):
    ids = [row["id"] for row in rows if row["class"] == query_class]
    return evaluate_rankings(
        {query_id: rankings[query_id] for query_id in ids},
        {query_id: qrels[query_id] for query_id in ids},
        ks=(1, 3),
    )


def main() -> None:
    corpus_rows = load_jsonl(ROOT / "benchmarks/m00_ir/corpus.jsonl")
    query_rows = load_jsonl(ROOT / "benchmarks/m02_retrieval/queries.jsonl")
    corpus = {row["id"]: row["text"] for row in corpus_rows}
    qrels = qrels_for(query_rows)

    retriever = SentenceTransformerRetriever(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
    )
    retriever.fit(corpus)

    rankings: dict[str, list[str]] = {}
    latencies = []
    per_query = []
    for row in query_rows:
        start = perf_counter()
        results = retriever.search(row["query"], k=10)
        latencies.append((perf_counter() - start) * 1000)
        ids = [document_id for document_id, _ in results]
        rankings[row["id"]] = ids
        per_query.append(
            {
                "id": row["id"],
                "class": row["class"],
                "query": row["query"],
                "relevant": row["relevant_document_ids"],
                "top1": ids[0] if ids else None,
                "top3": ids[:3],
            }
        )

    metrics = {
        "all": evaluate_rankings(rankings, qrels, ks=(1, 3)),
        "exact": subset_metrics(rankings, qrels, query_rows, "exact"),
        "semantic": subset_metrics(rankings, qrels, query_rows, "semantic"),
    }
    preserved = next(item for item in per_query if item["id"] == "s1")
    core = json.loads(
        (ROOT / "labs/02_retrieval_families/results/core.json").read_text()
    )

    result = {
        "experiment_id": "m02_pretrained_minilm_v1",
        "benchmark": "benchmarks/m02_retrieval@v1",
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "sentence_transformers_version": sentence_transformers.__version__,
            "dimensions": retriever.dimensions,
            "device": retriever.device,
        },
        "metrics": metrics,
        "preserved_paraphrase": preserved,
        "system": {
            "model_load_ms": retriever.model_load_ms,
            "index_build_ms": retriever.index_build_ms,
            "mean_query_ms": fmean(latencies),
            "logical_index_bytes_float32": retriever.logical_index_bytes(),
        },
        "core_comparison": {
            "bm25_recall_at_1": core["metrics"]["bm25"]["all"]["recall@1"],
            "neural_from_scratch_recall_at_1": core["metrics"]["neural"]["all"]["recall@1"],
            "hybrid_weighted_recall_at_1": core["metrics"]["hybrid_weighted"]["all"]["recall@1"],
        },
        "per_query": per_query,
    }

    output_dir = ROOT / "labs/02_retrieval_families/results"
    json_path = output_dir / "pretrained_sentence_transformer.json"
    md_path = output_dir / "pretrained_sentence_transformer.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    all_metrics = metrics["all"]
    exact_metrics = metrics["exact"]
    semantic_metrics = metrics["semantic"]
    fixed = preserved["top1"] in preserved["relevant"]
    md_path.write_text(
        "\n".join(
            [
                "# M02 Pretrained Dense Baseline — all-MiniLM-L6-v2",
                "",
                "Experiment: `m02_pretrained_minilm_v1`  ",
                f"Model: `{MODEL_NAME}`  ",
                f"Pinned revision: `{MODEL_REVISION}`  ",
                f"Sentence Transformers: `{sentence_transformers.__version__}`",
                "",
                "## Purpose",
                "",
                "Evaluate a broadly pretrained semantic sentence encoder under the same M02 ranking harness used by BM25, hashing vectors, and the tiny supervised dual encoder.",
                "",
                "## Retrieval quality",
                "",
                "| Metric | All | Exact | Semantic |",
                "| --- | ---: | ---: | ---: |",
                f"| Recall@1 | {all_metrics['recall@1']:.3f} | {exact_metrics['recall@1']:.3f} | {semantic_metrics['recall@1']:.3f} |",
                f"| Recall@3 | {all_metrics['recall@3']:.3f} | {exact_metrics['recall@3']:.3f} | {semantic_metrics['recall@3']:.3f} |",
                f"| MRR | {all_metrics['mrr']:.3f} | {exact_metrics['mrr']:.3f} | {semantic_metrics['mrr']:.3f} |",
                f"| nDCG@3 | {all_metrics['ndcg@3']:.3f} | {exact_metrics['ndcg@3']:.3f} | {semantic_metrics['ndcg@3']:.3f} |",
                "",
                "## Preserved paraphrase acceptance case",
                "",
                f"`s1` top-1: `{preserved['top1']}`; relevant: `{', '.join(preserved['relevant'])}`; fixed: **{str(fixed).lower()}**.",
                "",
                "## System sanity measurements",
                "",
                f"- model load: {retriever.model_load_ms:.1f} ms",
                f"- 10-document index encoding: {retriever.index_build_ms:.1f} ms",
                f"- mean query retrieval: {fmean(latencies):.2f} ms",
                f"- logical float32 embedding payload: {retriever.logical_index_bytes()} bytes",
                "",
                "## Interpretation",
                "",
                "The pretrained model supplies semantic knowledge learned outside this tiny benchmark; ranking remains our own cosine/dot-product implementation. Compare the result against the domain-trained-from-scratch encoder to separate broad pretraining from local supervision.",
                "",
                "Timings are GitHub Actions CPU measurements and are regression/sanity evidence, not universal performance claims.",
                "",
            ]
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
