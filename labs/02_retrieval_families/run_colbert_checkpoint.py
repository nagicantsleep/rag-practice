from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

import pylate
from huggingface_hub import model_info, snapshot_download

from rag_practice.evaluation.retrieval import evaluate_rankings
from rag_practice.retrieval.pretrained_colbert import PyLateColBERTRetriever

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "colbert-ir/colbertv2.0"


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

    info = model_info(MODEL_NAME)
    revision = info.sha
    snapshot_start = perf_counter()
    local_model_path = snapshot_download(repo_id=MODEL_NAME, revision=revision)
    snapshot_ms = (perf_counter() - snapshot_start) * 1000

    retriever = PyLateColBERTRetriever(local_model_path)
    retriever.fit(corpus)

    rankings = {}
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
    result = {
        "experiment_id": "m02_colbertv2_checkpoint_v1",
        "benchmark": "benchmarks/m02_retrieval@v1",
        "scope": "Full pretrained ColBERTv2 checkpoint with exhaustive MaxSim reranking over all 10 documents; no PLAID/ANN index.",
        "model": {
            "name": MODEL_NAME,
            "resolved_revision": revision,
            "pylate_version": pylate.__version__,
            "device": "cpu",
        },
        "metrics": metrics,
        "system": {
            "snapshot_resolution_ms": snapshot_ms,
            "model_load_ms": retriever.model_load_ms,
            "document_encoding_ms": retriever.index_build_ms,
            "mean_query_ms": fmean(latencies),
            "document_token_vectors": retriever.document_token_vectors(),
            "logical_embedding_bytes": retriever.logical_embedding_bytes(),
        },
        "per_query": per_query,
    }

    output_dir = ROOT / "labs/02_retrieval_families/results"
    json_path = output_dir / "colbert_checkpoint.json"
    md_path = output_dir / "colbert_checkpoint.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    all_metrics = metrics["all"]
    exact = metrics["exact"]
    semantic = metrics["semantic"]
    failures = [row for row in per_query if row["top1"] not in row["relevant"]]
    md_path.write_text(
        "\n".join(
            [
                "# M02 Full Pretrained ColBERTv2 Checkpoint",
                "",
                "Experiment: `m02_colbertv2_checkpoint_v1`  ",
                f"Model: `{MODEL_NAME}`  ",
                f"Resolved Hugging Face revision: `{revision}`  ",
                f"PyLate: `{pylate.__version__}`",
                "",
                "This evaluates the canonical pretrained ColBERTv2 checkpoint with token-level late interaction. Every document is supplied as a candidate and reranked exhaustively; this is checkpoint/scoring evaluation, not a PLAID index latency benchmark.",
                "",
                "| Metric | All | Exact | Semantic |",
                "| --- | ---: | ---: | ---: |",
                f"| Recall@1 | {all_metrics['recall@1']:.3f} | {exact['recall@1']:.3f} | {semantic['recall@1']:.3f} |",
                f"| Recall@3 | {all_metrics['recall@3']:.3f} | {exact['recall@3']:.3f} | {semantic['recall@3']:.3f} |",
                f"| MRR | {all_metrics['mrr']:.3f} | {exact['mrr']:.3f} | {semantic['mrr']:.3f} |",
                "",
                "## Multi-vector footprint",
                "",
                f"- stored document token vectors: {retriever.document_token_vectors()}",
                f"- logical embedding payload: {retriever.logical_embedding_bytes()} bytes",
                f"- document encoding: {retriever.index_build_ms:.1f} ms",
                f"- mean query encode + exhaustive MaxSim rerank: {fmean(latencies):.2f} ms",
                "",
                "## Top-1 failures",
                "",
                *(f"- `{row['id']}`: got `{row['top1']}`, expected one of `{', '.join(row['relevant'])}`" for row in failures),
                "" if failures else "- none",
                "",
                "Timings are GitHub Actions CPU sanity measurements. Production ColBERT retrieval uses a specialized multi-vector index; that serving problem is intentionally out of scope here.",
                "",
            ]
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
