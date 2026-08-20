from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from time import perf_counter

from rag_practice.evaluation.retrieval import evaluate_rankings
from rag_practice.ir.bm25 import BM25Index
from rag_practice.models.flan_t5 import FlanT5Backend
from rag_practice.query_transform import GenerativeQueryTransformer
from rag_practice.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from rag_practice.retrieval.pretrained import SentenceTransformerRetriever

ROOT = Path(__file__).resolve().parents[2]
FLAN_MODEL = "google/flan-t5-small"
FLAN_REVISION = "0fc9ddf"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_REVISION = "1c82ace116a2629de82404c4be48c0e5d4cf08be"
K = 5


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ranking_ids(results: list[tuple[str, float]]) -> list[str]:
    return [document_id for document_id, _ in results]


def score_map(results: list[tuple[str, float]]) -> dict[str, float]:
    return dict(results)


def complete_recall_at_k(
    rankings: dict[str, list[str]],
    rows: list[dict],
    *,
    k: int,
) -> float:
    if not rows:
        return 0.0
    values = []
    for row in rows:
        relevant = set(row["relevant_document_ids"])
        values.append(float(relevant.issubset(set(rankings.get(row["id"], [])[:k]))))
    return fmean(values)


def metrics_for_rows(rankings: dict[str, list[str]], rows: list[dict]) -> dict[str, float]:
    qrels = {
        row["id"]: {document_id: 1.0 for document_id in row["relevant_document_ids"]}
        for row in rows
    }
    metrics = evaluate_rankings(rankings, qrels, ks=(1, 3, 5))
    metrics["complete_recall@3"] = complete_recall_at_k(rankings, rows, k=3)
    return metrics


def breakdown(rankings: dict[str, list[str]], rows: list[dict]) -> dict[str, dict[str, float]]:
    classes = sorted({row["class"] for row in rows})
    result = {"all": metrics_for_rows(rankings, rows)}
    for query_class in classes:
        selected = [row for row in rows if row["class"] == query_class]
        result[query_class] = metrics_for_rows(rankings, selected)
    return result


def main() -> None:
    corpus_rows = load_jsonl(ROOT / "benchmarks/m00_ir/corpus.jsonl")
    queries = load_jsonl(ROOT / "benchmarks/m05_query_transform/queries.jsonl")
    documents = {row["id"]: row["text"] for row in corpus_rows}

    bm25 = BM25Index(documents)
    dense = SentenceTransformerRetriever(
        DENSE_MODEL,
        revision=DENSE_REVISION,
        device="cpu",
    )
    dense.fit(documents)
    flan = FlanT5Backend(FLAN_MODEL, revision=FLAN_REVISION, device="cpu")
    transformer = GenerativeQueryTransformer(flan)

    method_names = (
        "bm25_original",
        "bm25_rewrite",
        "multi_query_score_fusion",
        "rag_fusion_rrf",
        "query2doc_bm25",
        "decomposition_rrf",
        "dense_original",
        "hyde_dense",
    )
    rankings: dict[str, dict[str, list[str]]] = {name: {} for name in method_names}
    total_latencies: dict[str, list[float]] = defaultdict(list)
    transform_latencies: dict[str, list[float]] = defaultdict(list)
    generated_words: dict[str, list[int]] = defaultdict(list)
    traces: dict[str, dict] = {}

    for row in queries:
        query_id = row["id"]
        query = row["query"]
        trace: dict[str, object] = {
            "class": row["class"],
            "query": query,
            "relevant_document_ids": row["relevant_document_ids"],
        }

        start = perf_counter()
        original_results = bm25.search(query, k=K)
        total_latencies["bm25_original"].append((perf_counter() - start) * 1000)
        rankings["bm25_original"][query_id] = ranking_ids(original_results)

        start = perf_counter()
        transform_start = perf_counter()
        rewritten = transformer.rewrite(query)
        transform_latencies["bm25_rewrite"].append((perf_counter() - transform_start) * 1000)
        rewrite_results = bm25.search(rewritten, k=K)
        total_latencies["bm25_rewrite"].append((perf_counter() - start) * 1000)
        generated_words["bm25_rewrite"].append(len(rewritten.split()))
        rankings["bm25_rewrite"][query_id] = ranking_ids(rewrite_results)
        trace["rewrite"] = rewritten

        start = perf_counter()
        transform_start = perf_counter()
        variants = transformer.multi_query(query, count=3)
        multi_transform_ms = (perf_counter() - transform_start) * 1000
        variant_results = [bm25.search(variant, k=K) for variant in variants]
        fused_scores = weighted_score_fusion(
            [score_map(results) for results in variant_results],
            [1.0] * len(variant_results),
            limit=K,
        )
        multi_total_ms = (perf_counter() - start) * 1000
        transform_latencies["multi_query_score_fusion"].append(multi_transform_ms)
        total_latencies["multi_query_score_fusion"].append(multi_total_ms)
        generated_words["multi_query_score_fusion"].append(
            sum(len(item.split()) for item in variants[1:])
        )
        rankings["multi_query_score_fusion"][query_id] = ranking_ids(fused_scores)

        rrf_start = perf_counter()
        rrf = reciprocal_rank_fusion(
            [ranking_ids(results) for results in variant_results],
            limit=K,
        )
        rrf_extra_ms = (perf_counter() - rrf_start) * 1000
        transform_latencies["rag_fusion_rrf"].append(multi_transform_ms)
        total_latencies["rag_fusion_rrf"].append(multi_total_ms + rrf_extra_ms)
        generated_words["rag_fusion_rrf"].append(sum(len(item.split()) for item in variants[1:]))
        rankings["rag_fusion_rrf"][query_id] = ranking_ids(rrf)
        trace["multi_queries"] = variants

        start = perf_counter()
        transform_start = perf_counter()
        expanded = transformer.query2doc(query)
        transform_latencies["query2doc_bm25"].append((perf_counter() - transform_start) * 1000)
        expanded_results = bm25.search(expanded, k=K)
        total_latencies["query2doc_bm25"].append((perf_counter() - start) * 1000)
        generated_words["query2doc_bm25"].append(max(0, len(expanded.split()) - len(query.split())))
        rankings["query2doc_bm25"][query_id] = ranking_ids(expanded_results)
        trace["query2doc"] = expanded

        start = perf_counter()
        transform_start = perf_counter()
        parts = transformer.decompose(query, max_parts=3)
        transform_latencies["decomposition_rrf"].append((perf_counter() - transform_start) * 1000)
        part_rankings = [ranking_ids(bm25.search(part, k=K)) for part in parts]
        decomposed = reciprocal_rank_fusion(part_rankings, limit=K)
        total_latencies["decomposition_rrf"].append((perf_counter() - start) * 1000)
        generated_words["decomposition_rrf"].append(sum(len(part.split()) for part in parts))
        rankings["decomposition_rrf"][query_id] = ranking_ids(decomposed)
        trace["decomposition"] = parts

        start = perf_counter()
        dense_original = dense.search(query, k=K)
        total_latencies["dense_original"].append((perf_counter() - start) * 1000)
        rankings["dense_original"][query_id] = ranking_ids(dense_original)

        start = perf_counter()
        transform_start = perf_counter()
        hypothetical = transformer.hyde_document(query)
        transform_latencies["hyde_dense"].append((perf_counter() - transform_start) * 1000)
        hyde_results = dense.search(hypothetical, k=K)
        total_latencies["hyde_dense"].append((perf_counter() - start) * 1000)
        generated_words["hyde_dense"].append(len(hypothetical.split()))
        rankings["hyde_dense"][query_id] = ranking_ids(hyde_results)
        trace["hyde_document"] = hypothetical

        trace["top3"] = {
            method: ranking[query_id][:3]
            for method, ranking in rankings.items()
        }
        traces[query_id] = trace

    metrics = {method: breakdown(method_rankings, queries) for method, method_rankings in rankings.items()}

    system = {}
    for method in method_names:
        system[method] = {
            "mean_total_ms": fmean(total_latencies[method]),
            "mean_transform_ms": fmean(transform_latencies[method]) if transform_latencies[method] else 0.0,
            "mean_generated_words": fmean(generated_words[method]) if generated_words[method] else 0.0,
        }
    system["models"] = {
        "flan_model_load_ms": flan.model_load_ms,
        "dense_model_load_ms": dense.model_load_ms,
        "dense_index_build_ms": dense.index_build_ms,
    }

    try:
        from huggingface_hub import model_info
        resolved_flan = model_info(FLAN_MODEL, revision=FLAN_REVISION).sha
        resolved_dense = model_info(DENSE_MODEL, revision=DENSE_REVISION).sha
    except Exception:
        resolved_flan = FLAN_REVISION
        resolved_dense = DENSE_REVISION

    payload = {
        "experiment_id": "m05_query_transformation_v1",
        "benchmark": {
            "corpus": "benchmarks/m00_ir/corpus.jsonl",
            "queries": "benchmarks/m05_query_transform/queries.jsonl",
            "query_classes": sorted({row["class"] for row in queries}),
        },
        "hypothesis": "Generative query transformations should help vocabulary-mismatch and multi-aspect queries more than exact queries, but can also drift from the original intent. Each transformation is therefore evaluated against an original-query baseline using the same retriever family.",
        "controls": {
            "bm25_family_baseline": "bm25_original",
            "hyde_baseline": "dense_original",
            "multi_query_variants_include_original": True,
            "retrieval_k": K,
            "references_not_exposed_to_transformer": True,
        },
        "models": {
            "generator": {"name": FLAN_MODEL, "revision": resolved_flan},
            "dense_encoder": {"name": DENSE_MODEL, "revision": resolved_dense},
        },
        "metrics": metrics,
        "system": system,
        "per_query": traces,
        "warning": "Tiny controlled benchmark. Transformation outputs from FLAN-T5-small are retained even when poor; no test-query prompt tuning is used. CPU timings and generated-word counts are mechanism/cost sanity measurements, not production cost estimates.",
    }

    out = ROOT / "labs/05_query_transform/results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# M05 Query Transformation Baseline",
        "",
        "BM25-family methods are compared with `bm25_original`; HyDE is compared with `dense_original` because HyDE changes the query representation but keeps the dense retriever/index fixed.",
        "",
        "| Method | R@1 | R@3 | Complete R@3 | Exact R@1 | Semantic R@1 | Underspecified R@1 | Multi-aspect R@3 | Mean total ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in method_names:
        all_metrics = metrics[method]["all"]
        lines.append(
            f"| {method} | {all_metrics['recall@1']:.3f} | {all_metrics['recall@3']:.3f} | {all_metrics['complete_recall@3']:.3f} | {metrics[method]['exact']['recall@1']:.3f} | {metrics[method]['semantic']['recall@1']:.3f} | {metrics[method]['underspecified']['recall@1']:.3f} | {metrics[method]['multi_aspect']['recall@3']:.3f} | {system[method]['mean_total_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "`complete_recall@3` requires every relevant document to be present, which matters for multi-aspect queries where ordinary hit-rate can hide partial retrieval.",
            "",
        ]
    )
    (out / "baseline.md").write_text("\n".join(lines))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
