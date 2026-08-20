from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

from rag_practice.core.models import Document
from rag_practice.evaluation.chunking import evaluate_chunk_rankings
from rag_practice.indexing.chunking import MetadataEnrichedChunker, SentenceChunker
from rag_practice.ir.bm25 import BM25Index
from rag_practice.reranking.pretrained import CrossEncoderReranker
from rag_practice.reranking.selection import RankedCandidate

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "c5f2b386de279a97c53a702dd5189d1c407160dc"
DEPTHS = (2, 4, 6)
CONTEXT_K = 3


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def candidate(chunk, score: float) -> RankedCandidate:
    return RankedCandidate(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        first_stage_score=score,
        start_word=chunk.start_word,
        end_word=chunk.end_word,
    )


def candidate_recall(candidates: dict[str, list[RankedCandidate]], queries: list[dict]) -> dict[str, float]:
    doc_hits = 0
    evidence_hits = 0
    for query in queries:
        items = candidates[query["id"]]
        relevant = query["relevant_document_id"]
        relevant_text = " ".join(item.text.lower() for item in items if item.document_id == relevant)
        required = [phrase.lower() for phrase in query.get("required_phrases", [])]
        doc_hits += int(any(item.document_id == relevant for item in items))
        evidence_hits += int(bool(required) and all(phrase in relevant_text for phrase in required))
    count = len(queries)
    return {"document_recall": doc_hits / count, "evidence_recall": evidence_hits / count}


def main() -> None:
    documents = [
        Document(row["id"], row["text"], row.get("metadata", {}))
        for row in load_jsonl(ROOT / "benchmarks/m03_chunking/documents.jsonl")
    ]
    queries = load_jsonl(ROOT / "benchmarks/m03_chunking/queries.jsonl")
    chunks = MetadataEnrichedChunker(
        SentenceChunker(max_words=35),
        fields=("title", "section", "tags", "region"),
    ).chunk_many(documents)
    chunk_map = {chunk.id: chunk for chunk in chunks}
    index = BM25Index({chunk.id: chunk.text for chunk in chunks})
    reranker = CrossEncoderReranker(MODEL_NAME, revision=MODEL_REVISION, device="cpu")

    results = []
    for depth in DEPTHS:
        candidates: dict[str, list[RankedCandidate]] = {}
        rankings: dict[str, list[str]] = {}
        latencies = []
        for query in queries:
            items = [
                candidate(chunk_map[chunk_id], score)
                for chunk_id, score in index.search(query["query"], k=depth)
            ]
            candidates[query["id"]] = items
            start = perf_counter()
            reranked = reranker.rerank(query["query"], items)
            latencies.append((perf_counter() - start) * 1000)
            rankings[query["id"]] = [item.id for item in reranked[:CONTEXT_K]]

        recall = candidate_recall(candidates, queries)
        quality = evaluate_chunk_rankings(rankings, chunk_map, queries, k=CONTEXT_K)
        results.append(
            {
                "candidate_k": depth,
                "mean_candidates_returned": fmean(len(items) for items in candidates.values()),
                "candidate_document_recall": recall["document_recall"],
                "candidate_evidence_recall": recall["evidence_recall"],
                "reranked_context": quality,
                "mean_rerank_ms": fmean(latencies),
            }
        )

    payload = {
        "experiment_id": "m04_candidate_depth_sweep_v1",
        "benchmark": "benchmarks/m03_chunking@v1",
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION, "device": "cpu"},
        "hypothesis": "Retrieving more candidates should reduce first-stage evidence misses but make cross-encoder reranking more expensive; the useful operating point should be judged by both candidate recall and final context quality.",
        "results": results,
        "warning": "Tiny controlled corpus and CPU timings. This measures the retrieve-many/rerank-few mechanism, not production throughput.",
    }
    output_dir = ROOT / "labs/04_reranking_context/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "depth_sweep.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# M04 Candidate-depth latency/quality sweep",
        "",
        "| Candidate k | Candidate doc recall | Candidate evidence recall | Evidence@3 after rerank | Relevant context@3 | Mean rerank ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        quality = row["reranked_context"]
        lines.append(
            f"| {row['candidate_k']} | {row['candidate_document_recall']:.3f} | {row['candidate_evidence_recall']:.3f} | {quality['evidence_complete@3']:.3f} | {quality['relevant_context_fraction@3']:.3f} | {row['mean_rerank_ms']:.2f} |"
        )
    lines.extend(["", "Candidate recall is reported before reranking so a missing passage cannot be credited to the reranker.", ""])
    (output_dir / "depth_sweep.md").write_text("\n".join(lines))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
