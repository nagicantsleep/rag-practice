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
from rag_practice.reranking.selection import (
    RankedCandidate,
    context_source_utilization,
    mmr_select,
    pack_context,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "c5f2b386de279a97c53a702dd5189d1c407160dc"
CANDIDATE_K = 6
CONTEXT_K = 3
CONTEXT_BUDGET_WORDS = 100


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def candidate_from_chunk(chunk, score: float) -> RankedCandidate:
    return RankedCandidate(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        first_stage_score=score,
        start_word=chunk.start_word,
        end_word=chunk.end_word,
    )


def candidate_recall(candidates_by_query: dict[str, list[RankedCandidate]], queries: list[dict]) -> dict[str, float]:
    document_hits = 0
    evidence_hits = 0
    for row in queries:
        items = candidates_by_query[row["id"]]
        relevant_document = row["relevant_document_id"]
        relevant_text = " ".join(
            item.text.lower() for item in items if item.document_id == relevant_document
        )
        required = [phrase.lower() for phrase in row.get("required_phrases", [])]
        document_hits += int(any(item.document_id == relevant_document for item in items))
        evidence_hits += int(bool(required) and all(phrase in relevant_text for phrase in required))
    count = len(queries)
    return {
        f"candidate_document_recall@{CANDIDATE_K}": document_hits / count,
        f"candidate_evidence_recall@{CANDIDATE_K}": evidence_hits / count,
    }


def main() -> None:
    document_rows = load_jsonl(ROOT / "benchmarks/m03_chunking/documents.jsonl")
    queries = load_jsonl(ROOT / "benchmarks/m03_chunking/queries.jsonl")
    documents = [
        Document(row["id"], row["text"], row.get("metadata", {}))
        for row in document_rows
    ]

    chunker = MetadataEnrichedChunker(
        SentenceChunker(max_words=35),
        fields=("title", "section", "tags", "region"),
    )
    chunks = chunker.chunk_many(documents)
    chunk_map = {chunk.id: chunk for chunk in chunks}
    index = BM25Index({chunk.id: chunk.text for chunk in chunks})

    first_stage_candidates: dict[str, list[RankedCandidate]] = {}
    first_stage_rankings: dict[str, list[str]] = {}
    for row in queries:
        results = index.search(row["query"], k=CANDIDATE_K)
        items = [candidate_from_chunk(chunk_map[chunk_id], score) for chunk_id, score in results]
        first_stage_candidates[row["id"]] = items
        first_stage_rankings[row["id"]] = [item.id for item in items[:CONTEXT_K]]

    candidate_metrics = candidate_recall(first_stage_candidates, queries)
    reranker = CrossEncoderReranker(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
    )

    reranked_rankings: dict[str, list[str]] = {}
    mmr_rankings: dict[str, list[str]] = {}
    packed_rankings: dict[str, list[str]] = {}
    rerank_ms: list[float] = []
    packed_utilization: list[float] = []
    per_query: list[dict] = []

    for row in queries:
        candidates = first_stage_candidates[row["id"]]
        start = perf_counter()
        reranked = reranker.rerank(row["query"], candidates)
        rerank_ms.append((perf_counter() - start) * 1000)

        mmr = mmr_select(reranked, limit=CONTEXT_K, relevance_weight=0.75)
        packed = pack_context(
            reranked,
            budget_words=CONTEXT_BUDGET_WORDS,
            reject_source_overlap_above=0.4,
        )[:CONTEXT_K]

        reranked_rankings[row["id"]] = [item.id for item in reranked[:CONTEXT_K]]
        mmr_rankings[row["id"]] = [item.id for item in mmr]
        packed_rankings[row["id"]] = [item.id for item in packed]
        packed_utilization.append(context_source_utilization(packed))
        per_query.append(
            {
                "id": row["id"],
                "query": row["query"],
                "relevant_document_id": row["relevant_document_id"],
                "first_stage_top3": first_stage_rankings[row["id"]],
                "reranked_top3": reranked_rankings[row["id"]],
                "mmr_top3": mmr_rankings[row["id"]],
                "packed": packed_rankings[row["id"]],
            }
        )

    result = {
        "experiment_id": "m04_cross_encoder_context_phase1_v1",
        "benchmark": "benchmarks/m03_chunking@v1",
        "hypothesis": "A pretrained cross-encoder should improve ordering inside a frozen high-recall BM25 candidate set; MMR and budget-aware packing should trade some raw ranking preference for less redundant, more efficient context.",
        "experimental_control": {
            "candidate_source": "BM25 over metadata-enriched sentence-35 chunks",
            "candidate_k": CANDIDATE_K,
            "context_k": CONTEXT_K,
            "context_budget_words": CONTEXT_BUDGET_WORDS,
            "candidate_set_frozen_before_reranking": True,
        },
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "device": "cpu",
        },
        "candidate_recall": candidate_metrics,
        "metrics": {
            "first_stage": evaluate_chunk_rankings(first_stage_rankings, chunk_map, queries, k=CONTEXT_K),
            "cross_encoder": evaluate_chunk_rankings(reranked_rankings, chunk_map, queries, k=CONTEXT_K),
            "cross_encoder_mmr": evaluate_chunk_rankings(mmr_rankings, chunk_map, queries, k=CONTEXT_K),
            "cross_encoder_budget_pack": evaluate_chunk_rankings(packed_rankings, chunk_map, queries, k=CONTEXT_K),
        },
        "system": {
            "model_load_ms": reranker.model_load_ms,
            "mean_rerank_ms": fmean(rerank_ms),
            "mean_packed_source_utilization": fmean(packed_utilization),
        },
        "per_query": per_query,
        "warning": "Tiny controlled benchmark. Timings are CPU sanity measurements; MMR here uses lexical Jaccard redundancy and the packer uses source-span overlap.",
    }

    output_dir = ROOT / "labs/04_reranking_context/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase1.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# M04 Phase 1 — Cross-encoder, MMR, and Context Packing",
        "",
        f"Model: `{MODEL_NAME}` @ `{MODEL_REVISION}`",
        "",
        f"Frozen BM25 candidate set: top-{CANDIDATE_K}; returned context: top-{CONTEXT_K}; packing budget: {CONTEXT_BUDGET_WORDS} words.",
        "",
        f"Candidate document recall@{CANDIDATE_K}: **{candidate_metrics[f'candidate_document_recall@{CANDIDATE_K}']:.3f}**  ",
        f"Candidate evidence recall@{CANDIDATE_K}: **{candidate_metrics[f'candidate_evidence_recall@{CANDIDATE_K}']:.3f}**",
        "",
        "| Method | Doc hit@1 | Evidence@1 | Evidence@3 | Source util@3 | Relevant context@3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("first_stage", "cross_encoder", "cross_encoder_mmr", "cross_encoder_budget_pack"):
        metrics = result["metrics"][method]
        lines.append(
            f"| {method} | {metrics['document_hit@1']:.3f} | {metrics['evidence_complete@1']:.3f} | {metrics['evidence_complete@3']:.3f} | {metrics['source_token_utilization@3']:.3f} | {metrics['relevant_context_fraction@3']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The candidate-recall rows are a guardrail: reranking cannot repair evidence that the first stage failed to retrieve. MMR and packing are evaluated separately from the cross-encoder ordering so diversity and budget effects stay visible.",
            "",
        ]
    )
    (output_dir / "phase1.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
