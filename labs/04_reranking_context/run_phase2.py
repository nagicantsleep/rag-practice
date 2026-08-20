from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

from rag_practice.core.models import Document
from rag_practice.evaluation.chunking import evaluate_chunk_rankings
from rag_practice.evaluation.rag import (
    answer_contains_reference,
    citation_precision,
    citation_recall,
    grounded_token_recall,
    token_f1,
)
from rag_practice.generation.query_extract import QueryAwareExtractiveAnswerer
from rag_practice.indexing.chunking import MetadataEnrichedChunker, SentenceChunker
from rag_practice.ir.bm25 import BM25Index
from rag_practice.models.flan_t5 import FlanT5Backend
from rag_practice.reranking.llm import PointwiseLLMReranker
from rag_practice.reranking.pretrained import CrossEncoderReranker
from rag_practice.reranking.selection import (
    RankedCandidate,
    edge_biased_order,
    pack_context,
    source_order,
)

ROOT = Path(__file__).resolve().parents[2]
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_REVISION = "c5f2b386de279a97c53a702dd5189d1c407160dc"
FLAN_MODEL = "google/flan-t5-small"
FLAN_REVISION = "0fc9ddf"
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


def flan_prompt(question: str, context: list[RankedCandidate]) -> str:
    joined = "\n\n".join(f"[{index + 1}] {item.text}" for index, item in enumerate(context))
    return (
        "Answer the question using only the supplied context. "
        "Use one or two short sentences. If the context is insufficient, say you do not know.\n"
        f"Question: {question}\n"
        f"Context:\n{joined}\n"
        "Answer:"
    )


def generation_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {
            "token_f1": 0.0,
            "contains_reference": 0.0,
            "grounded_token_recall": 0.0,
        }
    return {
        "token_f1": fmean(row["token_f1"] for row in rows),
        "contains_reference": fmean(row["contains_reference"] for row in rows),
        "grounded_token_recall": fmean(row["grounded_token_recall"] for row in rows),
    }


def main() -> None:
    document_rows = load_jsonl(ROOT / "benchmarks/m03_chunking/documents.jsonl")
    retrieval_queries = load_jsonl(ROOT / "benchmarks/m03_chunking/queries.jsonl")
    answer_rows = load_jsonl(ROOT / "benchmarks/m04_context/questions.jsonl")
    answers_by_id = {row["id"]: row for row in answer_rows}

    documents = [Document(row["id"], row["text"], row.get("metadata", {})) for row in document_rows]
    chunker = MetadataEnrichedChunker(
        SentenceChunker(max_words=35),
        fields=("title", "section", "tags", "region"),
    )
    chunks = chunker.chunk_many(documents)
    chunk_map = {chunk.id: chunk for chunk in chunks}
    index = BM25Index({chunk.id: chunk.text for chunk in chunks})

    first_stage: dict[str, list[RankedCandidate]] = {}
    for row in retrieval_queries:
        results = index.search(row["query"], k=CANDIDATE_K)
        first_stage[row["id"]] = [
            candidate_from_chunk(chunk_map[chunk_id], score)
            for chunk_id, score in results
        ]

    cross = CrossEncoderReranker(CROSS_MODEL, revision=CROSS_REVISION, device="cpu")
    flan = FlanT5Backend(FLAN_MODEL, revision=FLAN_REVISION, device="cpu")
    llm_reranker = PointwiseLLMReranker(flan)
    extractive = QueryAwareExtractiveAnswerer(max_sentences=2)

    policies: dict[str, dict[str, list[RankedCandidate]]] = {
        name: {} for name in (
            "first_stage_top3",
            "cross_encoder_top3",
            "cross_pack100_relevance",
            "cross_pack100_source_order",
            "cross_pack100_edge_order",
            "llm_pack100_relevance",
        )
    }
    cross_ms: list[float] = []
    llm_rerank_ms: list[float] = []

    for row in retrieval_queries:
        query_id = row["id"]
        candidates = first_stage[query_id]

        start = perf_counter()
        cross_ranked = cross.rerank(row["query"], candidates)
        cross_ms.append((perf_counter() - start) * 1000)

        start = perf_counter()
        llm_ranked = llm_reranker.rerank(row["query"], candidates)
        llm_rerank_ms.append((perf_counter() - start) * 1000)

        cross_packed = pack_context(
            cross_ranked,
            budget_words=CONTEXT_BUDGET_WORDS,
            reject_source_overlap_above=0.4,
        )[:CONTEXT_K]
        llm_packed = pack_context(
            llm_ranked,
            budget_words=CONTEXT_BUDGET_WORDS,
            reject_source_overlap_above=0.4,
        )[:CONTEXT_K]

        policies["first_stage_top3"][query_id] = candidates[:CONTEXT_K]
        policies["cross_encoder_top3"][query_id] = cross_ranked[:CONTEXT_K]
        policies["cross_pack100_relevance"][query_id] = cross_packed
        policies["cross_pack100_source_order"][query_id] = source_order(cross_packed)
        policies["cross_pack100_edge_order"][query_id] = edge_biased_order(cross_packed)
        policies["llm_pack100_relevance"][query_id] = llm_packed

    policy_metrics: dict[str, dict] = {}
    per_query: dict[str, dict] = {}
    flan_generation_ms: dict[str, list[float]] = {name: [] for name in policies}

    for policy_name, contexts_by_query in policies.items():
        rankings = {
            query_id: [item.id for item in context]
            for query_id, context in contexts_by_query.items()
        }
        retrieval_metrics = evaluate_chunk_rankings(rankings, chunk_map, retrieval_queries, k=CONTEXT_K)
        extractive_rows: list[dict] = []
        flan_rows: list[dict] = []

        for query in retrieval_queries:
            query_id = query["id"]
            answer_spec = answers_by_id[query_id]
            context = contexts_by_query[query_id]
            context_texts = [item.text for item in context]
            reference = answer_spec["reference"]
            relevant_docs = set(answer_spec["relevant_document_ids"])

            extracted = extractive.answer(answer_spec["question"], context)
            cited_docs = [
                next(item.document_id for item in context if item.id == candidate_id)
                for candidate_id in extracted.cited_candidate_ids
            ]
            extractive_row = {
                "answer": extracted.text,
                "token_f1": token_f1(extracted.text, reference),
                "contains_reference": answer_contains_reference(extracted.text, reference),
                "grounded_token_recall": grounded_token_recall(extracted.text, context_texts),
                "citation_precision": citation_precision(cited_docs, relevant_docs),
                "citation_recall": citation_recall(cited_docs, relevant_docs),
            }
            extractive_rows.append(extractive_row)

            start = perf_counter()
            generated = flan.generate(flan_prompt(answer_spec["question"], context), max_new_tokens=64)
            flan_generation_ms[policy_name].append((perf_counter() - start) * 1000)
            flan_row = {
                "answer": generated,
                "token_f1": token_f1(generated, reference),
                "contains_reference": answer_contains_reference(generated, reference),
                "grounded_token_recall": grounded_token_recall(generated, context_texts),
            }
            flan_rows.append(flan_row)

            per_query.setdefault(query_id, {})[policy_name] = {
                "context_ids": [item.id for item in context],
                "context_words": sum(item.word_count for item in context),
                "extractive": extractive_row,
                "flan": flan_row,
            }

        extractive_metrics = generation_metrics(extractive_rows)
        extractive_metrics.update(
            {
                "citation_precision": fmean(row["citation_precision"] for row in extractive_rows),
                "citation_recall": fmean(row["citation_recall"] for row in extractive_rows),
            }
        )
        policy_metrics[policy_name] = {
            "context": retrieval_metrics,
            "mean_context_words": fmean(
                sum(item.word_count for item in context)
                for context in contexts_by_query.values()
            ),
            "extractive_generation": extractive_metrics,
            "flan_generation": generation_metrics(flan_rows),
            "mean_flan_generation_ms": fmean(flan_generation_ms[policy_name]),
        }

    try:
        from huggingface_hub import model_info
        resolved_flan_revision = model_info(FLAN_MODEL, revision=FLAN_REVISION).sha
    except Exception:
        resolved_flan_revision = FLAN_REVISION

    result = {
        "experiment_id": "m04_context_generation_phase2_v1",
        "benchmark": {
            "documents": "benchmarks/m03_chunking/documents.jsonl",
            "retrieval_queries": "benchmarks/m03_chunking/queries.jsonl",
            "answer_references": "benchmarks/m04_context/questions.jsonl",
        },
        "hypothesis": "At a fixed high-recall candidate set, reranking and packing should improve answer-relevant context density; changing only context order may affect a pretrained generator even when the selected evidence set is identical.",
        "experimental_control": {
            "candidate_source": "BM25 over metadata-enriched sentence-35 chunks",
            "candidate_k": CANDIDATE_K,
            "context_k": CONTEXT_K,
            "context_budget_words": CONTEXT_BUDGET_WORDS,
            "candidate_set_frozen_before_reranking": True,
            "source_order_and_edge_order_reuse_exact_cross_pack_selected_set": True,
        },
        "models": {
            "cross_encoder": {"name": CROSS_MODEL, "revision": CROSS_REVISION},
            "instruction_model": {
                "name": FLAN_MODEL,
                "requested_revision": FLAN_REVISION,
                "resolved_revision": resolved_flan_revision,
                "uses": ["pointwise yes/no relevance scoring", "answer generation"],
            },
        },
        "metrics": policy_metrics,
        "system": {
            "cross_encoder_model_load_ms": cross.model_load_ms,
            "flan_model_load_ms": flan.model_load_ms,
            "mean_cross_encoder_rerank_ms": fmean(cross_ms),
            "mean_pointwise_llm_rerank_ms": fmean(llm_rerank_ms),
        },
        "per_query": per_query,
        "warning": "Tiny controlled benchmark. FLAN-T5-small is an instruction-model sanity check, not a claim about frontier LLM reranking or generation. CPU timings are implementation sanity measurements.",
    }

    output_dir = ROOT / "labs/04_reranking_context/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase2.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# M04 Phase 2 — LLM Reranking, Context Ordering, and Answer Quality",
        "",
        f"Instruction model: `{FLAN_MODEL}` @ `{resolved_flan_revision}` (requested `{FLAN_REVISION}`).",
        "",
        "All policies share the same frozen BM25 top-6 candidate sets. Source-order and edge-order reuse the exact selected set from cross-encoder budget packing; only ordering changes.",
        "",
        "| Policy | Evidence@1 | Evidence@3 | Relevant ctx@3 | Ctx words | Extractive F1 | Extractive grounded | FLAN F1 | FLAN grounded |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in policy_metrics.items():
        ctx = metrics["context"]
        ext = metrics["extractive_generation"]
        gen = metrics["flan_generation"]
        lines.append(
            f"| {name} | {ctx['evidence_complete@1']:.3f} | {ctx['evidence_complete@3']:.3f} | {ctx['relevant_context_fraction@3']:.3f} | {metrics['mean_context_words']:.1f} | {ext['token_f1']:.3f} | {ext['grounded_token_recall']:.3f} | {gen['token_f1']:.3f} | {gen['grounded_token_recall']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Mean cross-encoder rerank latency: **{fmean(cross_ms):.2f} ms/query**  ",
            f"Mean pointwise FLAN rerank latency: **{fmean(llm_rerank_ms):.2f} ms/query**",
            "",
            "Extractive generation is qrel-blind and deterministic. FLAN receives only question + ordered context. References are used only after generation for evaluation.",
            "",
        ]
    )
    (output_dir / "phase2.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
