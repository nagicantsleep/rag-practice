from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from rag_practice.adaptive.control import AdaptiveRAGController
from rag_practice.adaptive.generation import AdaptiveGenerationPipeline
from rag_practice.adaptive.reflection import ActiveRetrievalPolicy, ReflectionCritic
from rag_practice.adaptive.router import AlwaysSingleRouter, NaiveBayesRouteClassifier
from rag_practice.evaluation.adaptive_generation import evaluate_adaptive_answers
from rag_practice.models.flan_t5 import FlanT5Backend

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks/m06_adaptive"
OUT = ROOT / "labs/06_adaptive_corrective/results"
FLAN_MODEL = "google/flan-t5-small"
FLAN_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
CONFIDENCE_THRESHOLD = 0.50
UTILITY_THRESHOLD = 0.65


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_learned_router(train: list[dict]) -> NaiveBayesRouteClassifier:
    router = NaiveBayesRouteClassifier(alpha=1.0)
    router.fit((row["query"], row["route"]) for row in train)
    return router


def build_critic() -> ReflectionCritic:
    return ReflectionCritic(
        relevance_threshold=0.12,
        support_threshold=0.80,
        active_policy=ActiveRetrievalPolicy(confidence_threshold=CONFIDENCE_THRESHOLD),
    )


def run_system(
    *,
    name: str,
    router,
    rows: list[dict],
    primary: dict[str, str],
    fallback: dict[str, str],
    flan: FlanT5Backend,
    enable_active_reflection: bool,
) -> dict:
    controller = AdaptiveRAGController(
        router=router,
        primary_documents=primary,
        fallback_documents=fallback,
        max_iterative_steps=2,
    )
    pipeline = AdaptiveGenerationPipeline(
        controller=controller,
        generator=flan,
        critic=build_critic(),
        utility_threshold=UTILITY_THRESHOLD,
        max_new_tokens=32,
    )

    traces = {}
    end_to_end_ms: list[float] = []
    for row in rows:
        start = perf_counter()
        traces[row["id"]] = pipeline.run(
            row["query"],
            enable_active_reflection=enable_active_reflection,
        )
        end_to_end_ms.append((perf_counter() - start) * 1000)

    metrics = evaluate_adaptive_answers(
        rows,
        traces,
        primary_documents=primary,
        fallback_documents=fallback,
    )
    metrics["mean_end_to_end_ms"] = sum(end_to_end_ms) / len(end_to_end_ms)

    serial = {}
    for query_id, trace in traces.items():
        serial[query_id] = {
            "route": trace.control.route.value,
            "control_retrieval_calls": trace.control.retrieval_calls,
            "active_retrieval_calls": trace.active_retrieval_calls,
            "total_retrieval_calls": trace.total_retrieval_calls,
            "control_context_ids": list(trace.control.selected_document_ids),
            "final_context_ids": list(trace.final_context_ids),
            "final_answer": trace.final_answer,
            "refused": trace.refused,
            "attempts": [
                {
                    "answer": attempt.answer,
                    "confidence": attempt.confidence,
                    "context_ids": list(attempt.context_ids),
                    "trigger_reason": attempt.trigger_reason,
                    "generation_ms": attempt.generation_ms,
                    "prompt_words": attempt.prompt_words,
                    "output_words": attempt.output_words,
                    "reflection": None
                    if attempt.reflection is None
                    else {
                        "retrieve": attempt.reflection.retrieve,
                        "relevant": attempt.reflection.relevant,
                        "supported": attempt.reflection.supported,
                        "utility": attempt.reflection.utility,
                    },
                }
                for attempt in trace.attempts
            ],
        }

    return {
        "name": name,
        "enable_active_reflection": enable_active_reflection,
        "metrics": metrics,
        "traces": serial,
    }


def main() -> None:
    rows = load_jsonl(BENCH / "queries.jsonl")
    train = load_jsonl(BENCH / "route_train.jsonl")
    primary = {row["id"]: row["text"] for row in load_jsonl(BENCH / "documents.jsonl")}
    fallback = {row["id"]: row["text"] for row in load_jsonl(BENCH / "fallback_documents.jsonl")}

    flan = FlanT5Backend(FLAN_MODEL, revision=FLAN_REVISION, device="cpu")
    systems = [
        run_system(
            name="always_single_rag",
            router=AlwaysSingleRouter(),
            rows=rows,
            primary=primary,
            fallback=fallback,
            flan=flan,
            enable_active_reflection=False,
        ),
        run_system(
            name="adaptive_control",
            router=build_learned_router(train),
            rows=rows,
            primary=primary,
            fallback=fallback,
            flan=flan,
            enable_active_reflection=False,
        ),
        run_system(
            name="adaptive_active_reflect",
            router=build_learned_router(train),
            rows=rows,
            primary=primary,
            fallback=fallback,
            flan=flan,
            enable_active_reflection=True,
        ),
    ]

    try:
        from huggingface_hub import model_info

        resolved_revision = model_info(FLAN_MODEL, revision=FLAN_REVISION).sha
    except Exception:
        resolved_revision = FLAN_REVISION

    payload = {
        "experiment_id": "m06_adaptive_generation_v1",
        "benchmark": "benchmarks/m06_adaptive@v2",
        "hypothesis": "Complexity routing plus corrective/iterative retrieval should improve evidence completeness while avoiding retrieval on no-RAG queries; confidence-triggered active retrieval and reflection should reduce unsupported answers, but may add latency or over-refuse.",
        "model": {
            "name": FLAN_MODEL,
            "requested_revision": FLAN_REVISION,
            "resolved_revision": resolved_revision,
            "model_load_ms": flan.model_load_ms,
        },
        "controls": {
            "router_train_split_separate_from_heldout": True,
            "references_or_qrels_exposed_to_runtime": False,
            "primary_retrieval": "BM25 top-1 per controller step",
            "max_iterative_steps": 2,
            "confidence_metric": "geometric mean selected-token probability",
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "reflection_relevance_threshold": 0.12,
            "reflection_support_threshold": 0.80,
            "reflection_utility_threshold": UTILITY_THRESHOLD,
            "active_retrieval_limit": 1,
            "active_query": "original question + first draft answer",
        },
        "systems": {system["name"]: system for system in systems},
        "warning": "Tiny controlled benchmark and FLAN-T5-small CPU sanity evaluation. Token-probability confidence is not calibrated factual correctness. Reflection uses transparent lexical support, not trained Self-RAG reflection tokens.",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "generation.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# M06 Adaptive/Active/Corrective Generation — Phase 2",
        "",
        f"Generator: `{FLAN_MODEL}` @ `{resolved_revision}`.",
        "",
        "| System | Answer F1 | Contains ref | Grounded | Evidence complete | Unsupported answer | Answerable refusal | Unanswerable refusal recall | Mean retrieval calls | Active calls | Attempts | E2E ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in systems:
        metrics = system["metrics"]
        lines.append(
            f"| {system['name']} | {metrics['answer_token_f1']:.3f} | {metrics['answer_contains_reference']:.3f} | {metrics['grounded_token_recall_retrieval_queries']:.3f} | {metrics['final_evidence_complete']:.3f} | {metrics['unsupported_answer_rate']:.3f} | {metrics['answerable_refusal_rate']:.3f} | {metrics['unanswerable_refusal_recall']:.3f} | {metrics['mean_total_retrieval_calls']:.2f} | {metrics['mean_active_retrieval_calls']:.2f} | {metrics['mean_generation_attempts']:.2f} | {metrics['mean_end_to_end_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Correctness is evaluated only on answerable questions. Grounded-token recall excludes no-retrieval questions. Unsupported-answer rate is measured only on the two deliberately unanswerable held-out questions.",
            "",
            "The active/reflection system may legitimately be a negative result if token confidence is poorly calibrated or lexical reflection over-refuses; those failures are retained for error analysis rather than tuned on this held-out set.",
            "",
        ]
    )
    (OUT / "generation.md").write_text("\n".join(lines))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
