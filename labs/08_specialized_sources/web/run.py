"""Run the deterministic M08 Web RAG benchmark."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from rag_practice.evaluation.web import evaluate_web_system
from rag_practice.web.pipeline import ExtractiveWebRAG
from rag_practice.web.ranking import WebRankingPolicy
from rag_practice.web.source import SnapshotWebSource, WebPage


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "m08_web"
RESULTS = Path(__file__).resolve().parent / "results"
AS_OF = date(2026, 8, 20)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_pages() -> list[WebPage]:
    pages = []
    for row in load_jsonl(BENCHMARK / "pages.jsonl"):
        pages.append(
            WebPage(
                id=str(row["id"]),
                url=str(row["url"]),
                domain=str(row["domain"]),
                title=str(row["title"]),
                text=str(row["text"]),
                updated_at=date.fromisoformat(str(row["updated_at"])),
                authority=float(row["authority"]),
                canonical_url=str(row["canonical_url"]),
            )
        )
    return pages


def main() -> None:
    pages = load_pages()
    questions = load_jsonl(BENCHMARK / "questions.jsonl")
    records = {page.id: page.to_record() for page in pages}

    body_source = SnapshotWebSource(pages, index_metadata=False)
    metadata_source = SnapshotWebSource(pages, index_metadata=True)
    systems = {
        "body_bm25": ExtractiveWebRAG(body_source),
        "metadata_bm25": ExtractiveWebRAG(metadata_source),
        "web_policy": ExtractiveWebRAG(
            metadata_source,
            policy=WebRankingPolicy(),
        ),
    }

    output: dict[str, object] = {
        "hypothesis": (
            "web retrieval needs explicit freshness, authority, and duplicate handling; "
            "a grounded answer can still be wrong when it cites a stale or low-authority page"
        ),
        "as_of": AS_OF.isoformat(),
        "benchmark": {
            "pages": len(pages),
            "queries": len(questions),
            "live_web_used": False,
            "reason": "frozen snapshots keep CI deterministic while preserving web-specific metadata/failures",
        },
        "systems": {},
    }

    for name, pipeline in systems.items():
        rankings = {}
        answers = {}
        traces = {}
        per_query = []
        for question in questions:
            query_id = str(question["id"])
            result = pipeline.ask(str(question["query"]), as_of=AS_OF)
            rankings[query_id] = list(result.retrieved_ids)
            answers[query_id] = result.answer
            traces[query_id] = result.trace
            top_id = result.retrieved_ids[0] if result.retrieved_ids else None
            top_record = records.get(top_id) if top_id else None
            per_query.append(
                {
                    "query_id": query_id,
                    "task": question["task"],
                    "query": question["query"],
                    "ranking": list(result.retrieved_ids),
                    "answer_contains_reference": str(question["answer"]).casefold()
                    in result.answer.casefold(),
                    "top_id": top_id,
                    "top_domain": top_record.metadata.get("domain") if top_record else None,
                    "top_authority": top_record.metadata.get("authority") if top_record else None,
                    "top_updated_at": top_record.metadata.get("updated_at") if top_record else None,
                    "stale_top1": top_id in set(question.get("stale_ids", [])) if top_id else False,
                    "citations": list(result.citations),
                    "trace": result.trace,
                }
            )

        metrics = evaluate_web_system(
            rankings=rankings,
            answers=answers,
            traces=traces,
            questions=questions,
            records=records,
        )
        output["systems"][name] = {
            "metrics": metrics,
            "per_query": per_query,
        }

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "results.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n")

    lines = [
        "# M08 Web RAG results",
        "",
        f"Frozen web snapshot as of `{AS_OF.isoformat()}`: {len(pages)} pages, {len(questions)} queries.",
        "",
        "| System | Hit@1 | Recall@3 | MRR | Stale top1 | Low-authority top1 | Duplicate@3 | Answer contains ref | Grounded | E2E ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in systems:
        m = output["systems"][name]["metrics"]
        lines.append(
            f"| {name} | {m['hit_rate@1']:.3f} | {m['recall@3']:.3f} | "
            f"{m['mrr']:.3f} | {m['stale_top1_rate']:.3f} | "
            f"{m['low_authority_top1_rate']:.3f} | {m['duplicate_rate@3']:.3f} | "
            f"{m['answer_contains_reference']:.3f} | {m['grounded_answer_rate']:.3f} | "
            f"{m['mean_end_to_end_ms']:.3f} |"
        )

    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Retrieval/source metrics are evaluated independently from answer text.",
        "- The answerer returns the top page verbatim; groundedness therefore cannot hide stale-source errors.",
        "- Authority scores are controlled benchmark metadata, not a claim that production trust can be reduced to one scalar.",
        "- The snapshot is deterministic and intentionally does not claim live-search coverage or production latency.",
    ]
    md_path = RESULTS / "results.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path.read_text())


if __name__ == "__main__":
    main()
