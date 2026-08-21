"""Finalize M08.2 only after the SQL / Structured RAG workflow has passed."""

from __future__ import annotations

import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ROADMAP = ROOT / "ROADMAP.md"
README = Path(__file__).resolve().parent / "README.md"


def main() -> None:
    run_id = os.environ.get("GITHUB_RUN_ID", "unknown")
    head_sha = os.environ.get("GITHUB_SHA", "unknown")

    roadmap = ROADMAP.read_text()
    section = f"""### M08 — Specialized Sources and Modalities — `IN PROGRESS`

M08 keeps source boundaries explicit so source-specific retrieval failures are evaluated before they are hidden behind a common orchestrator.

Sub-labs:

- **Web RAG — `DONE`**
- **SQL / structured RAG — `DONE`**
- metadata / filter-aware RAG — `TODO`
- Code RAG — `TODO`
- multimodal RAG — `TODO`
- visual-document / page-image RAG — `TODO`
- long-context vs retrieval routing — `TODO`

### Web RAG summary

Web RAG implements a minimal shared `Source` contract, deterministic web snapshots, body-only and metadata BM25 controls, query-aware authority/freshness reranking, canonical deduplication, and an extractive URL-citing pipeline.

| System | Hit@1 | Recall@3 | MRR | Stale top1 | Low-authority top1 | Duplicate@3 | Answer contains ref | Grounded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| body BM25 | 0.500 | 0.875 | 0.688 | 0.400 | 0.375 | 0.167 | 0.500 | **1.000** |
| metadata BM25 | 0.375 | **1.000** | 0.667 | 0.800 | 0.625 | 0.167 | 0.375 | **1.000** |
| Web policy | **1.000** | **1.000** | **1.000** | **0.000** | **0.000** | **0.000** | **1.000** | **1.000** |

Web findings remain unchanged: grounded citations can still be stale/wrong, metadata can worsen lexical top-1, recency is not authority, and canonical duplicates consume evidence budget. Web policy perfection is controlled tiny-snapshot evidence, not a live-web claim.

Artifacts: `benchmarks/m08_web/`, `src/rag_practice/sources/`, `src/rag_practice/web/`, `src/rag_practice/evaluation/web.py`, `labs/08_specialized_sources/web/`, and `.github/workflows/m08-web.yml`.

### SQL / Structured RAG summary

SQL / Structured RAG reuses the shared `Source` contract for a flat-row BM25 control and extends it with explicit schema discovery, transparent planning, read-only validation, relational execution, and row-level `sqlite://` provenance.

| System | Evidence recall | Evidence complete | Answer exact | Execution success | Unsafe reject | Empty correct | Unsupported handled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat row BM25@5 | 0.500 | 0.500 | n/a | n/a | n/a | n/a | n/a |
| schema-aware validated SQL | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

Important SQL / structured findings:

- **Flat row retrieval is not relational execution.** Aggregate queries can retrieve lexically plausible order/customer rows while missing the contributing `order_items`.
- **Answer correctness and row provenance are separate contracts.** Aggregate results persist contributing row IDs/citations rather than treating the scalar alone as sufficient evidence.
- **Empty is a first-class answer state.** The empty benchmark returns `NO_ROWS` without fabricated evidence.
- **Unsafe and unsupported requests fail closed.** Mutation is rejected before execution, and an absent loyalty-tier concept produces a planning error rather than invented schema.
- **Perfect scores are controlled-mechanism evidence only.** The planner is rule-based for a frozen four-table benchmark and does not claim Spider/BIRD-style text-to-SQL generalization.
- **SQLite safety/latency are teaching controls.** Production needs database permissions, query budgets, tenant filters, robust parsing/policy, and real workload measurements.

Evaluation evidence:

- Initial PR gate `32449251416`: **89 tests passed** plus successful SQL / Structured RAG evaluation.
- Final source-of-truth gate `{run_id}` passed before this ROADMAP update on head `{head_sha}`.
- SQL execution quality is evaluated independently from row-level provenance and final answer formatting.
- Persisted JSON/Markdown includes SQL, planned tables, answers, row evidence, citations, failure status, and latency.

Artifacts: `benchmarks/m08_sql/`, `src/rag_practice/structured_sql/`, `src/rag_practice/evaluation/sql_structured.py`, `labs/08_specialized_sources/sql/`, and `.github/workflows/m08-sql.yml`.
"""
    roadmap, count = re.subn(
        r"### M08 — Specialized Sources and Modalities — `(?:TODO|IN PROGRESS)`\n.*?(?=\n### M09)",
        section.rstrip(),
        roadmap,
        flags=re.S,
    )
    if count != 1:
        raise SystemExit(f"expected one M08 section, replaced {count}")

    next_step = """## Immediate next step

Continue **M08.3 — Metadata / Filter-aware RAG**. Keep filtering semantics separate from relevance ranking: define a corpus with tenant, product, region, time, and access-control metadata; compare post-filter vs pre-filter retrieval; measure constraint satisfaction, relevant recall under filters, empty-filter handling, unnecessary candidates, latency, and any security/permission leakage. Keep filter correctness separate from answer quality."""
    roadmap, count = re.subn(
        r"## Immediate next step\n\n.*\Z",
        next_step,
        roadmap,
        flags=re.S,
    )
    if count != 1:
        raise SystemExit(f"expected one immediate-next-step section, replaced {count}")
    ROADMAP.write_text(roadmap.rstrip() + "\n")

    readme = README.read_text()
    readme = readme.replace(
        "Status: **COMPLETION CANDIDATE** — final source-of-truth gate pending.",
        f"Status: **DONE** — final full-suite + SQL RAG gate passed in GitHub Actions run `{run_id}`.",
    )
    marker = "Initial PR gate `32449251416` passed the full repository suite (**89 tests**) and the SQL / Structured RAG evaluator."
    if "Final source-of-truth gate" not in readme:
        readme = readme.replace(
            marker,
            marker
            + f"\n\nFinal source-of-truth gate `{run_id}` passed the same full-suite/evaluator sequence before this completion update.",
        )
    readme = readme.replace(
        "- [ ] ROADMAP marks SQL / Structured RAG DONE only after the final gate passes",
        "- [x] ROADMAP marks SQL / Structured RAG DONE only after the final gate passes",
    )
    readme = readme.replace(
        "SQL / Structured RAG is not merged until the final unchecked gate passes.",
        "SQL / Structured RAG satisfies the sub-lab evaluation contract and is eligible to merge; M08 overall remains IN PROGRESS.",
    )
    README.write_text(readme.rstrip() + "\n")


if __name__ == "__main__":
    main()
