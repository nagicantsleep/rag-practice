"""Finalize M08.1 only after the Web RAG workflow has passed."""

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
- SQL / structured RAG — `TODO`
- metadata / filter-aware RAG — `TODO`
- Code RAG — `TODO`
- multimodal RAG — `TODO`
- visual-document / page-image RAG — `TODO`
- long-context vs retrieval routing — `TODO`

Web RAG implements a minimal shared `Source` contract, deterministic web snapshots, body-only and metadata BM25 controls, query-aware authority/freshness reranking, canonical deduplication, and an extractive URL-citing pipeline.

Web RAG held-out summary:

| System | Hit@1 | Recall@3 | MRR | Stale top1 | Low-authority top1 | Duplicate@3 | Answer contains ref | Grounded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| body BM25 | 0.500 | 0.875 | 0.688 | 0.400 | 0.375 | 0.167 | 0.500 | **1.000** |
| metadata BM25 | 0.375 | **1.000** | 0.667 | 0.800 | 0.625 | 0.167 | 0.375 | **1.000** |
| Web policy | **1.000** | **1.000** | **1.000** | **0.000** | **0.000** | **0.000** | **1.000** | **1.000** |

Important Web RAG findings:

- **Grounded does not mean current or correct.** Both lexical baselines copy their selected page exactly and therefore score groundedness `1.0`, while answer correctness is only `0.500` / `0.375`.
- **More metadata can make lexical ranking worse.** Query-shaped forum/blog titles reduce metadata-BM25 Hit@1 even while Recall@3 improves.
- **Recency alone is not trust.** Recently updated low-authority pages can still conflict with official sources; the controlled policy needs both freshness and authority.
- **Canonical duplicates consume evidence budget.** Deduplication reduces mean duplicate rate@3 from `0.167` to `0.0`.
- **Temporal intent matters.** Historical queries must not be reranked as if newest evidence were always preferred.
- **The perfect policy result is controlled-mechanism evidence only.** The benchmark is tiny, frozen, and intentionally shaped around stale/authority/duplicate failures; weights were not tuned after test results.
- **Live-web coverage is intentionally not claimed.** Frozen snapshots make CI reproducible; later adapters can implement the same `Source` contract against real providers.

Evaluation evidence:

- Initial PR gate `32447107848`: **82 tests passed** plus successful Web RAG evaluation.
- Final source-of-truth gate `{run_id}` passed before this ROADMAP update on head `{head_sha}`.
- Retrieval/source metrics are evaluated independently from answer text.
- Machine-readable and human-readable artifacts persist rankings, source metadata, citations, latency, and per-query failure traces.

Artifacts: `benchmarks/m08_web/`, `src/rag_practice/sources/`, `src/rag_practice/web/`, `src/rag_practice/evaluation/web.py`, `labs/08_specialized_sources/web/`, and `.github/workflows/m08-web.yml`.
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

Continue **M08.2 — SQL / Structured RAG**. Reuse the shared `Source` boundary, but evaluate structured-source behavior separately: schema discovery, query construction/validation, execution correctness, row-level evidence/citations, empty/error handling, latency, and unnecessary scans. Keep SQL execution quality separate from final answer generation."""
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
        f"Status: **DONE** — final full-suite + Web RAG gate passed in GitHub Actions run `{run_id}`.",
    )
    if "Final source-of-truth gate" not in readme:
        marker = "Initial CI run `32447107848` passed the full repository suite (**82 tests**) and the Web RAG evaluator."
        readme = readme.replace(
            marker,
            marker
            + f"\n\nFinal source-of-truth gate `{run_id}` passed the same full-suite/evaluator sequence before this completion update.",
        )
    readme = readme.replace(
        "- [ ] ROADMAP marks Web RAG sub-lab DONE only after the final gate passes",
        "- [x] ROADMAP marks Web RAG sub-lab DONE only after the final gate passes",
    )
    readme = readme.replace(
        "Web RAG is not merged until the final unchecked gate passes.",
        "Web RAG satisfies the sub-lab evaluation contract and is eligible to merge; M08 overall remains IN PROGRESS.",
    )
    README.write_text(readme.rstrip() + "\n")


if __name__ == "__main__":
    main()
