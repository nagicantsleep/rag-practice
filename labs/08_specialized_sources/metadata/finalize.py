"""Finalize M08.3 only after the metadata/filter workflow has passed."""

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
- **metadata / filter-aware RAG — `DONE`**
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

Important SQL / structured findings: flat row retrieval is not relational execution; answer correctness and row provenance are separate contracts; empty results are first-class; unsafe/unsupported requests fail closed. Perfect scores are controlled frozen-schema evidence, not general text-to-SQL ability.

Artifacts: `benchmarks/m08_sql/`, `src/rag_practice/structured_sql/`, `src/rag_practice/evaluation/sql_structured.py`, `labs/08_specialized_sources/sql/`, and `.github/workflows/m08-sql.yml`.

### Metadata / Filter-aware RAG summary

Metadata / Filter-aware RAG holds BM25 text/scoring fixed and changes only where tenant/role/product/region/time predicates are applied.

| System | Recall@3 | Hit@1 | Constraint satisfied | Security leakage | Filter violation | Answer correct | Indexed records | Examined candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unfiltered BM25 | **1.000** | 0.250 | 0.333 | 0.444 | 0.667 | 0.222 | 19.0 | 3.0 |
| post-filter k=2 | 0.375 | 0.375 | **1.000** | **0.000** | **0.000** | 0.444 | 19.0 | 2.0 |
| post-filter oversample k=5 | **1.000** | **1.000** | **1.000** | **0.000** | **0.000** | **1.000** | 19.0 | 5.0 |
| pre-filter BM25 | **1.000** | **1.000** | **1.000** | **0.000** | **0.000** | **1.000** | **2.7** | **1.4** |

Important metadata/filter findings:

- **Recall is not a security metric.** Unfiltered Recall@3 is `1.0` while security leakage is `0.444` and explicit-filter violation is `0.667`.
- **Safe post-filtering can lose recall.** Candidate `k=2` removes invalid returned records but drops Recall@3 to `0.375` because invalid candidates consumed the ranking window.
- **Oversampling can recover quality at higher candidate cost.** `k=5` restores controlled quality but still ranks the global corpus and rejects many candidates after scoring.
- **Hard authorization belongs before relevance ranking.** Pre-filtering reaches the same controlled quality while ranking only eligible records in this benchmark.
- **Groundedness does not imply authorization.** Extractive answers remain grounded even when the source should never have been exposed.
- **Production IAM is not claimed.** Real systems need authoritative storage/index enforcement, policy/versioning, cache isolation, auditability, and side-channel defenses.

Evaluation evidence:

- Initial PR gate `32450243891`: **95 tests passed** plus successful metadata/filter evaluation.
- Final source-of-truth gate `{run_id}` passed before this ROADMAP update on head `{head_sha}`.
- Retrieval relevance, constraint correctness, security leakage, answer quality, and candidate cost are evaluated separately.
- Persisted JSON/Markdown includes rankings, filter context, metadata, eligibility, rejection counts, answers, and latency.

Artifacts: `benchmarks/m08_metadata/`, `src/rag_practice/metadata_filter/`, `src/rag_practice/evaluation/metadata_filter.py`, `labs/08_specialized_sources/metadata/`, and `.github/workflows/m08-metadata.yml`.
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

Continue **M08.4 — Code RAG**. Treat code structure as first-class retrieval metadata instead of flattening a repository into anonymous text: benchmark file/function/class/symbol lookup, implementation-vs-call-site retrieval, cross-file dependency context, duplicate identifiers, and change-locality. Compare plain text retrieval with symbol-aware/chunk-aware retrieval, evaluate exact source locations and dependency evidence separately from answer quality, and measure indexing/query cost."""
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
        f"Status: **DONE** — final full-suite + metadata/filter gate passed in GitHub Actions run `{run_id}`.",
    )
    marker = "Initial PR gate `32450243891` passed the full repository suite (**95 tests**) and the metadata/filter evaluator."
    if "Final source-of-truth gate" not in readme:
        readme = readme.replace(
            marker,
            marker + f"\n\nFinal source-of-truth gate `{run_id}` passed the same full-suite/evaluator sequence before this completion update.",
        )
    readme = readme.replace(
        "- [ ] ROADMAP marks metadata/filter-aware sub-lab DONE only after the final gate passes",
        "- [x] ROADMAP marks metadata/filter-aware sub-lab DONE only after the final gate passes",
    )
    readme = readme.replace(
        "Metadata / Filter-aware RAG is not merged until the final unchecked gate passes.",
        "Metadata / Filter-aware RAG satisfies the sub-lab evaluation contract and is eligible to merge; M08 overall remains IN PROGRESS.",
    )
    README.write_text(readme.rstrip() + "\n")


if __name__ == "__main__":
    main()
