"""Finalize M08.4 only after the Code RAG workflow has passed."""

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
    old_line = "- Code RAG — `TODO`"
    new_line = "- **Code RAG — `DONE`**"
    if old_line not in roadmap and new_line not in roadmap:
        raise SystemExit("expected Code RAG sub-lab line")
    roadmap = roadmap.replace(old_line, new_line, 1)

    if "### Code RAG summary" not in roadmap:
        summary = f"""
### Code RAG summary

Code RAG compares whole-file BM25, AST-symbol BM25, and symbol retrieval with explicit forward/reverse call-graph expansion over a frozen 13-file Python repository.

| System | Recall@4 | Complete@4 | Primary Hit@1 | Single-answer location | Dependency complete | Call-site confusion | Context chars@4 | Exact line locators |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| file BM25 | **1.000** | **1.000** | 0.500 | 0.571 | **1.000** | 1.000 | 1018.6 | 0.000 |
| symbol BM25 | 0.950 | 0.900 | 0.500 | 0.571 | 0.750 | 1.000 | **740.5** | **1.000** |
| symbol + graph | **1.000** | **1.000** | **0.800** | **1.000** | **1.000** | **0.000** | 748.0 | **1.000** |

Important Code RAG findings:

- **File recall is not exact code evidence.** Coarse files can cover the qrels while failing to identify the implementation or exact source span.
- **Symbol chunking alone can regress repository evidence.** Smaller AST units reduce context size and provide line locators, but isolated ranking loses cross-file dependency completeness.
- **Graph expansion repairs controlled repository relationships.** Forward edges recover callees and reverse edges recover change-locality callers while preserving exact AST locators.
- **Conservative resolution avoids false evidence.** Local-variable attribute calls such as `rates.get(...)` are not guessed into unrelated repository methods.
- **This is not semantic program analysis.** Results are for a tiny deterministic Python repository with direct import/call resolution only.

Evaluation evidence:

- First candidate gate `32452392799`: 101 tests passed and one stale line-span assertion failed; no evaluator result was accepted from that run.
- Initial successful PR gate `32452536862`: **102 tests passed** plus successful Code RAG evaluation.
- Final source-of-truth gate `{run_id}` passed before this ROADMAP update on head `{head_sha}`.
- File retrieval, exact symbol/location retrieval, dependency evidence, and context cost are evaluated separately.

Artifacts: `benchmarks/m08_code/`, `src/rag_practice/code_rag/`, `src/rag_practice/evaluation/code_rag.py`, `labs/08_specialized_sources/code/`, and `.github/workflows/m08-code.yml`.
"""
        marker = "\n### M09 — Agentic RAG — `TODO`"
        if marker not in roadmap:
            raise SystemExit("expected M09 marker")
        roadmap = roadmap.replace(marker, summary.rstrip() + marker, 1)

    next_step = """## Immediate next step

Continue **M08.5 — Multimodal RAG**. Keep modality-specific evidence explicit: build a controlled corpus where some answers are recoverable from text/alt-text metadata while others require image-native evidence; compare text-only surrogate retrieval with multimodal retrieval; evaluate retrieval relevance, modality coverage, evidence provenance, answer correctness, and cost separately. Do not treat captions/OCR as equivalent to visual understanding."""
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
        f"Status: **DONE** — final full-suite + Code RAG gate passed in GitHub Actions run `{run_id}`.",
    )
    marker = "Initial successful PR gate `32452536862` passed the full repository suite (**102 tests**) and the Code RAG evaluator."
    if "Final source-of-truth gate" not in readme:
        readme = readme.replace(
            marker,
            marker + f"\n\nFinal source-of-truth gate `{run_id}` passed the same full-suite/evaluator sequence before this completion update.",
        )
    readme = readme.replace(
        "- [ ] ROADMAP marks Code RAG DONE only after the final gate passes",
        "- [x] ROADMAP marks Code RAG DONE only after the final gate passes",
    )
    readme = readme.replace(
        "Code RAG is not merged until the final unchecked gate passes. This remains a Python-only deterministic mechanism lab.",
        "Code RAG satisfies the sub-lab evaluation contract and is eligible to merge; M08 overall remains IN PROGRESS. This remains a Python-only deterministic mechanism lab.",
    )
    README.write_text(readme.rstrip() + "\n")


if __name__ == "__main__":
    main()
