from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROADMAP = ROOT / "ROADMAP.md"
README = Path(__file__).resolve().parent / "README.md"


def finalize_roadmap() -> None:
    run_id = os.environ.get("GITHUB_RUN_ID", "unknown")
    head_sha = os.environ.get("GITHUB_SHA", "unknown")
    text = ROADMAP.read_text()

    m07 = f'''### M07 — Hierarchical, Graph, and Memory-oriented RAG — `DONE`

M07 isolates structural retrieval and temporal memory from generation so tree/graph/memory failures cannot be hidden by an LLM.

Implemented and evaluated:

- flat text-only BM25 and metadata-enriched BM25 controls on one shared 19-document corpus
- RAPTOR-style deterministic extractive hierarchy with leaf → group → collection-root routing
- KAG-style provenance-preserving relation/path retrieval
- GraphRAG-style directed community/global evidence expansion
- HippoRAG-style query-seeded personalized PageRank with multi-seed bridge scoring
- version-aware temporal memory supporting current and previous facts plus incremental updates
- LightRAG-style transparent low/high controller between path and global retrieval
- exact evidence-budget completeness, Recall@3/5/10, MRR, per-task traces, construction/query/update latency, structural footprint, and temporal freshness metrics

Predeclared static-system summary:

| System | Recall@3 | Recall@5 | Evidence complete@budget | MRR |
| --- | ---: | ---: | ---: | ---: |
| flat BM25 | 0.447 | 0.567 | 0.100 | 0.660 |
| flat metadata BM25 | 0.497 | 0.700 | 0.100 | 0.708 |
| RAPTOR-style hierarchy | 0.630 | 0.683 | 0.300 | 0.725 |
| KAG-style path | **0.717** | 0.717 | **0.600** | 0.800 |
| GraphRAG-style global | 0.580 | **0.767** | **0.600** | **0.808** |
| HippoRAG-style PPR | 0.630 | **0.767** | 0.200 | **0.808** |

Temporal-memory summary:

| System | Hit@1 | Current Hit@1 | Previous Hit@1 | Stale-current rate |
| --- | ---: | ---: | ---: | ---: |
| flat BM25 all versions | 0.250 | 0.000 | 1.000 | 1.000 |
| temporal memory | **1.000** | **1.000** | **1.000** | **0.000** |

Important findings:

- **No predeclared structure dominates every task class.** RAPTOR-style hierarchy is perfect on collection-wide hierarchical evidence; KAG-style paths are complete on local/multi-hop/associative relation paths; GraphRAG-style expansion is complete on global and hierarchical aggregation; HippoRAG-style PPR specifically solves the controlled associative bridge.
- **Evidence completeness is more diagnostic than an early relevant hit.** Flat BM25 reaches MRR `0.660` but exact Evidence Complete@budget only `0.100`.
- **Hierarchy is not graph reasoning.** The Atlas-country global query deliberately fails closed when its routed subgroup has no lexical bridge; a regression test prevents the earlier empty-index crash.
- **Path and community retrieval have opposite failure envelopes.** KAG-style paths do not enumerate broad global evidence, while GraphRAG-style expansion over-broadens entity-specific relation chains.
- **Associative diffusion is not ordered path execution.** HippoRAG-style propagation solves the two-seed association but does not complete controlled 3-hop paths inside the exact evidence budget.
- **Freshness needs explicit version policy.** Flat BM25 returns the old version first on every current-fact query; temporal memory reduces stale-current rate from `1.0` to `0.0` while retaining previous-version access.
- **The LightRAG-style controller is exploratory, not fresh held-out evidence.** It routes complementary low/high mechanisms and reaches Evidence Complete@budget `1.0` on this benchmark, but it was added after phase-1 result inspection. A fresh untouched benchmark or route-development split is required before claiming generalization.
- **These are controlled mechanism results.** The corpus is tiny/template-like, triples are gold annotations, RAPTOR grouping is deterministic, graph relation/entity recognition is rule-based, and latency numbers are CPU sanity measurements rather than serving claims.

Evaluation evidence:

- Final M07 completion gate run `{run_id}` succeeded on push head `{head_sha}` after the repaired workflow was active.
- The gate runs the full repository suite (**78 tests**) before the M07 hierarchy/graph/memory evaluator.
- Per-query rankings, task breakdowns, timings, structure sizes, freshness traces, route decisions, and the post-hoc caveat are persisted as JSON and Markdown.
- Generation/groundedness evaluation is **not applicable by design** because M07 intentionally isolates structured retrieval and temporal-memory quality from generation.

Artifacts: `benchmarks/m07_structured/`, `src/rag_practice/structured/`, `src/rag_practice/evaluation/structured.py`, `labs/07_hierarchical_graph_memory/`, and `.github/workflows/m07-structured.yml`.
'''

    pattern = r"### M07 — Hierarchical, Graph, and Memory-oriented RAG — `(?:TODO|IN PROGRESS|DONE)`\n.*?(?=\n### M08)"
    text, count = re.subn(pattern, m07.rstrip(), text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f"expected one M07 section, replaced {count}")

    next_step = '''## Immediate next step

Start **M08 — Specialized Sources and Modalities**. Keep source boundaries visible instead of hiding them behind one orchestration framework: begin with source/tool contracts and source-appropriate benchmarks for Web RAG, SQL/structured retrieval, metadata/filter-aware retrieval, and Code RAG; then add multimodal/visual-document and long-context-vs-retrieval routing. For every sub-lab, evaluate source/retrieval success independently from final answer quality, and record freshness, latency, token/tool cost, and source-specific failure modes.'''
    text, count = re.subn(r"## Immediate next step\n\n.*\Z", next_step, text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f"expected one immediate-next-step section, replaced {count}")

    ROADMAP.write_text(text.rstrip() + "\n")


def finalize_readme() -> None:
    run_id = os.environ.get("GITHUB_RUN_ID", "unknown")
    text = README.read_text()
    text = re.sub(
        r"Status: \*\*COMPLETION CANDIDATE\*\* — final full-suite/evaluation gate pending\.",
        f"Status: **DONE** — final full-suite + M07 evaluation gate passed in GitHub Actions run `{run_id}`.",
        text,
        count=1,
    )
    text = re.sub(
        r"Final-gate trigger note:.*?\n\n",
        "",
        text,
        count=1,
        flags=re.S,
    )
    text = text.replace(
        "- [ ] final completion tree passes full repository CI + M07 evaluation and ROADMAP is updated on that successful run",
        "- [x] final completion tree passes full repository CI + M07 evaluation and ROADMAP is updated from that successful run",
    )
    text = text.replace(
        "M07 is not merged until the final unchecked gate passes.",
        "M07 satisfies the evaluation contract and is eligible to merge.",
    )
    README.write_text(text.rstrip() + "\n")


def main() -> None:
    finalize_roadmap()
    finalize_readme()


if __name__ == "__main__":
    main()
