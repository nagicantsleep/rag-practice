from __future__ import annotations

import os
from pathlib import Path

ROADMAP = Path("ROADMAP.md")
LAB_README = Path("labs/08_specialized_sources/long_context/README.md")

run_id = os.environ["GITHUB_RUN_ID"]
head_sha = os.environ["GITHUB_SHA"]

roadmap = ROADMAP.read_text()
readme = LAB_README.read_text()

if "### Long-context vs retrieval routing summary" not in roadmap:
    roadmap = roadmap.replace(
        "### M08 — Specialized Sources and Modalities — `IN PROGRESS`",
        "### M08 — Specialized Sources and Modalities — `DONE`",
        1,
    )
    roadmap = roadmap.replace(
        "- long-context vs retrieval routing — `TODO`",
        "- **long-context vs retrieval routing — `DONE`**",
        1,
    )

    marker = "\n### M09 — Agentic RAG — `TODO`"
    assert marker in roadmap
    summary = f'''\n### Long-context vs retrieval routing summary\n\nM08.7 freezes one 4-bundle/12-query benchmark before pretrained inspection and compares direct full-context reading, fixed-budget BM25 retrieval, and an explicit qrel-blind router while separating route quality, evidence completeness, reader correctness, grounding, abstention, context footprint, retrieval calls, and latency.\n\n| System | Route acc | Evidence complete | Answer acc | Grounded | Abstention | Context words | Retrieval calls |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n| deterministic always direct | 0.583 | **1.000** | **1.000** | **1.000** | **1.000** | 490.5 | **0.00** |\n| deterministic always retrieve | 0.417 | 0.700 | 0.750 | **1.000** | **1.000** | **100.2** | 1.00 |\n| deterministic explicit router | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 275.5 | 0.42 |\n| SmolLM2 explicit router | **1.000** | **1.000** | 0.000 | 0.083 | 0.000 | 275.5 | 0.42 |\n\nImportant long-context findings:\n\n- **Full-context reading is a quality ceiling with a context cost.** It preserves complete evidence on this frozen benchmark but wastes full-context budget on retrieval-preferred sparse-fact queries.\n- **Retrieval can save context while destroying global evidence completeness.** Frozen BM25 top-2 reduces selected context sharply but misses sections needed by Atlas/Orion global count/list tasks.\n- **The explicit router is a mechanism control, not learned generalization.** Frozen bundle-size and global-language rules recover the declared route boundary without looking at qrels or reader outputs.\n- **Correct routing does not imply reader competence.** Pinned `HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac` gets complete evidence under the explicit router but strict raw answer accuracy remains `0.0`.\n- **Reader failures include both formatting and semantics.** The frozen strict metric retains verbose fact answers, wrong comparison/list/count answers, and hallucinated answers on both no-evidence cases rather than adding expected-answer-aware cleanup.\n- **Routing changes cost independently from quality.** The pretrained explicit route sits between always-direct and always-retrieve in prompt size and CPU generation time while preserving deterministic evidence completeness.\n- **This remains controlled evidence.** The benchmark is tiny/synthetic and the pretrained reader is one small pinned model; neither result is a general long-context leaderboard claim.\n\nEvaluation evidence:\n\n- Benchmark frozen before pretrained inspection in commit `b018a52b112f113ad18447bfc8ab862b5ccded98`.\n- Repaired deterministic gate `32481131658` / job `96767529712`: **121 tests passed** and deterministic evaluation passed.\n- Full pinned pretrained gate `32481464647` / job `96768559380`: **124 tests passed**, deterministic evaluation passed, and pinned SmolLM2 evaluation passed.\n- Final source-of-truth push gate run `{run_id}` passed on head `{head_sha}` before this automated `[skip ci]` completion update; the finalizer executes only after full tests, deterministic evaluation, and pinned SmolLM2 evaluation succeed.\n\nArtifacts: `benchmarks/m08_long_context/`, `src/rag_practice/long_context/`, `src/rag_practice/evaluation/long_context.py`, `src/rag_practice/evaluation/long_context_pretrained.py`, `labs/08_specialized_sources/long_context/`, and `.github/workflows/m08-long-context.yml`.\n'''
    roadmap = roadmap.replace(marker, summary + marker, 1)

    old_next = "Continue **M08.7 — Long-context vs retrieval routing**. Freeze a benchmark with short, long, and mixed-context tasks before tuning; compare direct long-context reading, retrieval-first RAG, and an explicit routing policy on the same evidence. Evaluate route correctness, retrieval/evidence completeness, answer correctness and grounding, unnecessary retrieval/context use, latency, token/context footprint, abstention, and the failure boundary where full-context reading should replace or defer to retrieval."
    assert old_next in roadmap
    roadmap = roadmap.replace(
        old_next,
        "Continue **M09 — Agentic RAG**. Implement planner/search strategy, source and tool routing, retrieval loops, evidence evaluation, retry/stop policy, and explicit state before comparing multi-agent variants. Evaluate task success, tool precision, unnecessary actions, recovery, grounding, latency, and cost independently.",
        1,
    )

assert "Status: **EVIDENCE RECORDED — FINAL GATE PENDING**" in readme
readme = readme.replace(
    "Status: **EVIDENCE RECORDED — FINAL GATE PENDING**",
    "Status: **DONE**",
    1,
)
readme = readme.replace(
    "- [ ] Pass the final source-of-truth full-regression + deterministic + pretrained evaluation gate on the findings head.",
    "- [x] Pass the final source-of-truth full-regression + deterministic + pretrained evaluation gate on the findings head.",
    1,
)
readme = readme.replace(
    "- [ ] Mark M08.7 and M08 complete in ROADMAP.",
    "- [x] Mark M08.7 and M08 complete in ROADMAP.",
    1,
)
needle = "- A final source-of-truth full-regression + deterministic + pretrained gate must pass on the findings/ROADMAP head before M08.7 or M08 is marked `DONE`."
assert needle in readme
readme = readme.replace(
    needle,
    f"- Final source-of-truth push gate run `{run_id}` passed on head `{head_sha}`: full tests, deterministic evaluation, and pinned SmolLM2 evaluation all succeeded before this automated `[skip ci]` completion update.",
    1,
)

ROADMAP.write_text(roadmap)
LAB_README.write_text(readme)
