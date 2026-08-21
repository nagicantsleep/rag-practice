# M08.4 — Code RAG

Status: **DONE** — final full-suite + Code RAG gate passed in GitHub Actions run `32455642731`.

## Hypothesis

Flattening a repository into anonymous file text loses distinctions that matter for code retrieval: definition vs call site, duplicate symbol names, exact source locations, and cross-file dependencies. A code-aware RAG path should expose these structures without hiding them behind a framework.

## Mechanism

The sub-lab indexes a frozen Python repository in three ways:

1. whole-file BM25;
2. AST symbol BM25 over functions/classes/methods;
3. symbol-aware retrieval with explicit identifier boosts plus a small static call graph.

`PythonRepositoryIndex` uses the standard-library `ast` module to recover exact symbol spans and direct calls. Imported functions are resolved through inspectable import maps. Ambiguous attribute calls on local variables deliberately remain unresolved rather than inventing graph edges.

The symbol-aware path is exposed through the shared M08 `Source` contract and returns exact locators such as `code://repo/pricing/engine.py#L6-L10`.

No Tree-sitter, LangChain, LlamaIndex, or LLM is used.

## Controlled benchmark

`benchmarks/m08_code/repo/` contains 13 Python files, 17 AST symbols, 5 resolved call edges, and 10 held-out queries covering:

- duplicate `parse_token` and `normalize` definitions;
- implementation vs wrapper/call-site collisions;
- checkout pricing spread across engine/discount/tax modules;
- forward dependency questions;
- reverse dependency/change-locality questions;
- class/method source-location lookup.

`queries.jsonl` defines exact symbol qrels and a primary symbol used for top-1 location answers.

Research context: RepoBench separates repository retrieval from completion, while RepoCoder demonstrates iterative repository-level retrieval/generation. This lab stays lower-level and retrieval-focused.

## Evaluation contract

Retrieval/source quality:
- Recall@4;
- exact evidence completeness@4;
- primary implementation Hit@1;
- dependency completeness;
- implementation-vs-call-site confusion;
- duplicate-symbol confusion.

Answer/location quality:
- exact top-1 location answer for single-evidence queries;
- AST-derived `code://...#Lx-Ly` source locators.

System behavior:
- indexed units;
- retrieved context characters;
- index build/query latency;
- persisted call graph.

File and symbol retrieval units are reported separately. A file-level hit is not treated as an exact function/line hit.

## Evaluation evidence

Initial successful PR gate `32452536862` passed the full repository suite (**102 tests**) and the Code RAG evaluator.

Final source-of-truth gate `32455642731` passed the same full-suite/evaluator sequence before this completion update.

An earlier gate `32452392799` failed with **101 passed, 1 failed** because the test expected the prototype line span `L10-L12` after the frozen benchmark had been formatted to `L11-L13`. The AST index returned the frozen source span correctly; only the assertion was corrected. Retrieval code, graph scoring, benchmark queries, and qrels were unchanged.

| System | Recall@4 | Complete@4 | Primary Hit@1 | Single-answer location | Dependency complete | Call-site confusion | Context chars@4 | Exact line locators |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| file BM25 | **1.000** | **1.000** | 0.500 | 0.571 | **1.000** | 1.000 | 1018.6 | 0.000 |
| symbol BM25 | 0.950 | 0.900 | 0.500 | 0.571 | 0.750 | 1.000 | **740.5** | **1.000** |
| symbol + graph | **1.000** | **1.000** | **0.800** | **1.000** | **1.000** | **0.000** | 748.0 | **1.000** |

## Error analysis and findings

- **File-level recall can hide source-location failure.** Whole-file BM25 reaches Recall/Complete@4 = `1.0`, yet it has no exact line locators, only `0.571` single-answer location accuracy, and ranks the call site above the implementation on controlled implementation-vs-call-site queries.
- **Symbol chunking alone is a useful negative result.** It reduces mean retrieved context from about `1019` to `741` characters and exposes exact spans, but Recall@4 falls to `0.950`, Complete@4 to `0.900`, and dependency completeness to `0.750` because independently ranked symbols do not reconstruct repository relationships.
- **Structure needs routing/expansion, not only smaller chunks.** Symbol + graph restores Recall/Complete/Dependency@4 to `1.0`, raises primary Hit@1 to `0.800`, reaches exact single-answer location `1.0`, and removes controlled call-site confusion.
- **Reverse edges matter for maintenance questions.** The change-locality query asking which caller must change after renaming `apply_discount` requires reverse call-graph expansion, not merely lexical similarity to the renamed definition.
- **Conservative graph construction is preferable to invented edges.** `rates.get(...)` is a local dictionary call and is deliberately not resolved to the unrelated unique method `Cache.get`; the regression test locks this behavior.
- **Perfect graph metrics are controlled-mechanism evidence only.** The graph resolves direct Python imports/calls in a tiny deterministic repository; it does not claim type-aware dispatch, dynamic language analysis, semantic code search, multilingual support, or RepoBench-scale generalization.

Machine-readable evidence: `results/results.json`. Human-readable aggregate table: `results/results.md`.

## Definition of Done

- [x] deterministic multi-file code benchmark defined
- [x] whole-file BM25 control implemented
- [x] AST symbol index and exact line locators implemented
- [x] duplicate symbol names represented explicitly
- [x] direct import/call graph implemented
- [x] reverse dependency expansion implemented for change-locality queries
- [x] retrieval/location/dependency metrics separated
- [x] regression tests added
- [x] full repository CI + Code RAG evaluator passes
- [x] persisted JSON/Markdown results reviewed
- [x] representative implementation/call-site/dependency failures written down
- [x] ROADMAP marks Code RAG DONE only after the final gate passes

Code RAG satisfies the sub-lab evaluation contract and is eligible to merge; M08 overall remains IN PROGRESS. This remains a Python-only deterministic mechanism lab.
