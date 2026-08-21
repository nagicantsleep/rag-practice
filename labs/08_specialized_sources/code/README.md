# M08.4 — Code RAG

Status: **IN PROGRESS** — implementation/evaluation candidate pending CI evidence.

## Hypothesis

Flattening a repository into anonymous file text loses distinctions that matter for code retrieval: definition vs call site, duplicate symbol names, exact source locations, and cross-file dependencies. A code-aware RAG path should expose these structures without hiding them behind a framework.

## Mechanism

The sub-lab indexes a frozen Python repository in three ways:

1. whole-file BM25;
2. AST symbol BM25 over functions/classes/methods;
3. symbol-aware retrieval with explicit identifier boosts plus a small static call graph.

`PythonRepositoryIndex` uses the standard-library `ast` module to recover exact symbol spans and direct calls. Imported functions are resolved through inspectable import maps. Ambiguous attribute calls on local variables deliberately remain unresolved rather than inventing graph edges.

The symbol-aware path is exposed through the shared M08 `Source` contract and returns exact locators such as:

`code://repo/pricing/engine.py#L5-L9`

No Tree-sitter, LangChain, LlamaIndex, or LLM is used.

## Controlled benchmark

`benchmarks/m08_code/repo/` contains a small multi-package Python repository with:

- duplicate `parse_token` definitions in auth vs billing;
- duplicate `normalize` definitions;
- implementation vs wrapper/call-site collisions;
- checkout pricing spread across engine/discount/tax modules;
- forward dependency questions;
- reverse dependency/change-locality questions;
- class/method source-location lookup.

`queries.jsonl` defines 10 held-out retrieval questions with exact symbol qrels and a primary symbol used for top-1 location answers.

Research context: RepoBench (arXiv:2306.03091) separates repository retrieval from completion, while RepoCoder (arXiv:2303.12570) demonstrates iterative repository-level retrieval/generation. The newer CodeRAG work (arXiv:2509.16112) also emphasizes multi-path repository retrieval. This lab stays lower-level and retrieval-focused.

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

## Definition of Done

- [x] deterministic multi-file code benchmark defined
- [x] whole-file BM25 control implemented
- [x] AST symbol index and exact line locators implemented
- [x] duplicate symbol names represented explicitly
- [x] direct import/call graph implemented
- [x] reverse dependency expansion implemented for change-locality queries
- [x] retrieval/location/dependency metrics separated
- [x] regression tests added
- [ ] full repository CI + Code RAG evaluator passes
- [ ] persisted JSON/Markdown results reviewed
- [ ] representative implementation/call-site/dependency failures written down
- [ ] ROADMAP marks Code RAG DONE only after the final gate passes

This is a Python-only deterministic mechanism lab. It does not claim full type resolution, dynamic dispatch analysis, semantic code search, or RepoBench-scale generalization.
