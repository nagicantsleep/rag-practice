# RAG Practice — Learning Roadmap

This file is the source of truth for the learning sequence and completion status in this repository.

## Goal

Learn Retrieval-Augmented Generation by implementing mechanisms directly, evaluating them on shared benchmarks, comparing them with explicit baselines, and only then introducing higher-level frameworks or production abstractions.

## Non-negotiable learning contract

1. **Evaluation is mandatory.** A demo or working pipeline without a baseline, benchmark, metrics, saved results, and error analysis is incomplete.
2. **Baseline before improvement.** New methods must be compared with the simplest relevant prior implementation.
3. **Shared benchmarks where tasks overlap.** Do not change both method and benchmark and then attribute the difference to the method.
4. **Mechanisms before frameworks.** Early labs keep retrieval/ranking/control logic visible instead of hiding it behind orchestration frameworks.
5. **Observable pipelines.** Trace transformed queries, retrieval scores, selected context, answer, citations, metrics, latency, tokens, and cost where applicable.
6. **Reproducibility.** Record dataset/model versions, chunking, retrieval parameters, seeds where applicable, and evaluator configuration.

## Evaluation contract for every lab

Before a lab can be `DONE`, it must define and save hypothesis, baseline, benchmark, relevant retrieval/generation/system metrics, error analysis, and machine-readable plus human-readable result artifacts.

A key rule: **retrieval quality must be evaluated independently from final generation whenever retrieval is involved.** A language model must not be allowed to hide retrieval failures.

## Definition of Done

- [ ] learning objective written
- [ ] algorithm/pipeline implemented
- [ ] core behavior covered by automated tests
- [ ] baseline identified and runnable
- [ ] benchmark/evaluation dataset defined
- [ ] retrieval evaluation present when retrieval is involved
- [ ] generation/groundedness evaluation present when generation is involved
- [ ] relevant system metrics recorded
- [ ] configuration reproducible
- [ ] results saved, not only printed
- [ ] representative failures inspected
- [ ] findings/trade-offs written down

## Milestones

Statuses: `TODO`, `IN PROGRESS`, `DONE`.

### M00 — Information Retrieval Fundamentals — `DONE`

Implemented minimal tokenization/inverted index, TF-IDF, dense/sparse cosine, BM25, top-k retrieval, relevance judgments, Hit Rate/Precision/Recall/MRR/MAP/nDCG, deterministic benchmark, hand-checkable metric tests, baseline comparison, and failure analysis.

Key finding: lexical retrieval fails a vocabulary-mismatch paraphrase. That failure is retained as a regression target.

### M01 — Naive RAG From Scratch — `DONE`

Implemented the inspectable pipeline:

```text
documents → fixed chunks → embeddings → in-memory vector index
          → top-k retrieval → context/prompt → generator → answer + citations
```

Includes data models, fixed-size chunking, hashing embeddings, vector index, generator abstraction, extractive generator, context/prompt construction, citations, tracing, retrieval evaluation, answer/grounding/citation evaluation, latency/token measurements, BM25 baseline, and no-retrieval baseline.

Key finding: a RAG answer can be **fully grounded but wrong** when retrieval selects the wrong evidence.

### M02 — Retrieval Families — `DONE`

M02 compares retrieval families under the same held-out exact/semantic benchmark and separates representation quality from storage/index mechanics.

Implemented and evaluated:

- BM25 lexical retrieval
- hashing-vector dense-storage mechanics baseline
- supervised neural dual encoder trained from scratch
- Reciprocal Rank Fusion (RRF)
- weighted sparse+dense score fusion tuned only on a separate dev split
- transparent SPLADE-style learned-sparse mechanics
- transparent ColBERT-style MaxSim mechanics
- pretrained `sentence-transformers/all-MiniLM-L6-v2`
- full pretrained SPLADE-family `naver/splade-v3-distilbert`
- full pretrained `colbert-ir/colbertv2.0` via PyLate exhaustive MaxSim
- deterministic 10 → 100 → 1000 candidate-set scaling stress test
- representation/index footprint and CPU latency sanity measurements

Held-out checkpoint summary:

| Method | Recall@1 | Exact R@1 | Semantic R@1 | Recall@3 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.800 | 1.000 | 0.600 | 0.900 | 0.850 |
| Hashing vector | 0.800 | 1.000 | 0.600 | 0.800 | 0.835 |
| Tiny neural dual encoder | 0.800 | 0.800 | 0.800 | 1.000 | — |
| Tiny weighted hybrid | **0.900** | **1.000** | **0.800** | **1.000** | — |
| MiniLM pretrained dense | **0.900** | **1.000** | **0.800** | 0.900 | 0.925 |
| SPLADE pretrained sparse | **0.900** | **1.000** | **0.800** | **1.000** | **0.950** |
| ColBERTv2 pretrained late interaction | **0.900** | **1.000** | **0.800** | **1.000** | **0.950** |

Important findings:

- **A dense vector index does not create semantics; learned/pretrained representations do.** The hashing baseline remains lexical-like, while learned models recover vocabulary-mismatch queries.
- **Aggregate scores hide different errors.** MiniLM fixes preserved query `s1` (`conceptual likeness between paraphrases → d5`) but misses `s2`; SPLADE and ColBERTv2 rank `s1` second but solve `s2` at rank 1.
- **Pretrained contextual backbones matter.** The earlier mechanism-only SPLADE/ColBERT implementations underperform their real checkpoints; the formulas alone are not the research systems.
- **Hybrid is not automatically better.** With MiniLM, dev-set tuning selects BM25 weight `0.0`; adding sparse score hurts or adds no value on this tiny dev set. The earlier from-scratch neural model did benefit from sparse+dense fusion.
- **Representations have different serving costs.** Dense stores one vector/document; SPLADE stores sparse vocabulary activations; ColBERT stores many token vectors/document. ColBERTv2 used 211 document token vectors / 108,032 logical embedding bytes on the 10-document corpus; SPLADE averaged 183.2 non-zero values/document.
- **Candidate-set size matters.** In the deterministic stress test, MiniLM quality stays at Recall@1 `0.9` from 10 to 1000 docs while the educational Python hashing scan falls from `0.8` to `0.7` and grows from roughly `0.4 ms` to `31 ms/query`. These are implementation sanity measurements, not ANN serving claims.

Evaluation evidence:

- Final GitHub Actions PR run `32395089427` completed successfully.
- Final CI run: **32 tests passed**; pretrained dense, SPLADE checkpoint, scaling stress test, PyLate installation, and ColBERTv2 checkpoint steps all succeeded.
- Model revisions are resolved/pinned in result artifacts.
- Generation evaluation is **not applicable** to M02 because this milestone intentionally isolates retrieval quality; M01 already established separate end-to-end generation evaluation.

Artifacts: `benchmarks/m02_retrieval/`, `src/rag_practice/retrieval/`, `labs/02_retrieval_families/`, and `.github/workflows/m02-pretrained-eval.yml`.

### M03 — Indexing and Chunking — `DONE`

M03 holds BM25 scoring fixed and changes only chunk/index representation so boundary, metadata, parent expansion, and hierarchy effects remain attributable.

Implemented and evaluated:

- fixed 24-word chunks without overlap
- fixed 24-word chunks with 8-word overlap
- sentence-aware packing
- paragraph-aware packing
- deterministic sentence-boundary similarity chunking
- metadata-enriched sentence chunks
- sentence-child retrieval with paragraph-parent expansion
- document-level metadata+body routing followed by plain sentence-leaf retrieval
- evidence completeness, context redundancy/utilization, relevant-context fraction, route accuracy, searchable-index footprint, build latency, and query-latency sanity measurements

Phase-1 held-out summary:

| Strategy | Doc Hit@1 | Evidence@1 | Evidence@3 | Source-token utilization@3 |
| --- | ---: | ---: | ---: | ---: |
| Fixed 24 | **1.000** | 0.200 | 0.800 | **1.000** |
| Fixed 24 + overlap 8 | **1.000** | 0.400 | **1.000** | 0.886 |
| Sentence 35 | 0.800 | **0.800** | **1.000** | **1.000** |
| Paragraph 80 | 0.800 | **0.800** | **1.000** | **1.000** |
| Hashing-similarity semantic boundaries | 0.800 | 0.200 | 0.800 | **1.000** |
| Sentence 35 + metadata | **1.000** | **0.800** | **1.000** | 0.635 |

Phase-2 held-out summary:

| Strategy | Doc Hit@1 | Evidence@1 | Evidence@3 | Utilization@3 | Searchable words | Route Hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed 24 + overlap 8 | **1.000** | 0.400 | **1.000** | 0.886 | 367 | — |
| Sentence 35 + metadata | **1.000** | 0.800 | **1.000** | 0.635 | 434 | — |
| Parent-child | 0.800 | 0.600 | **1.000** | **1.000** | **271** | — |
| Hierarchical metadata root → plain sentence leaf | **1.000** | **1.000** | **1.000** | **1.000** | 601 | **1.000** |

Important findings:

- **Document hit is not evidence completeness.** Fixed 24-word chunks retrieve the right document for every query at rank 1 but contain all required evidence at rank 1 only 20% of the time.
- **Overlap buys coverage with context duplication.** Eight-word overlap lifts Evidence@3 from `0.8` to `1.0`, while source-token utilization falls from `1.0` to `0.886`.
- **Natural boundaries help evidence packaging on this corpus.** Sentence and paragraph packing reach Evidence@1 `0.8`, but both rank the Tropical deployment above the Arctic deployment for the metadata-dependent query because the distinguishing region is absent from body text.
- **Repeating metadata in every leaf works but spends context budget.** Metadata-enriched sentence chunks restore Doc Hit@1 `1.0` while utilization drops to `0.635`.
- **A simplistic “semantic” boundary heuristic is not automatically better.** The deterministic hashing-similarity strategy over-splits to 19 chunks and falls back to Evidence@1 `0.2`; it is retained as a negative result rather than tuned away.
- **Parent-child expansion is useful but not sufficient for routing.** It reaches Evidence@1 `0.6` with perfect utilization and the smallest searchable representation in phase 2, but still selects the Tropical parent first for the Arctic query because metadata is absent from the child search layer.
- **Metadata belongs naturally in a routing layer when answer context should stay clean.** Hierarchical metadata+body roots route every held-out query correctly, then return plain sentence leaves with Doc Hit@1/Evidence@1/utilization all `1.0`. The trade-off is a larger searchable representation: 601 words versus 367 for the overlap baseline and 434 for metadata-enriched flat chunks.
- **These results are controlled-mechanism evidence, not a universal chunking leaderboard.** The benchmark is intentionally tiny and timings are GitHub Actions CPU sanity measurements.

Evaluation evidence:

- Final phase-2 GitHub Actions PR run `32407289218` completed successfully.
- Final CI run: **39 tests passed**; phase-1 and phase-2 evaluation steps both succeeded.
- Results are persisted as JSON and Markdown; representative Arctic/Tropical ambiguity and semantic over-splitting failures are retained in the reports.
- Generation evaluation is **not applicable** to M03 because this milestone isolates retrieval/index representation and context-selection quality.

Artifacts: `benchmarks/m03_chunking/`, `src/rag_practice/indexing/`, `src/rag_practice/evaluation/chunking.py`, `labs/03_indexing_chunking/`, and `.github/workflows/m03-indexing-chunking.yml`.

### M04 — Reranking and Context Construction — `DONE`

M04 freezes the first-stage candidate set before reranking, then separates ranking, diversity, packing, ordering, and generation so improvements are attributable to the correct stage.

Implemented and evaluated:

- retrieve-many/rerank-few with candidate recall measured before reranking
- pretrained cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2`
- pointwise instruction reranking with pinned `google/flan-t5-small`
- MMR diversity selection with explicit relevance/diversity trade-off
- source-span overlap rejection and fixed-word-budget context packing
- relevance order, source order, and edge-biased context ordering
- qrel-blind deterministic extractive answer generation with citations
- pinned FLAN answer generation using only question + ordered context
- answer correctness, grounded-token recall, citation precision/recall, context density/utilization, and CPU latency
- candidate-depth `k=2,4,6` latency-quality sweep

Phase-1 summary with frozen BM25 top-6 candidate document/evidence recall both `1.0`:

| Method | Evidence@1 | Evidence@3 | Source util@3 | Relevant context@3 |
| --- | ---: | ---: | ---: | ---: |
| BM25 first-stage | 0.800 | 1.000 | 0.635 | 0.742 |
| Cross-encoder | 0.800 | 1.000 | 0.632 | 0.814 |
| Cross-encoder + MMR | 0.800 | 1.000 | 0.639 | 0.799 |
| Cross-encoder + 100-word packing | 0.800 | 1.000 | **0.654** | **0.904** |

Generation/context summary:

| Policy | Relevant context@3 | Mean context words | Extractive F1 | FLAN F1 | FLAN grounded-token recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| first-stage top-3 | 0.742 | 112.2 | 0.331 | 0.382 | 0.775 |
| cross-encoder top-3 | 0.814 | 111.6 | **0.387** | **0.700** | 0.775 |
| cross + pack 100 | **0.904** | 84.4 | **0.387** | 0.452 | 0.575 |
| pointwise FLAN rerank + pack 100 | 0.819 | 83.2 | **0.387** | 0.452 | 0.575 |

Important findings:

- **Candidate recall is a reranking ceiling.** Candidate document/evidence recall is recorded before any reranker so a missing passage cannot be credited to reranking.
- **Reranking can improve generation without changing Evidence@K.** Evidence@3 is already `1.0`, yet cross-encoder ordering raises relevant-context fraction from `0.742` to `0.814` and FLAN answer F1 from `0.382` to `0.700`.
- **Denser context is not automatically better for the generator.** Packing lowers average context from `111.6` to `84.4` words and raises relevant-context fraction to `0.904`, but FLAN F1 falls to `0.452`. The fixed 100-word budget is retained rather than tuned on the test set.
- **Generator robustness must be evaluated separately from context sufficiency.** For the overlapping-chunk and natural-boundaries questions, FLAN-T5-small sometimes emits bracket labels such as `[2]`/`[1]` even though relevant evidence is present. This is retained as a prompt/model failure.
- **The simple pointwise instruction reranker is slower and not better here.** It costs roughly `323 ms/query` on CPU versus roughly `68 ms/query` for the cross-encoder and yields lower packed relevant-context fraction (`0.819` vs `0.904`).
- **More candidates are not automatically better.** Candidate `k=2` already has document/evidence recall `1.0` and Evidence@3 `1.0`; increasing to `k=6` roughly doubles reranking latency without improving those metrics on this corpus.
- **Ordering alone can change rank-sensitive behavior.** Source-order and edge-order reuse exactly the same packed candidate set. This tiny benchmark shows no universal FLAN ordering winner, while source order can reduce rank-1 evidence/extractive quality.
- **Groundedness metrics have semantics.** Exact-token grounded recall is reproducible but can penalize semantically supported paraphrases; it is not treated as a semantic faithfulness oracle.

Evaluation evidence:

- GitHub Actions M04 run `32410053158` completed successfully on the completed mechanism/evaluation tree.
- Final mechanism/evaluation run: **51 tests passed**; phase 1, phase 2, and candidate-depth sweep all succeeded.
- Phase-1, phase-2, and depth-sweep results are persisted as JSON and Markdown on the feature branch.
- Model revisions are pinned/resolved in artifacts.
- Completion summary retains representative ranking wins, generator failures, negative results, metric limitations, and latency-quality trade-offs.

Artifacts: `benchmarks/m04_context/`, `src/rag_practice/reranking/`, `src/rag_practice/models/flan_t5.py`, `src/rag_practice/generation/query_extract.py`, `labs/04_reranking_context/`, and `.github/workflows/m04-reranking-context.yml`.

### M05 — Query Transformation — `DONE`

M05 freezes the corpus and retriever family within each comparison and changes only query-side representation/control so retrieval differences remain attributable to query transformation.

Implemented and evaluated:

- single generative query rewrite
- multi-query retrieval with original-query retention
- normalized score fusion
- RAG-Fusion via Reciprocal Rank Fusion
- Query2Doc-style pseudo-document expansion
- HyDE using the same fixed MiniLM dense retriever/index as its original-query baseline
- query decomposition + RRF
- query classes: exact, semantic/vocabulary mismatch, underspecified, and multi-aspect
- `complete_recall@3` to require all evidence for multi-aspect information needs
- transformation latency, end-to-end retrieval latency, generated-word cost proxy, and persisted per-query transformation traces
- capacity control using the exact same prompts/benchmark/retrievers with FLAN-T5-small versus FLAN-T5-base

FLAN-T5-small summary:

| Method | Recall@1 | Recall@3 | Complete R@3 |
| --- | ---: | ---: | ---: |
| BM25 original | 0.792 | 0.875 | 0.833 |
| Rewrite | 0.792 | 0.875 | 0.833 |
| Multi-query score fusion | 0.792 | 0.875 | 0.833 |
| RAG-Fusion / RRF | 0.792 | 0.875 | 0.833 |
| Query2Doc + BM25 | 0.792 | 0.875 | 0.833 |
| Decomposition + RRF | 0.167 | 0.250 | 0.167 |
| Dense original | 0.792 | **0.917** | **0.917** |
| HyDE + dense | 0.625 | 0.833 | 0.750 |

FLAN-T5-base capacity-control summary:

| Method | Recall@1 | Recall@3 | Complete R@3 |
| --- | ---: | ---: | ---: |
| BM25 original | **0.792** | 0.875 | 0.833 |
| Rewrite | **0.792** | 0.875 | 0.833 |
| Multi-query score fusion | **0.792** | 0.875 | 0.833 |
| RAG-Fusion / RRF | **0.792** | 0.875 | 0.833 |
| Query2Doc + BM25 | **0.792** | 0.833 | 0.750 |
| Decomposition + RRF | 0.625 | 0.667 | 0.667 |
| Dense original | **0.792** | **0.917** | **0.917** |
| HyDE + dense | 0.708 | **0.917** | **0.917** |

Important findings:

- **Query transformation is not a free quality upgrade.** No generative method beats its fair original-query baseline on aggregate in this controlled benchmark.
- **Keeping the original query is a useful safety measure but not an improvement guarantee.** Rewrite/multi-query/RAG-Fusion preserve BM25 quality here while adding generation latency.
- **Transformer capacity matters, but reliability still dominates some failures.** FLAN-T5-base raises decomposition R@1 from `0.167` to `0.625` and HyDE from `0.625` to `0.708`, yet neither beats its original-query baseline and malformed outputs remain.
- **Query2Doc can reduce evidence completeness.** Under the base control it drops R@3/complete R@3 relative to BM25 original, especially on multi-aspect needs.
- **Dense representation solves the preserved vocabulary-mismatch case more directly than these lexical rewrites.** The M00 paraphrase failure remains missed by BM25 transformations while the pretrained dense original query retrieves the relevant document first.
- **Decomposition should be routed, not universal.** Bad subquestions can destroy retrieval, directly motivating M06 adaptive routing/correction.
- **HyDE is sensitive to hypothetical-document drift.** A fluent pseudo-document can move the embedding away from useful evidence; stronger generation does not guarantee better rank-1 retrieval.
- **Transformation cost belongs in the policy.** Generative paths add hundreds to thousands of CPU milliseconds in these runs versus near-zero BM25 and low-millisecond dense retrieval, so a transformation must earn its cost through measurable quality gains.
- **This is a valid negative-result milestone.** The methods are complete because mechanisms, controls, baselines, benchmark, metrics, capacity control, costs, failures, and reproducible artifacts are all recorded—not because a transformed query was forced to win.

Evaluation evidence:

- GitHub Actions M05 capacity-control PR run `32413220357` completed successfully on the mechanism tree.
- Capacity-control mechanism run: **59 tests passed**; FLAN-T5-small baseline and FLAN-T5-base evaluation both succeeded.
- Final completion-tree run `32414426515` also passed 59 tests plus both FLAN-small and FLAN-base evaluations before merge.
- Model revisions, transformed queries, rankings, class breakdowns, costs, and representative drift failures are persisted as JSON and Markdown.
- Generation answer-quality evaluation is **not applicable** to M05 because generation is intentionally used only as a query-transformation mechanism; retrieval effects are evaluated independently.

Artifacts: `benchmarks/m05_query_transform/`, `src/rag_practice/query_transform/`, `labs/05_query_transform/`, and `.github/workflows/m05-query-transform.yml`.

### M06 — Multi-hop, Active, Adaptive, and Self-correcting RAG — `DONE`

M06 adds explicit control flow after M05 showed that expensive transformations should not run unconditionally. It separates **route selection**, **iterative/corrective retrieval**, **generation**, and **post-generation reflection** so failures can be attributed to the right layer.

Implemented and evaluated:

- from-scratch multinomial Naive Bayes router over separate training data for `no_retrieval`, `single`, and `iterative` routes
- transparent keyword-router baseline and always-single baseline
- two-hop bridge planning from newly discovered entities plus unresolved query terms
- qrel-blind lexical retrieval-quality judge
- controlled CRAG-style fallback for explicitly stale primary evidence
- FLARE-style active retrieval using pinned FLAN token-probability confidence as an explicit trigger signal
- Self-RAG-style separate retrieve/relevance/support/utility reflection primitives
- one-retry active retrieval + refusal path
- deliberately unanswerable held-out questions for unsupported-answer/refusal evaluation
- per-step control traces, generation confidence, reflection signals, latency, retrieval calls, attempts, prompt/output word proxies, and persisted JSON/Markdown reports

Phase-1 control summary:

| System | Route acc | Evidence recall | Evidence complete | Mean calls | Unnecessary retrieval | Iterative under-route | Correction P | Correction R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_single | 0.583 | 0.812 | 0.625 | 1.33 | 1.000 | 1.000 | 0.500 | 1.000 |
| keyword_router | 1.000 | 1.000 | 1.000 | 1.25 | 0.000 | 0.000 | 1.000 | 1.000 |
| naive_bayes_router | 1.000 | 1.000 | 1.000 | 1.25 | 0.000 | 0.000 | 1.000 | 1.000 |
| oracle_route_ceiling | 1.000 | 1.000 | 1.000 | 1.25 | 0.000 | 0.000 | 1.000 | 1.000 |

Phase-2 generation summary with pinned `google/flan-t5-small`:

| System | Answer F1 | Contains ref | Grounded | Evidence complete | Unsupported answer | Answerable refusal | Unanswerable refusal recall | Mean retrieval calls | Active calls | Attempts | E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_single_rag | 0.517 | 0.600 | 0.875 | 0.625 | 1.000 | 0.000 | 0.000 | 1.33 | 0.00 | 1.00 | 93.90 |
| adaptive_control | **0.717** | **0.800** | 0.875 | **1.000** | 1.000 | **0.000** | 0.000 | **1.25** | 0.00 | 1.00 | 100.43 |
| adaptive_active_reflect | **0.717** | **0.800** | 0.875 | **1.000** | **0.500** | 0.100 | **0.500** | 1.42 | 0.17 | 1.17 | 122.63 |

Important findings:

- **Routing is a first-class RAG decision.** The adaptive controller eliminates retrieval on held-out no-RAG tasks and recovers all three controlled two-hop chains while the always-single baseline under-routes every iterative query.
- **Adaptive retrieval improves downstream generation here.** Evidence completeness rises from `0.625` to `1.000`, answer F1 from `0.517` to `0.717`, and reference containment from `0.600` to `0.800`.
- **Correct routing cannot fix generator competence.** The no-RAG arithmetic query is routed correctly with zero retrieval calls, yet FLAN still answers incorrectly.
- **Complete evidence is not answer correctness.** On the Atlas → Vega → Raft query, both required documents are retrieved, but FLAN emits `[d3]` with token-probability confidence around `0.97`.
- **Active retrieval needs failure attribution.** Reflection correctly notices that `[d3]` is unsupported, but another retrieval adds unrelated context; the system then falsely refuses an answerable query. More retrieval is the wrong recovery action for a generation-format failure.
- **Reflection can reduce unsupported output without improving aggregate correctness.** The active/reflection system correctly refuses one of two deliberately unanswerable queries, reducing unsupported-answer rate from `1.0` to `0.5`, but F1 stays `0.717` and answerable false-refusal rises to `0.10`.
- **Lexical support is not semantic answer relevance.** The other unanswerable query asks for a package manager; FLAN answers `Python`, which appears in context, so the lexical support critic accepts a non-responsive answer.
- **Adaptive loops have measurable cost.** Active/reflection raises mean calls `1.25 → 1.42`, attempts `1.00 → 1.17`, and recorded end-to-end CPU latency from roughly `100 → 123 ms/query` without improving F1.
- **Perfect route accuracy is a benchmark limitation, not a production claim.** The held-out split is tiny and template-like; the synthetic `STALE:` marker and controlled fallback similarly isolate mechanism behavior rather than model real-world freshness.

Evaluation evidence:

- M06 mechanism/evaluation PR run `32416264406` completed successfully.
- Full repository suite on that tree: **70 tests passed**; phase-1 routing/correction and phase-2 pinned-FLAN generation/active/reflection evaluations both succeeded.
- Runtime never receives qrels, answer references, or answerability labels.
- Per-query failures, confidence/reflection traces, system costs, and limitations are persisted in `labs/06_adaptive_corrective/results/m06_summary.md` plus machine-readable artifacts.
- Final completion-documentation tree must pass the same M06 workflow before merge.

Artifacts: `benchmarks/m06_adaptive/`, `src/rag_practice/adaptive/`, `src/rag_practice/evaluation/adaptive.py`, `src/rag_practice/evaluation/adaptive_generation.py`, `labs/06_adaptive_corrective/`, and `.github/workflows/m06-adaptive-corrective.yml`.

### M07 — Hierarchical, Graph, and Memory-oriented RAG — `DONE`

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

- Final M07 completion gate run `32446068577` succeeded on push head `03d02d8d8492f0bffe66d1b47f19b2603aa0afb8` after the repaired workflow was active.
- The gate runs the full repository suite (**78 tests**) before the M07 hierarchy/graph/memory evaluator.
- Per-query rankings, task breakdowns, timings, structure sizes, freshness traces, route decisions, and the post-hoc caveat are persisted as JSON and Markdown.
- Generation/groundedness evaluation is **not applicable by design** because M07 intentionally isolates structured retrieval and temporal-memory quality from generation.

Artifacts: `benchmarks/m07_structured/`, `src/rag_practice/structured/`, `src/rag_practice/evaluation/structured.py`, `labs/07_hierarchical_graph_memory/`, and `.github/workflows/m07-structured.yml`.
### M08 — Specialized Sources and Modalities — `IN PROGRESS`

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
- Final source-of-truth gate `32451228304` passed before this ROADMAP update on head `ea03484c4dc898453f78c92ef3ea118159430a2c`.
- Retrieval relevance, constraint correctness, security leakage, answer quality, and candidate cost are evaluated separately.
- Persisted JSON/Markdown includes rankings, filter context, metadata, eligibility, rejection counts, answers, and latency.

Artifacts: `benchmarks/m08_metadata/`, `src/rag_practice/metadata_filter/`, `src/rag_practice/evaluation/metadata_filter.py`, `labs/08_specialized_sources/metadata/`, and `.github/workflows/m08-metadata.yml`.
### M09 — Agentic RAG — `TODO`

Implement planner, search strategy, tool/source router, retrieval loop, evidence evaluator, retry/stop policy, memory/state, then multi-agent variants. Evaluate task success, tool precision, steps, unnecessary actions, recovery, grounding, latency, and cost.

### M10 — Training and Production RAG — `TODO`

Study/implement retriever fine-tuning, hard-negative mining, learned rerankers, end-to-end RAG concepts, caching, incremental indexing, observability, permissions, adversarial retrieval defenses, freshness policies, scaling, and serving. Evaluate offline quality, system performance, robustness/security, freshness, and regressions.

## Immediate next step

Continue **M08.4 — Code RAG**. Treat code structure as first-class retrieval metadata instead of flattening a repository into anonymous text: benchmark file/function/class/symbol lookup, implementation-vs-call-site retrieval, cross-file dependency context, duplicate identifiers, and change-locality. Compare plain text retrieval with symbol-aware/chunk-aware retrieval, evaluate exact source locations and dependency evidence separately from answer quality, and measure indexing/query cost.
