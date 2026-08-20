# M05 Completion Summary — Query Transformation

## Scope and hypothesis

M05 freezes the corpus and retriever family within each comparison and changes only the query-side representation/control. It implements and evaluates single-query rewrite, multi-query score fusion, RAG-Fusion/RRF, Query2Doc-style expansion, HyDE, and decomposition.

Hypothesis: generative transformations may help vocabulary-mismatch and multi-aspect information needs, but can also drift from the original intent. Therefore every method is judged against an original-query baseline using the same retriever family, with transformation outputs and costs persisted for inspection.

## Experimental controls

- Corpus: `benchmarks/m00_ir/corpus.jsonl`.
- Held-out benchmark: 12 queries in `benchmarks/m05_query_transform/queries.jsonl`.
- Query classes: `exact`, `semantic`, `underspecified`, `multi_aspect`.
- BM25-family methods are compared only with `bm25_original`.
- HyDE is compared only with `dense_original` using pinned `sentence-transformers/all-MiniLM-L6-v2`.
- Multi-query retains the original query as one member.
- Query2Doc retains the original query plus generated pseudo-document text.
- Qrels/relevant-document IDs are never exposed to the query transformer.
- `complete_recall@3` requires every relevant document in top 3, preventing a one-of-two multi-aspect hit from appearing complete.
- No prompt tuning was performed after inspecting held-out results.

## FLAN-T5-small baseline

Transformer: `google/flan-t5-small` @ `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`.

| Method | R@1 | R@3 | Complete R@3 | Semantic R@1 | Multi-aspect R@3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 original | 0.792 | 0.875 | 0.833 | 0.667 | 0.833 |
| Rewrite | 0.792 | 0.875 | 0.833 | 0.667 | 0.833 |
| Multi-query score fusion | 0.792 | 0.875 | 0.833 | 0.667 | 0.833 |
| RAG-Fusion / RRF | 0.792 | 0.875 | 0.833 | 0.667 | 0.833 |
| Query2Doc + BM25 | 0.792 | 0.875 | 0.833 | 0.667 | 0.833 |
| Decomposition + RRF | 0.167 | 0.250 | 0.167 | 0.000 | 0.333 |
| Dense original | 0.792 | 0.917 | 0.917 | 0.667 | 1.000 |
| HyDE + dense | 0.625 | 0.833 | 0.750 | 0.333 | 0.667 |

The simple transformations did not improve BM25 aggregate retrieval. Decomposition failed severely, and HyDE regressed relative to the original dense query.

## Capacity control: FLAN-T5-base

To distinguish method failure from insufficient transformer capacity, the exact same prompts, queries, retrievers, index, and evaluation were rerun with only the transformer changed to `google/flan-t5-base` @ `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`.

| Method | R@1 | R@3 | Complete R@3 | Semantic R@1 | Multi-aspect R@3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 original | 0.792 | 0.875 | 0.833 | 0.667 | 0.833 |
| Rewrite | 0.792 | 0.875 | 0.833 | 0.667 | 0.833 |
| Multi-query score fusion | 0.792 | 0.875 | 0.833 | 0.667 | 0.833 |
| RAG-Fusion / RRF | 0.792 | 0.875 | 0.833 | 0.667 | 0.833 |
| Query2Doc + BM25 | 0.792 | 0.833 | 0.750 | 0.667 | 0.667 |
| Decomposition + RRF | 0.625 | 0.667 | 0.667 | 0.333 | 0.333 |
| Dense original | 0.792 | 0.917 | 0.917 | 0.667 | 1.000 |
| HyDE + dense | 0.708 | 0.917 | 0.917 | 0.333 | 1.000 |

Capacity helps some mechanisms: decomposition improves from R@1 `0.167` to `0.625`, and HyDE improves from `0.625` to `0.708`. However, neither beats its fair original-query baseline. Rewrite, multi-query, and RAG-Fusion remain quality-neutral while adding generation cost, and Query2Doc now reduces R@3/complete recall.

## Error analysis

### Vocabulary mismatch is not automatically fixed by rewriting

The preserved semantic case `conceptual likeness between paraphrases` still fails under BM25 rewrite, multi-query, RAG-Fusion, and Query2Doc. The pretrained dense original query retrieves the correct dense-retrieval document at rank 1. This is evidence that a representation mismatch cannot always be repaired by surface-form query generation.

### Decomposition is only useful when decomposition output is reliable

FLAN-T5-small frequently echoes instructions or emits unusable outputs. Increasing capacity removes some of those failures, but FLAN-T5-base still produces examples such as code-like repeated `a=[] for i in range(...)`, alphabet-like text, and repeated `__author__ = "samuel"` strings on held-out semantic/multi-aspect queries. Those outputs are retained in the JSON report instead of filtered after seeing qrels.

### HyDE can move the query away from useful evidence

HyDE occasionally creates a document whose semantic neighborhood is worse than the original question. The `conceptual likeness between paraphrases` case is especially clear: the original MiniLM query ranks the relevant document first, while both small/base hypothetical documents push unrelated documents above it. FLAN-T5-base does recover complete top-3 recall overall, but its rank-1 recall remains below the original dense query.

### Multi-aspect metrics need completeness, not just hit rate

Some methods retrieve one of two required documents while missing the other. `complete_recall@3` exposes these failures even when ordinary hit-rate is 1.0. On the base capacity control, Query2Doc drops multi-aspect R@3 from `0.833` to `0.667` and complete recall from `0.833` to `0.750` overall.

## Cost and latency

Absolute timings are GitHub Actions CPU sanity measurements, not production serving estimates, but the ordering is unambiguous: query generation dominates retrieval cost. BM25 original is roughly hundredths of a millisecond/query and MiniLM original-query search is around ten milliseconds/query in the persisted runs, while generative transformations generally add hundreds to thousands of milliseconds. The base model is materially slower than the small model.

This means a transformation should earn its latency/cost through measurable retrieval gains or be invoked conditionally; applying it to every query is not justified by this benchmark.

## Findings

1. **Query transformation is not a free quality upgrade.** On this controlled benchmark, no generative method beats its fair original-query baseline on aggregate.
2. **Keep the original query when fusing variants.** This protects good exact/underspecified queries from some rewrite drift, but it does not guarantee a gain.
3. **Generator capacity matters, but reliability matters too.** FLAN-T5-base substantially improves decomposition and HyDE compared with FLAN-T5-small, yet pathological outputs and regressions remain.
4. **Dense representation can solve vocabulary mismatch more directly than lexical rewriting.** The preserved M00 paraphrase failure is still best handled by pretrained dense retrieval here.
5. **Decomposition should be routed, not universal.** Simple questions do not need it, and poor subquestions can destroy retrieval. This finding directly motivates M06 adaptive routing and corrective control.
6. **HyDE needs a strong, domain-appropriate generator and evaluation.** A plausible hypothetical document can still be retrieval-worse than the original question.
7. **Cost belongs in the decision rule.** Quality-neutral transformations that add hundreds of milliseconds should not be default behavior.
8. **Negative results satisfy the learning contract when they are controlled and reproducible.** The milestone is complete because the mechanisms, baselines, benchmark, metrics, costs, failures, and capacity control are all recorded—not because a new method was forced to win.

## Limitations

- The corpus and held-out benchmark are intentionally tiny and hand-checkable.
- FLAN-T5-small/base are educational instruction-model controls, not claims about frontier query-rewriting systems.
- The prompts are deliberately simple and were not optimized against the held-out set.
- CPU timings are implementation sanity measurements and should not be generalized to production hardware or hosted APIs.
- These results establish mechanism behavior and failure modes; they are not a universal ranking of query-transformation techniques.

## Evaluation evidence

- GitHub Actions M05 capacity-control PR run `32413220357` completed successfully.
- Final capacity-control run on the mechanism tree: **59 tests passed**; FLAN-T5-small baseline and FLAN-T5-base capacity-control evaluation both succeeded.
- Model revisions, per-query transformations, rankings, class metrics, latency, and generated-word counts are persisted in JSON/Markdown artifacts.
- The final documentation/ROADMAP tree still requires one final CI pass before merge.

Artifacts: `benchmarks/m05_query_transform/`, `src/rag_practice/query_transform/`, `labs/05_query_transform/`, and `.github/workflows/m05-query-transform.yml`.
