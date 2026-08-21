# M09 frozen agentic RAG benchmark

This benchmark is frozen before implementing or inspecting any pretrained agent policy.

## Hypothesis

Agentic control should earn its extra actions by selecting the right source/tool, composing evidence across tools, recovering from explicit misses, and stopping when evidence is absent. A loop that merely calls more tools is not better RAG.

## Frozen tool contract

Four deterministic tools expose disjoint evidence boundaries:

- `docs_search(query)` — BM25 over five short documents, returning top 2 positive-scoring documents.
- `inventory_lookup(key)` — exact structured lookup returning a stock count or `NOT_FOUND`.
- `status_lookup(key)` — exact structured lookup returning a service status or `NOT_FOUND`.
- `calculator(expression)` — integer addition only for expressions of the form `a + b`; invalid expressions return `ERROR`.

Each tool call has a frozen cost unit: docs `2.0`, inventory `1.0`, status `1.0`, calculator `0.5`. The single-agent action budget is `4` calls and the retry budget is `1` recovery transition after an explicit `NOT_FOUND`/insufficient-evidence observation. `ABSTAIN` is the only no-evidence final answer.

## Tasks

Twelve frozen tasks cover:

- direct document facts;
- direct structured lookup;
- direct calculation;
- document → structured joins;
- document → two structured lookups → calculation;
- two structured lookups for comparison;
- deliberate first-tool miss followed by source recovery;
- no-evidence with and without a failed structured lookup;
- one multi-source answer requiring both document and status evidence.

Each task stores expected answer/action/evidence fields for evaluation only. Runtime planners, tools, readers, critics, and stop policies must not inspect them.

## Integrity rules

- Do not change tool corpora, costs, action/retry budgets, task text, expected answers, expected actions, evidence ids, recovery labels, or abstention labels after agent implementation or pretrained inspection.
- Evaluate final task success separately from tool/action quality.
- Persist every action, argument, observation, evidence id, recovery transition, final answer, latency, and cost.
- A failed tool call can be part of the frozen expected recovery path; do not silently delete it from traces.
- Grounding for calculator/comparison answers must be derived from recorded tool observations, not answer-string coincidence.
- Phase 1 uses transparent deterministic policies. Any pretrained/single-model or multi-agent control must run on this unchanged benchmark and retain negative cases.
