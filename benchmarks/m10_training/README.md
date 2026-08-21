# M10.1 frozen retrieval-training benchmark

Status: **FROZEN BEFORE FINE-TUNING**

This benchmark is the first M10 training control. It is frozen before implementing or inspecting any fine-tuned retriever result.

## Hypothesis

Domain fine-tuning should improve or at least measurably change retrieval on terminology mismatches without using test labels. Explicit hard negatives should be evaluated separately from pair-only fine-tuning rather than assumed to help.

## Split design

`dataset.json` contains three disjoint entity splits. Each split has eight documents and eight single-relevance queries. Test entities never appear in train or dev.

The benchmark repeats four domain terminology mismatches across different entities:

- `restore lane` in queries vs `rollback channel` in documents;
- `health-state reference` vs `service status`;
- `dispatch window` vs `shipping SLA`;
- `supply key` vs `inventory code`.

Each entity has an intent-confuser document so entity-name overlap alone is insufficient. This intentionally creates candidates where hard-negative selection can focus on the wrong-intent document for the same entity.

## Frozen systems

1. **Pinned pretrained baseline** — `sentence-transformers/all-MiniLM-L6-v2@1c82ace116a2629de82404c4be48c0e5d4cf08be`, no training.
2. **Pair-only fine-tune** — same checkpoint, fixed two-epoch in-batch softmax training on train positive pairs only.
3. **Hard-negative fine-tune** — same checkpoint and hyperparameters, plus exactly one explicit negative per train query. The negative is mined by the untouched pinned baseline as the highest-ranked non-positive TRAIN document.

Hard-negative mining may use train query text, train documents, and the train positive id only to exclude the positive. It must not score or inspect dev/test documents or qrels.

## Frozen optimization contract

- seed: `23`
- epochs: `2`
- batch size: `4`
- learning rate: `2e-5`
- temperature: `0.05`
- max sequence length: `128`
- CPU-compatible float32 execution
- no hyperparameter selection after seeing fine-tuned test results

Dev metrics are diagnostic only for this frozen run; they are not a license to choose a different test-facing configuration after inspection.

## Evaluation contract

Persist separately for baseline, pair-only, and hard-negative systems:

- Recall@1 and Recall@3;
- MRR;
- mean relevant-vs-best-negative score margin;
- per-query ranking and scores;
- metrics by terminology class;
- model load, training, index-build, and query latency;
- parameter count / parameter bytes;
- mined hard-negative ids and baseline ranks;
- training loss history.

Negative results are valid results. Do not modify the frozen data, qrels, model revision, mining rule, or training hyperparameters to make fine-tuning look beneficial.

## Integrity boundary

The freeze commit containing this README is the benchmark boundary. Any implementation commit must be a descendant of that freeze. If a transport/runtime bug is found later, repair the implementation while preserving documents, queries, qrels, split membership, terminology mapping, model revision, optimizer hyperparameters, and mining policy.
