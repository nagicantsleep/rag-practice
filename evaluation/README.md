# Retrieval evaluation

This milestone evaluates retrieval **before** generation. The goal is to know whether the system found the evidence the LLM needs.

## Dataset format

Use JSON Lines (`.jsonl`): one JSON object per question.

```json
{"id":"rag-definition","question":"What is retrieval-augmented generation?","relevant_passages":[{"source":"data/raw/rag-notes.md","contains":"retrieval-augmented generation combines retrieval"}]}
```

Each `relevant_passages` item has:

- `source`: the document path used during indexing. Relative paths are allowed and match the suffix of an absolute indexed path.
- `contains`: a short evidence anchor that must occur inside a retrieved chunk.

Using a passage anchor instead of only a filename matters: retrieving the correct document but the wrong chunk should not count as success.

Copy the example and replace it with questions/evidence from your own documents:

```bash
cp evaluation/dataset.example.jsonl evaluation/dataset.jsonl
```

## Metrics

### Recall@k

For one question:

```text
number of relevant passages found in top-k
------------------------------------------
number of relevant passages in ground truth
```

The CLI reports macro Recall@k: the mean across questions.

### MRR

MRR (Mean Reciprocal Rank) rewards putting the first relevant chunk near the top.

```text
first relevant chunk at rank 1 -> 1.0
first relevant chunk at rank 2 -> 0.5
first relevant chunk at rank 5 -> 0.2
no relevant chunk              -> 0.0
```

### HitRate@k

The fraction of questions for which at least one relevant passage appears in the top-k results.

## Run

Index your corpus first:

```bash
rag-practice index data/raw --chunk-size 900 --overlap 150
```

Then evaluate:

```bash
rag-practice eval-retrieval evaluation/dataset.jsonl --top-k 5 --show-misses
```

Run the metric unit tests without Qdrant or Ollama:

```bash
python -m unittest discover -s tests
```

## First experiment

Keep the dataset fixed and compare configurations such as:

```text
A: chunk_size=400, overlap=80,  top_k=5
B: chunk_size=900, overlap=150, top_k=5
C: chunk_size=1400, overlap=200, top_k=5
D: chunk_size=900, overlap=150, top_k=10
```

For each run, record Recall@k, MRR, HitRate@k, and the missed questions. Do not change the evaluation dataset while comparing configurations.

A good retrieval experiment changes one variable at a time and inspects misses, not only aggregate scores.
