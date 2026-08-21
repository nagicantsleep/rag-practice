"""Run the frozen M08.7 pinned SmolLM2 reader evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from rag_practice.evaluation.long_context_pretrained import (
    format_pretrained_markdown,
    pretrained_suite,
)
from rag_practice.long_context import load_benchmark
from rag_practice.long_context.smollm import SmolLM2ContextReader

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "m08_long_context" / "benchmark.json"
RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    benchmark = load_benchmark(BENCHMARK)
    reader = SmolLM2ContextReader()
    result = pretrained_suite(benchmark, reader)
    markdown = format_pretrained_markdown(result)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "smollm_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (RESULTS / "smollm_results.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
