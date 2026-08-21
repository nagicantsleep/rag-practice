"""Run the frozen M08.7 deterministic routing evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from rag_practice.evaluation.long_context import format_mechanism_markdown, mechanism_suite
from rag_practice.long_context import load_benchmark

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "m08_long_context" / "benchmark.json"
RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    benchmark = load_benchmark(BENCHMARK)
    result = mechanism_suite(benchmark)
    markdown = format_mechanism_markdown(result)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (RESULTS / "results.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
