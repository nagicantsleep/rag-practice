from __future__ import annotations

import json
from pathlib import Path

from rag_practice.evaluation.otc import evaluate_baselines, render_markdown


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "benchmarks" / "m11_otc_logistics"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    results = evaluate_baselines(DATA_ROOT, split="test")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "baseline_results.json"
    md_path = RESULTS_DIR / "baseline_results.md"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    markdown = render_markdown(results)
    md_path.write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
