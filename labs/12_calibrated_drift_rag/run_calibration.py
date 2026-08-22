from __future__ import annotations

import json
from pathlib import Path

from rag_practice.evaluation.calibration import evaluate_calibration, render_calibration_markdown


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks" / "m12_calibration"
RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    results = evaluate_calibration(DATA)
    markdown = render_calibration_markdown(results)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "calibration_results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    (RESULTS / "calibration_results.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
