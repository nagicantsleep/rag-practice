from __future__ import annotations

import json
from pathlib import Path

from rag_practice.evaluation.otc_integrated import evaluate_integrated, render_integrated_markdown


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks" / "m11_otc_logistics"
OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = evaluate_integrated(DATA, split="test")
    (OUT / "integrated_results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    (OUT / "integrated_results.md").write_text(render_integrated_markdown(results))
    print(json.dumps(results["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
