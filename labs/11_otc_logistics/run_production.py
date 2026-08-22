from __future__ import annotations

import json
from pathlib import Path

from rag_practice.evaluation.otc_production import (
    evaluate_production,
    render_production_markdown,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks" / "m11_otc_logistics"
RESULTS = ROOT / "labs" / "11_otc_logistics" / "results"


def main() -> None:
    results = evaluate_production(DATA)
    markdown = render_production_markdown(results)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "production_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True)
    )
    (RESULTS / "production_results.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
