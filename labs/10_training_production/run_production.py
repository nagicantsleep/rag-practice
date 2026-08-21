from __future__ import annotations

import json
from pathlib import Path

from rag_practice.evaluation.production import (
    evaluate_scale,
    evaluate_serving_system,
    load_production_benchmark,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "m10_production" / "scenarios.json"
RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_JSON = RESULT_DIR / "production_results.json"
RESULT_MD = RESULT_DIR / "production_results.md"


def main() -> None:
    payload = load_production_benchmark(BENCHMARK)
    unsafe = evaluate_serving_system(payload, guarded=False)
    guarded = evaluate_serving_system(payload, guarded=True)
    scale = evaluate_scale(payload)
    result = {
        "experiment_id": "m10_production_serving_v1",
        "benchmark": "benchmarks/m10_production@v1",
        "unsafe_baseline": unsafe,
        "guarded": guarded,
        "scale": scale,
        "guardrail": "Production correctness/security metrics are independent from offline retrieval quality and latency.",
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    RESULT_MD.write_text(
        "# M10.2 production-serving results\n\n"
        "| System | Scenario accuracy | Cache expectation | Invalidation | No-evidence | Unauthorized exposure | Stale exposure | Untrusted exposure | Observability |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| unsafe baseline | {unsafe['scenario_result_accuracy']:.3f} | {unsafe['cache_expectation_accuracy']:.3f} | {unsafe['cache_invalidation_accuracy']:.3f} | {unsafe['no_evidence_accuracy']:.3f} | {unsafe['unauthorized_exposure_rate']:.3f} | {unsafe['stale_exposure_rate']:.3f} | {unsafe['untrusted_exposure_rate']:.3f} | {unsafe['observability_completeness']:.3f} |\n"
        f"| guarded | {guarded['scenario_result_accuracy']:.3f} | {guarded['cache_expectation_accuracy']:.3f} | {guarded['cache_invalidation_accuracy']:.3f} | {guarded['no_evidence_accuracy']:.3f} | {guarded['unauthorized_exposure_rate']:.3f} | {guarded['stale_exposure_rate']:.3f} | {guarded['untrusted_exposure_rate']:.3f} | {guarded['observability_completeness']:.3f} |\n\n"
        "## Scale sanity\n\n"
        "| Documents | Hit@1 | Build ms | Query ms | Upsert ms | Delete ms | Posting entries |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(
            f"| {item['size']} | {int(item['target_hit@1'])} | {item['build_ms']:.3f} | {item['query_ms']:.3f} | {item['upsert_ms']:.3f} | {item['delete_ms']:.3f} | {item['posting_entries']} |"
            for item in scale
        )
        + "\n\nTimings are GitHub Actions CPU implementation sanity measurements, not production throughput claims.\n"
    )
    print(json.dumps({
        "unsafe": {k: v for k, v in unsafe.items() if not isinstance(v, list)},
        "guarded": {k: v for k, v in guarded.items() if not isinstance(v, list)},
        "scale": scale,
    }, indent=2))


if __name__ == "__main__":
    main()
