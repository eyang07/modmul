"""Evaluate a submission on the public-benchmark distribution, per tier.

Goes through the real `evaluate_local` pipeline (manifest validation -> static
check -> load -> preprocess-isolation -> determinism -> inference -> decode ->
score), so a passing run also demonstrates the submission is compliant with the
static check and the structural anti-cheat checks.

Reproduces the public benchmark by seeding private generation with the public
seed at 100 problems/tier.

Usage:
    .venv312/bin/python examples/dlp_grokking/eval_tiers.py [DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

from modchallenge.config import EvalConfig, PublicBenchmarkConfig
from modchallenge.evaluation.pipeline import evaluate_local


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent
    )
    seed = PublicBenchmarkConfig().seed  # reproduces the public benchmark
    config = EvalConfig(total_problems=1100, timeout_seconds=900)
    result = evaluate_local(model_dir, master_seed=seed, config=config)

    print(f"\n=== {model_dir.name} ===")
    print(f"overall_accuracy (scored tiers 1-10): {result.overall_accuracy:.3f}")
    print(f"deterministic:    {result.deterministic}")
    print("tier accuracies:")
    for t in sorted(result.tier_results, key=lambda r: r.tier_id):
        flag = "" if t.completed else "  (incomplete)"
        print(f"  tier {t.tier_id:>2}: {t.accuracy:.3f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
