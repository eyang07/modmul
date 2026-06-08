# always_zero

Trivial baseline that emits `[0]` for every problem.

## Score (public benchmark, seed = deadbeef × 8)

| Total problems | overall_accuracy | highest_tier_above_90 |
|---|---|---|
| 110 | 0.240 | -1 |
| **1100** | **0.079** | -1 |

Per-tier at total=1100:

| Tier | Accuracy |
|------|----------|
| 0 | 0.020 |
| 1 | **0.580** |
| 2 | 0.050 |
| 3-10 | 0.020 each |

Tier 1's 58% reflects only 4 fixed primes (2, 3, 5, 7) — small enough that random `a*b mod p == 0` happens often. Other tiers' ~2% comes from the a=0 / b=0 edge cases the test generator includes in every tier.

## Purpose

Smoke test for the pipeline (manifest validation → static check → load → preprocess → predict_digits → decode → score). Also a floor reference for the leaderboard: any submission that doesn't beat this is broken.
