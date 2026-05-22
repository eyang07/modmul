"""Global configuration for the evaluation system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TierConfig:
    """Configuration for a single difficulty tier."""

    tier_id: int
    min_bits: int       # prime p bit range
    max_bits: int
    operand_bits: int = 0  # max bits for a, b (0 = same as max_bits)
    fixed_primes: tuple[int, ...] = ()  # For tier 1: explicit small primes
    is_multiplication_only: bool = False  # Tier 0: pure multiplication, unscored


# Tier 0 sub-tier bit ranges (pure multiplication diagnostic, 10 sub-levels).
# Covers the full operand range of scored tiers so that when a model fails
# at high tiers, Tier 0 results can distinguish "multiplication broke" from
# "modular reduction broke".
MULT_SUB_TIERS: tuple[tuple[int, int], ...] = (
    (1, 4),       # ~1 digit
    (5, 16),      # ~2-5 digits
    (17, 32),     # ~5-10 digits
    (33, 64),     # ~10-19 digits
    (65, 128),    # ~19-39 digits
    (129, 256),   # ~39-77 digits
    (257, 512),   # ~77-154 digits
    (513, 1024),  # ~154-309 digits
    (1025, 2048), # ~309-617 digits
    (2049, 4096), # ~617-1233 digits
)

# 11 tiers: tier 0 = pure multiplication (unscored), tiers 1-10 = modular arithmetic (scored).
# operand_bits controls a, b range independently from p. For lower tiers,
# a and b can be much larger than p, testing the model's ability to do
# large multiplication followed by modular reduction.
TIERS: tuple[TierConfig, ...] = (
    TierConfig(tier_id=0, min_bits=1, max_bits=4096, is_multiplication_only=True),
    TierConfig(tier_id=1, min_bits=1, max_bits=3, fixed_primes=(2, 3, 5, 7), operand_bits=32),
    TierConfig(tier_id=2, min_bits=4, max_bits=8, operand_bits=48),
    TierConfig(tier_id=3, min_bits=9, max_bits=16, operand_bits=64),
    TierConfig(tier_id=4, min_bits=17, max_bits=32, operand_bits=96),
    TierConfig(tier_id=5, min_bits=33, max_bits=64, operand_bits=128),
    TierConfig(tier_id=6, min_bits=65, max_bits=128, operand_bits=256),
    TierConfig(tier_id=7, min_bits=129, max_bits=256, operand_bits=512),
    TierConfig(tier_id=8, min_bits=257, max_bits=512, operand_bits=1024),
    TierConfig(tier_id=9, min_bits=513, max_bits=1024, operand_bits=2048),
    TierConfig(tier_id=10, min_bits=1025, max_bits=2048, operand_bits=4096),
)

NUM_TIERS = len(TIERS)  # 11
NUM_SCORED_TIERS = NUM_TIERS - 1  # 10 (tiers 1-10)


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation run configuration."""

    total_problems: int = 1100
    primes_per_tier: int = 5
    edge_cases_per_tier: int = 4  # a=0, b=0, a=1, b=1
    timeout_seconds: int = 300  # 5 minutes total
    max_artifact_bytes: int = 20 * 1024 * 1024 * 1024  # 20 GB
    skip_static_check: bool = False  # set True for trusted-use during development

    def __post_init__(self) -> None:
        if self.total_problems % NUM_TIERS != 0:
            raise ValueError(
                f"total_problems ({self.total_problems}) must be divisible by "
                f"NUM_TIERS ({NUM_TIERS})"
            )

    @property
    def problems_per_tier(self) -> int:
        return self.total_problems // NUM_TIERS


@dataclass(frozen=True)
class PublicBenchmarkConfig:
    """Public benchmark configuration (fixed seed, fully open).

    This is the standardized public benchmark: fixed seed, answers included,
    available to all contestants for local testing and cross-model comparison.
    Private evaluation uses a secret random seed generated at eval time.
    """

    seed: bytes = b"modchallenge-public-benchmark-v1"
    problems_per_tier: int = 100
