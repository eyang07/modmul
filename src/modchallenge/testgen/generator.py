"""Test case generation with cryptographically secure seeding.

Supports both public (fixed seed, open) and private (secret seed, real-time) test sets.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from modchallenge.config import TIERS, MULT_SUB_TIERS, EvalConfig, PublicBenchmarkConfig, TierConfig
from modchallenge.testgen.primes import generate_primes_for_tier


@dataclass
class TestCase:
    """A single test case: compute a*b mod p."""

    a: str
    b: str
    p: str
    expected: str  # ground truth: (a*b) mod p
    tier_id: int

    def to_input_dict(self) -> dict:
        """Return input-only dict (no answer), for sending to sandbox."""
        return {"a": self.a, "b": self.b, "p": self.p, "tier_id": self.tier_id}

    def to_full_dict(self) -> dict:
        """Return full dict including expected answer."""
        return {
            "a": self.a,
            "b": self.b,
            "p": self.p,
            "expected": self.expected,
            "tier_id": self.tier_id,
        }


@dataclass
class TierTestSet:
    """Test cases for a single tier."""

    tier_id: int
    cases: list[TestCase] = field(default_factory=list)


@dataclass
class FullTestSet:
    """Complete test set across all tiers."""

    tiers: list[TierTestSet] = field(default_factory=list)
    seed_hex: str = ""  # hex of the master seed (empty for public sets)

    @property
    def total_cases(self) -> int:
        return sum(len(t.cases) for t in self.tiers)


def _tier0_prime_for_sub(max_bits: int) -> int:
    """Get a prime p such that p > (2^max_bits - 1)^2 for pure multiplication.

    For small sub-tiers, uses sympy.nextprime. For large sub-tiers (>= 256 bits),
    uses Mersenne primes to avoid expensive primality searches.
    """
    # Known Mersenne primes large enough for each range
    # M_p = 2^p - 1. We need 2*max_bits bits, so pick the smallest M_p > 2^(2*max_bits).
    _MERSENNE_EXPONENTS = (521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941)

    needed_bits = 2 * max_bits + 1
    for exp in _MERSENNE_EXPONENTS:
        if exp >= needed_bits:
            return (2 ** exp) - 1

    # Fallback for very large ranges: use 2^19937 - 1 (known Mersenne prime)
    return (2 ** 19937) - 1


def _generate_multiplication_cases(
    num_cases: int,
    seed: bytes,
    edge_cases: int,
) -> list[TestCase]:
    """Generate pure multiplication cases for tier 0 (unscored diagnostic).

    Cases are evenly distributed across sub-tiers by operand bit size.
    Each sub-tier uses a prime p > max_product so that a*b mod p = a*b.
    """
    from sympy import nextprime

    tier_seed = hmac.new(seed, b"cases-tier-0-mult", hashlib.sha256).digest()
    rng = random.Random(tier_seed)

    cases: list[TestCase] = []
    num_sub_tiers = len(MULT_SUB_TIERS)
    per_sub = num_cases // num_sub_tiers
    remainder = num_cases % num_sub_tiers

    # Precompute one prime per sub-tier: p > (2^max_bits - 1)^2
    sub_primes: list[int] = []
    for _, max_bits in MULT_SUB_TIERS:
        if max_bits <= 128:
            max_product = (2 ** max_bits - 1) ** 2
            sub_primes.append(int(nextprime(max_product)))
        else:
            sub_primes.append(_tier0_prime_for_sub(max_bits))

    for sub_idx, (min_bits, max_bits) in enumerate(MULT_SUB_TIERS):
        sub_count = per_sub + (1 if sub_idx < remainder else 0)
        lo = 2 ** (min_bits - 1) if min_bits > 1 else 0
        hi = 2 ** max_bits
        p = sub_primes[sub_idx]

        for j in range(sub_count):
            # Edge cases for first sub-tier
            if sub_idx == 0 and j < edge_cases:
                edge_ops = [(0, None), (None, 0), (1, None), (None, 1)]
                a_fixed, b_fixed = edge_ops[j % len(edge_ops)]
                a = a_fixed if a_fixed is not None else rng.randrange(lo, hi)
                b = b_fixed if b_fixed is not None else rng.randrange(lo, hi)
            else:
                a = rng.randrange(lo, hi)
                b = rng.randrange(lo, hi)

            product = a * b
            expected = product  # a*b < p guaranteed, so a*b mod p = a*b

            cases.append(
                TestCase(
                    a=str(a), b=str(b), p=str(p),
                    expected=str(expected), tier_id=0,
                )
            )

    return cases


def _generate_tier_cases(
    tier: TierConfig,
    num_cases: int,
    num_primes: int,
    edge_cases: int,
    seed: bytes,
) -> list[TestCase]:
    """Generate test cases for a single modular arithmetic tier (tiers 1-10)."""
    # Derive per-tier seed
    tier_seed = hmac.new(seed, f"cases-tier-{tier.tier_id}".encode(), hashlib.sha256).digest()
    rng = random.Random(tier_seed)

    primes = generate_primes_for_tier(tier, num_primes, seed)
    operand_bits = tier.operand_bits if tier.operand_bits > 0 else tier.max_bits
    max_val = 2 ** operand_bits  # a, b range: [0, 2^operand_bits)

    cases: list[TestCase] = []

    # Edge cases: a=0, b=0, a=1, b=1
    edge_count = min(edge_cases, num_cases)
    edge_ops = [(0, None), (None, 0), (1, None), (None, 1)]
    for i in range(edge_count):
        p = primes[i % len(primes)]
        a_fixed, b_fixed = edge_ops[i % len(edge_ops)]
        a = a_fixed if a_fixed is not None else rng.randrange(0, max_val)
        b = b_fixed if b_fixed is not None else rng.randrange(0, max_val)
        expected = (a * b) % p
        cases.append(
            TestCase(
                a=str(a), b=str(b), p=str(p),
                expected=str(expected), tier_id=tier.tier_id,
            )
        )

    # Random cases for the rest
    for _ in range(num_cases - edge_count):
        p = primes[rng.randrange(len(primes))]
        a = rng.randrange(0, max_val)
        b = rng.randrange(0, max_val)
        expected = (a * b) % p
        cases.append(
            TestCase(
                a=str(a), b=str(b), p=str(p),
                expected=str(expected), tier_id=tier.tier_id,
            )
        )

    return cases


def generate_private_test_set(
    master_seed: bytes | None = None,
    config: EvalConfig = EvalConfig(),
) -> FullTestSet:
    """Generate a private test set with a secret master seed.

    If master_seed is None, a fresh cryptographically random seed is created.
    The seed is never exposed to contestants.
    """
    if master_seed is None:
        master_seed = secrets.token_bytes(32)

    tiers = []
    for tier_cfg in TIERS:
        if tier_cfg.is_multiplication_only:
            cases = _generate_multiplication_cases(
                num_cases=config.problems_per_tier,
                seed=master_seed,
                edge_cases=config.edge_cases_per_tier,
            )
        else:
            cases = _generate_tier_cases(
                tier=tier_cfg,
                num_cases=config.problems_per_tier,
                num_primes=config.primes_per_tier,
                edge_cases=config.edge_cases_per_tier,
                seed=master_seed,
            )
        tiers.append(TierTestSet(tier_id=tier_cfg.tier_id, cases=cases))

    return FullTestSet(tiers=tiers, seed_hex=master_seed.hex())


def generate_public_test_set(
    config: PublicBenchmarkConfig = PublicBenchmarkConfig(),
) -> FullTestSet:
    """Generate the public benchmark test set (fixed seed, fully open)."""
    seed = config.seed
    tiers = []
    for tier_cfg in TIERS:
        if tier_cfg.is_multiplication_only:
            cases = _generate_multiplication_cases(
                num_cases=config.problems_per_tier,
                seed=seed,
                edge_cases=4,
            )
        else:
            cases = _generate_tier_cases(
                tier=tier_cfg,
                num_cases=config.problems_per_tier,
                num_primes=5,
                edge_cases=4,
                seed=seed,
            )
        tiers.append(TierTestSet(tier_id=tier_cfg.tier_id, cases=cases))

    return FullTestSet(tiers=tiers)


def write_test_inputs(test_set: FullTestSet, output_dir: Path) -> None:
    """Write test inputs (no answers) to per-tier JSONL files.

    These files are what gets mounted into the sandbox.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for tier_ts in test_set.tiers:
        path = output_dir / f"tier_{tier_ts.tier_id}_input.jsonl"
        with open(path, "w") as f:
            for case in tier_ts.cases:
                f.write(json.dumps(case.to_input_dict()) + "\n")


def write_test_full(test_set: FullTestSet, output_dir: Path) -> None:
    """Write full test set (with answers) for scoring or public benchmark.

    Only used server-side for scoring, or for the public benchmark.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for tier_ts in test_set.tiers:
        path = output_dir / f"tier_{tier_ts.tier_id}.jsonl"
        with open(path, "w") as f:
            for case in tier_ts.cases:
                f.write(json.dumps(case.to_full_dict()) + "\n")
